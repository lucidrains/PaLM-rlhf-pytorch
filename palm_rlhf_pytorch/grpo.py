"""
GRPO based training logic - https://arxiv.org/abs/2402.03300
"""

from __future__ import annotations
from typing import Callable, Deque

from pathlib import Path
from functools import partial
from collections import deque, namedtuple
from random import randrange

import torch
from torch import nn, Tensor, is_tensor, tensor, einsum
from torch.nn import Module
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from adam_atan2_pytorch import AdoptAtan2

from palm_rlhf_pytorch.palm import PaLM
from palm_rlhf_pytorch.reward import RewardModel
from palm_rlhf_pytorch.utils import masked_mean, eval_decorator, log

from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm

from torch_einops_utils import (
    cast_tensor,
    shift,
    entropy
)

from einx import get_at, add, multiply
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# einstein notation

# b - batch
# n - sequence
# d - feature dimension
# l - logits

# grpo based training

# critic completely replaced with monte carlo sampling from actor + reward model
# https://www.youtube.com/watch?v=bAWV_yrqx4w

GRPOActionReturn = namedtuple('GRPOActionReturn', [
    'actions',
    'sequence',
    'mask',
    'prompt_mask',
    'action_logits',
])

class Actor(Module):
    def __init__(
        self,
        palm: PaLM,
        actor_lora = True,
        actor_lora_r = 8,
        actor_lora_scope = 'actor',
        actor_dropout = 0.,
    ):
        super().__init__()
        self.actor_palm = palm

        self.actor_palm.set_dropout(actor_dropout)

        self.actor_lora = actor_lora

        self.actor_lora_scope = actor_lora_scope if actor_lora else None

        if self.actor_lora:
            self.actor_palm.add_finetune_params(actor_lora_scope, lora_r = actor_lora_r)

    def parameters(self):
        if not self.actor_lora:
            return self.actor_palm.parameters()

        return [
            *self.actor_palm.finetune_parameters(self.actor_lora_scope)
        ]

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        state,
        max_seq_len,
        eos_token = None,
        **kwargs
    ):
        actions = self.actor_palm.generate(
            max_seq_len,
            prompt = state,
            eos_token = eos_token,
            finetune_scope = self.actor_lora_scope,
            use_tqdm = True,
            **kwargs
        )

        sequence = torch.cat((state, actions), dim = -1)
        action_len = actions.shape[-1]
        state_len = state.shape[-1]

        prompt_mask = torch.arange(sequence.shape[-1], device = state.device) < state_len
        prompt_mask = repeat(prompt_mask, 'n -> b n', b = sequence.shape[0])

        action_mask = ~prompt_mask

        mask = None
        if exists(eos_token):
            mask = ((sequence == eos_token).cumsum(dim = -1) == 0)
            mask = F.pad(mask, (1, -1), value = True) # include eos token
            action_mask &= mask

        action_logits = self.forward(
            sequence,
            mask = action_mask,
        )

        return GRPOActionReturn(
            actions,
            sequence,
            mask,
            prompt_mask,
            action_logits
        )

    def forward(
        self,
        x,
        mask = None,
    ):
        return self.actor_palm(x, finetune_scope = self.actor_lora_scope)

# data

Memory = namedtuple('Memory', [
    'sequence',
    'prompt_mask',
    'mask',
    'action_prob',
    'action_log_prob',
    'group_relative_normalized_reward',
])

class ExperienceDataset(Dataset):
    def __init__(
        self,
        data,
        device = None
    ):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind].to(self.device), self.data))

def create_dataloader(data, batch_size, shuffle = True, device = None, **kwargs):
    ds = ExperienceDataset(data, device = device)
    return DataLoader(ds, batch_size = batch_size, shuffle = shuffle, **kwargs)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def first(x):
    return x[0]

def divisible_by(num, den):
    return (num % den) == 0

def pad_sequence_fixed(sequences, *args, **kwargs):
    first_el = sequences[0]
    has_no_dimension = first_el.ndim == 0

    # if no dimensions, add a single dimension
    if has_no_dimension:
        sequences = tuple(map(lambda t: t[None], sequences))

    out = pad_sequence(sequences, *args, **kwargs)

    if not has_no_dimension:
        return out

    return rearrange(out, '... 1 -> ...')

def safe_div(num, den, eps = 1e-8):
    den = cast_tensor(den)
    assert (den >= 0).all()
    return num / den.clamp(min = eps)

def masked_kl_div(prob1, prob2, mask = None, reduce_batch = False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(dim = -1)
    loss = masked_mean(kl_divs, mask)

    if not reduce_batch:
        return loss

    return loss.mean()

# score centering - Marek & Ryabinin (2026) - https://arxiv.org/abs/2609.20807
# cancels systematic drift caused by training-inference mismatch

def score_centering_loss(
    train_log_probs: Tensor,
    samp_probs: Tensor,
    actions: Tensor,
    advantages: Tensor,
    topk: int | None = None,
    weight_fn: Callable = torch.ones_like,
    eps: float = 1e-6,
) -> Tensor:
    l = train_log_probs.shape[-1]

    advantages, _ = pack([advantages], 'b *')

    samp_probs = samp_probs.detach()

    # sampled action loss

    action_log_prob = get_at('b n [l], b n -> b n', train_log_probs, actions)
    action_samp_prob = get_at('b n [l], b n -> b n', samp_probs, actions)

    ratio = (action_log_prob.exp() / action_samp_prob.clamp(min = eps)).detach()
    weighted_action_log_prob = weight_fn(ratio) * action_log_prob

    # head tokens (full vocabulary or top-k)

    is_topk = exists(topk) and topk < l

    head_probs, head_log_probs = samp_probs, train_log_probs

    if is_topk:
        head_probs, head_ids = samp_probs.topk(topk, dim = -1)
        head_log_probs = get_at('b n [l], b n k -> b n k', train_log_probs, head_ids)

    head_train_probs = head_log_probs.exp()
    residual = head_probs * weight_fn(head_train_probs / head_probs.clamp(min = eps))

    # model sampler tail with trainer tail if top-k

    if is_topk:
        samp_tail = (1. - head_probs.sum(dim = -1)).clamp(min = eps)
        train_tail = (1. - head_train_probs.sum(dim = -1)).clamp(min = eps)
        tail_ratio = train_tail / samp_tail
        tail_scale = weight_fn(tail_ratio) / tail_ratio
        residual = residual - multiply('b n, b n k -> b n k', tail_scale, head_train_probs)

    # stop gradient through centering coefficients - eq. 8, 14

    centered_score = weighted_action_log_prob - einsum('b n l, b n l -> b n', residual.detach(), head_log_probs)
    return - advantages * centered_score

# rlhf trainer

class RLHFTrainer(Module):

    def __init__(
        self,
        *,
        prompts: list[str] | None = None,
        prompts_path: str | None = None,
        prompt_token_ids: Tensor | None = None,
        tokenizer: Callable | None = None,
        palm: PaLM,
        reward_model: RewardModel,
        grpo_num_times_sample_rewards = 10,
        actor_lr = 1e-4,
        actor_wd = 0.,
        actor_lora = True,
        actor_lora_r = 8,
        actor_dropout = 0.,
        betas = (0.9, 0.999),
        max_norm = None,
        eps_clip = 0.2,
        beta_s = .01,
        pad_value = 0.,
        minibatch_size = 16,
        epochs = 1,
        kl_div_loss_weight = 0.1,               # between old action probs and new action probs - not sure what the right value is
        use_simple_policy_optimization = False, # Xie et al. https://arxiv.org/abs/2401.16025v9
        use_dr_grpo = False,                    # Sea AI lab - https://arxiv.org/abs/2503.20783
        dr_grpo_constant = None,                # constant scaling factor for Dr. GRPO (defaults to action_sample_times + 1)
        use_max_rl = False,                     # maxRL - https://arxiv.org/abs/2602.02710
        use_score_centering = False,            # Marek & Ryabinin - https://arxiv.org/abs/2609.20807
        score_centering_topk = None,            # top-k approximation (e.g. 128 or 32); recommended for large vocabularies, defaults to full vocabulary
        score_centering_weight_fn: Callable = torch.ones_like, # optional weight function f(r) to compose score centering with IS (e.g. TIS, MIS)
        add_entropy_to_advantage = False,
        entropy_to_advantage_kappa = 2.,
        entropy_to_advantage_scale = 0.4,       # they use 0.4 for GRPO, 0.1 for PPO
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        self.accelerate = Accelerator(**accelerate_kwargs)

        # take care of prompts -> token ids

        assert (exists(prompts) + exists(prompts_path) + exists(prompt_token_ids)) == 1

        if exists(prompts_path):
            path = Path(prompts_path)
            prompts = path.read_text().split('\n')

        if exists(prompts):
            assert len(prompts) > 0, 'no prompts'
            assert exists(tokenizer), 'tokenizer must be passed in if raw text prompts are given'
            prompt_token_ids = tokenizer(prompts)

        self.pad_value = pad_value # token pad value
        self.num_prompts = prompt_token_ids.shape[0]
        self.register_buffer('prompt_token_ids', prompt_token_ids)

        # models

        self.palm = palm

        actor = Actor(
            palm = palm,
            actor_lora = actor_lora,
            actor_lora_r = actor_lora_r,
            actor_dropout = actor_dropout,
        )

        self.actor = actor

        self.actor_generate = self.actor.generate

        self.reward_model = reward_model.eval()

        # train hyperparameters

        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_norm = max_norm

        self.kl_div_loss_weight = kl_div_loss_weight

        # optimizers

        self.actor_optim = AdoptAtan2(actor.parameters(), lr = actor_lr, weight_decay = actor_wd, betas = betas)

        # spo

        self.use_spo = use_simple_policy_optimization

        # dr grpo

        self.use_dr_grpo = use_dr_grpo or exists(dr_grpo_constant)
        self.dr_grpo_constant = dr_grpo_constant

        # maxrl

        self.use_max_rl = use_max_rl

        assert not (self.use_dr_grpo and self.use_max_rl), 'cannot use both Dr. GRPO and MaxRL'

        # score centering

        self.use_score_centering = use_score_centering
        self.score_centering_topk = score_centering_topk
        self.score_centering_weight_fn = score_centering_weight_fn

        assert not (self.use_spo and self.use_score_centering), 'cannot use both SPO and Score Centering'

        # "reasoning from exploration" paper

        self.add_entropy_to_advantage = add_entropy_to_advantage
        self.entropy_to_advantage_scale = entropy_to_advantage_scale
        self.entropy_to_advantage_kappa = entropy_to_advantage_kappa

        # grpo hyperparams

        self.eps_clip = eps_clip
        self.beta_s = beta_s

        # grpo - the number of times to sample rewards for a given state (prompt) for normalization

        self.grpo_num_times_sample_rewards = grpo_num_times_sample_rewards

        # prepare with accelerator

        (
            self.actor,
            self.reward_model,
            self.actor_optim,
        ) = self.accelerate.prepare(
            self.actor,
            self.reward_model,
            self.actor_optim,
        )


    def print(self, msg):
        return self.accelerate.print(msg)

    def save(self, filepath = './checkpoint.pt'):
        torch.save(self.actor.state_dict(), filepath)

    def load(self, filepath = './checkpoint.pt'):
        state_dict = torch.load(filepath)
        self.actor.load_state_dict(state_dict)

    @property
    def device(self):
        return self.accelerate.device

    @torch.no_grad()
    def generate(
        self,
        max_seq_len,
        *args,
        prompt,
        num_samples = 4,  # sample 4 per prompt and select the one with highest reward
        **kwargs
    ):
        assert prompt.ndim == 1, 'only one prompt allowed at a time for now'
        prompt = repeat(prompt, 'n -> b n', b = num_samples)

        actor = self.accelerate.unwrap_model(self.actor)
        reward_model = self.accelerate.unwrap_model(self.reward_model)

        actor.eval()

        (
            actions,
            sequences,
            mask,
            prompt_mask,
            action_logits
        ) = actor.generate(
            prompt,
            *args,
            max_seq_len = max_seq_len,
            **kwargs
        )

        rewards = reward_model(
            sequences,
            prompt_mask = prompt_mask,
            mask = mask
        )

        best_sequence_index = rewards.topk(1, dim = -1).indices

        best_sequence = sequences[best_sequence_index]
        best_sequence = rearrange(best_sequence, '1 ... -> ...')

        return best_sequence

    def learn(
        self,
        memories: Deque[Memory]
    ):
        # stack all data stored in the memories

        all_memories_stacked_and_padded = list(map(partial(pad_sequence_fixed, batch_first = True), zip(*memories)))

        # prepare dataloader for policy phase training

        dl = create_dataloader(all_memories_stacked_and_padded, self.minibatch_size, device = self.device)

        self.actor.train()

        # GRPO training

        for _ in range(self.epochs):
            for (
                sequences,
                prompt_masks,
                masks,
                old_action_probs,
                old_log_probs,
                advantages,
            ) in dl:
                action_masks = ~prompt_masks & masks

                action_logits = self.actor(
                    sequences,
                    mask = action_masks
                )

                action_logits = shift(action_logits, 1, dim = -2) # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token

                action_log_probs_all = action_logits.log_softmax(dim = -1)
                action_probs = action_log_probs_all.exp()
                action_log_probs = get_at('b n [l], b n -> b n', action_log_probs_all, sequences)

                # calculate entropies, taking into account which part of the sequence is actually an action

                per_token_entropies = entropy(action_probs, reduce = False)

                # calculate kl div between old action probs and new ones, taking into account which part of the sequence is action or not

                kl_penalty = 0.

                if self.kl_div_loss_weight > 0:
                    kl_penalty = masked_kl_div(old_action_probs, action_probs, mask = action_masks) * self.kl_div_loss_weight

                # subtract the kl penalty from the advantages

                advantages = advantages - kl_penalty

                # to encourage exploration, they add per token entropies

                if self.add_entropy_to_advantage:
                    entropy_scale, kappa = self.entropy_to_advantage_scale, self.entropy_to_advantage_kappa

                    entropy_reward = entropy_scale * per_token_entropies.masked_fill(~action_masks, 0.).detach()
                    max_entropy_reward = rearrange(advantages.abs() / kappa, 'b -> b 1')

                    advantages = add('b, b n', advantages, entropy_reward.clamp(max = max_entropy_reward))

                else:
                    advantages = rearrange(advantages, 'b -> b 1')

                entropies = masked_mean(per_token_entropies, mask = action_masks)

                # calculate policy loss

                if self.use_spo:
                    # SPO - Line 14 Algorithm 1 - https://arxiv.org/abs/2401.16025v9
                    ratios = (action_log_probs - old_log_probs).exp()
                    policy_loss = - (ratios * advantages) + (ratios - 1.).square() * (advantages.abs() / (2 * self.eps_clip))

                elif self.use_score_centering:
                    policy_loss = score_centering_loss(
                        action_log_probs_all,
                        old_action_probs,
                        sequences,
                        advantages,
                        topk = self.score_centering_topk,
                        weight_fn = self.score_centering_weight_fn,
                    )

                else:
                    # classic ppo
                    ratios = (action_log_probs - old_log_probs).exp()
                    surr1 = ratios * advantages
                    surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    policy_loss = - torch.min(surr1, surr2)

                # entropy loss

                policy_loss = masked_mean(policy_loss, mask = action_masks) - self.beta_s * entropies

                # combine losses

                loss = policy_loss.mean()

                # update actor

                self.accelerate.backward(loss)

                self.print(f'policy_loss: {loss.item():.3f}')

                if exists(self.max_norm):
                    self.accelerate.clip_grad_norm_(self.actor.parameters(), self.max_norm)

                self.actor_optim.step()
                self.actor_optim.zero_grad()

    def train(
        self,
        num_episodes = 50000,
        max_timesteps = 500,
        update_timesteps = 5000,
        max_batch_size = 16,
        max_seq_len = 2048,
        eos_token = None,
        temperature = 1.
    ):
        action_sample_times = self.grpo_num_times_sample_rewards

        device = self.device

        time = 0
        memories = deque([])

        for eps in tqdm(range(num_episodes), desc = 'episodes'):
            for timestep in range(max_timesteps):
                time += 1

                # select a bunch of random states (prompts)
                # and get the action (sampled sequence from palm as well as the action probs)
                # also calculate the reward using reward model and store

                rand_prompt_index = randrange(0, self.num_prompts)

                state = self.prompt_token_ids[rand_prompt_index]

                # remove padding from state

                state_mask = state != self.pad_value
                state = state[state_mask]

                # will sample each state more than once to get an estimate of the value, for removing the critic altogether from GRPO paper, Shao et al.

                states = repeat(state, 'n -> b n', b = action_sample_times + 1)

                # get predicted sequence

                (
                    actions,
                    sequence,
                    mask,
                    prompt_mask,
                    action_logits,
                ) = self.actor_generate(
                    states,
                    max_seq_len = max_seq_len,
                    eos_token = eos_token,
                    temperature = temperature,
                )

                action_logits = shift(action_logits, 1, dim = -2) # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token

                action_log_probs_all = action_logits.log_softmax(dim = -1)
                action_prob = action_log_probs_all.exp()
                action_log_prob = get_at('b n [l], b n -> b n', action_log_probs_all, sequence)

                # get reward as given by supervised trained reward model

                sequence = torch.cat((states, actions), dim = 1)

                prompt_length = states.shape[1]
                prompt_mask = torch.arange(sequence.shape[-1], device = device) < prompt_length
                prompt_mask = repeat(prompt_mask, 'n -> b n', b = action_sample_times + 1)

                mask = default(mask, torch.ones(sequence.shape, dtype = torch.bool, device = device))

                rewards = self.reward_model(
                    sequence,
                    prompt_mask = prompt_mask,
                    mask = mask,
                )

                rewards = rewards.float()

                # rewards are normalized for use as advantages
                # 1. Dr. GRPO (Sea AI Lab https://arxiv.org/abs/2503.20783) - removes standard deviation, normalizes by constant
                # 2. MaxRL (https://arxiv.org/abs/2602.02710) - normalizes by average reward
                # 3. Old way (Shao et al. https://arxiv.org/abs/2402.03300) - normalizes by standard deviation

                if self.use_dr_grpo:
                    divisor = default(self.dr_grpo_constant, action_sample_times + 1)
                elif self.use_max_rl:
                    divisor = rewards.mean().abs()
                else:
                    divisor = rewards.std()

                normalized_rewards = safe_div(rewards - rewards.mean(), divisor)

                # store memory for learning

                detach_to_cpu_ = lambda t: t.detach().cpu()

                memories.extend([Memory(*memories) for memories in zip(*map(detach_to_cpu_, (
                    sequence,
                    prompt_mask,
                    mask,
                    action_prob,
                    action_log_prob,
                    normalized_rewards,
                )))])

                # learn from the stored memories

                if divisible_by(time, update_timesteps):
                    self.learn(memories)
                    memories.clear()

        self.print('grpo rlhf training complete')
