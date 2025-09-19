"""
FlowRL - reward distribution matching via flow balance (Eq. 6)
"""

from __future__ import annotations
from typing import Callable, Deque

import math
import copy
from pathlib import Path
from functools import partial
from collections import deque, namedtuple
from random import randrange

import torch
from torch import nn, Tensor
from torch.nn import Module, Sequential
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from adam_atan2_pytorch import AdoptAtan2

from palm_rlhf_pytorch.palm import PaLM
from palm_rlhf_pytorch.reward import RewardModel
from palm_rlhf_pytorch.utils import masked_mean, eval_decorator

from accelerate import Accelerator

from x_mlps_pytorch import MLP
from accelerate.utils.tqdm import tqdm

import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# einstein notation

# b - batch 
# n - sequence
# d - feature dimension
# l - logits

# FlowRL based training
# Uses flow-balanced optimization with trajectory balance loss

FlowRLActionReturn = namedtuple('FlowRLActionReturn', [
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

        return FlowRLActionReturn(
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
    if exists(val):
        return val
    return d() if callable(d) else d

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

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def shift(t, value = 0, shift = 1, dim = -1):
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, shift, -shift), value = value)

def entropy(prob, dim = -1):
    return (-prob * log(prob)).sum(dim = -1)

def masked_kl_div(prob1, prob2, mask = None, reduce_batch = False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(dim = -1)
    loss = masked_mean(kl_divs, mask)

    if not reduce_batch:
        return loss

    return loss.mean()

# partition function model for FlowRL

class PartitionFunction(Module):
    def __init__(self, hidden_dim=768, num_layers=3):
        super().__init__()
        dims = [hidden_dim] * num_layers + [1]

        self.mlp = Sequential(
            MLP(*dims),
            Rearrange('... 1 -> ...')
        )
        
    def forward(self, prompt_embeddings):
        return self.mlp(prompt_embeddings)

# FlowRL trainer

class FlowRLTrainer(Module):

    def __init__(
        self,
        *,
        prompts: list[str] | None = None,
        prompts_path: str | None = None,
        prompt_token_ids: Tensor | None = None,
        tokenizer: Callable | None = None,
        palm: PaLM,
        reward_model: RewardModel,
        flowrl_num_times_sample = 8,
        beta = 15.0,  # inverse temperature for FlowRL
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
        kl_div_loss_weight = 0.1, # between old action probs and new action probs - not sure what the right value is
        partition_function_hidden_dim = 768,
        partition_function_lr = 1e-4,
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

        # flowrl hyperparams
        
        self.eps_clip = eps_clip
        self.beta = beta  # inverse temperature for reward distribution
        self.flowrl_num_times_sample = flowrl_num_times_sample
        
        # partition function model
        self.partition_function = PartitionFunction(
            hidden_dim=partition_function_hidden_dim,
            num_layers=3
        )
        
        # optimizer for partition function
        self.partition_function_optim = AdoptAtan2(
            self.partition_function.parameters(),
            lr=partition_function_lr,
            betas=betas
        )

        # prepare with accelerator

        (
            self.actor,
            self.reward_model,
            self.actor_optim,
            self.partition_function,
            self.partition_function_optim,
        ) = self.accelerate.prepare(
            self.actor,
            self.reward_model,
            self.actor_optim,
            self.partition_function,
            self.partition_function_optim,
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

        action_return = actor.generate(
            prompt,
            *args,
            max_seq_len = max_seq_len,
            **kwargs
        )
        
        actions = action_return.actions
        sequences = action_return.sequence
        mask = action_return.mask
        prompt_mask = action_return.prompt_mask
        action_logits = action_return.action_logits

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
        self.partition_function.train()

        # FlowRL: minimize squared trajectory balance (Eq. 6)
        for _ in range(self.epochs):
            for (
                sequences,
                prompt_masks,
                masks,
                old_action_probs,
                old_log_probs,
                normalized_rewards,
            ) in dl:
                action_masks = ~prompt_masks & masks
                
                # π_θ(y|x)
                action_logits = self.actor(sequences, mask = action_masks)
                action_logits = shift(action_logits, shift = 1, dim = -2)
                action_len = old_log_probs.shape[-1]
                action_probs = action_logits.softmax(dim = -1)
                action_log_probs = einx.get_at('b n [l], b n -> b n', action_probs, sequences)
                action_log_probs = action_log_probs[:, -action_len:]
                
                # π_ref(y|x) 
                with torch.no_grad():
                    ref_logits = self.palm(sequences, finetune_scope = None)
                    ref_logits = shift(ref_logits, shift = 1, dim = -2)
                    ref_probs = ref_logits.softmax(dim = -1)
                    ref_log_probs = einx.get_at('b n [l], b n -> b n', ref_probs, sequences)
                    ref_log_probs = ref_log_probs[:, -action_len:]
                
                # w = clip(π_θ/π_old)
                trajectory_log_ratio = (action_log_probs - old_log_probs).sum(dim=-1)
                trajectory_ratio = torch.exp(trajectory_log_ratio)
                clipped_ratio = torch.clamp(trajectory_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                importance_weight = clipped_ratio.detach()
                
                # Z_φ(x) from prompt embeddings
                with torch.no_grad():
                    prompt_lens = prompt_masks.sum(dim=1)
                    max_prompt_len = prompt_lens.max().item()
                    prompt_tokens = sequences[:, :max_prompt_len]
                    
                    prompt_final_embeddings = self.actor.actor_palm(
                        prompt_tokens,
                        return_only_embedding=True,
                        finetune_scope=self.actor.actor_lora_scope if self.actor.actor_lora else None
                    )
                    
                    prompt_mask_truncated = prompt_masks[:, :max_prompt_len]
                    masked_embeddings = einx.multiply('... d, ...', prompt_final_embeddings, prompt_mask_truncated)
                    prompt_embeddings = einx.divide('b d, b', masked_embeddings.sum(dim=1), prompt_lens)
                
                log_z_phi = self.partition_function(prompt_embeddings)
                
                # 1/|y| normalization
                response_lengths = action_masks.sum(dim=1).float().clamp(min=1.0)
                norm_policy_log_probs = action_log_probs.sum(dim=-1) / response_lengths
                norm_ref_log_probs = ref_log_probs.sum(dim=-1) / response_lengths
                
                # Eq. 6: (log Z_φ(x) + 1/|y| log π_θ(y|x) - βr̂ - 1/|y| log π_ref(y|x))²
                trajectory_balance = (
                    log_z_phi
                    + norm_policy_log_probs
                    - self.beta * normalized_rewards
                    - norm_ref_log_probs
                )
                
                flowrl_loss = importance_weight * (trajectory_balance ** 2)
                loss = flowrl_loss.mean()
                
                # Update both actor and partition function
                self.accelerate.backward(loss)
                
                self.print(f'flowrl_loss: {loss.item():.3f}, trajectory_balance: {trajectory_balance.mean().item():.3f}')
                
                if exists(self.max_norm):
                    self.accelerate.clip_grad_norm_(self.actor.parameters(), self.max_norm)
                
                self.actor_optim.step()
                self.actor_optim.zero_grad()
                
                self.partition_function_optim.step()
                self.partition_function_optim.zero_grad()

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
        action_sample_times = self.flowrl_num_times_sample

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

                action_return = self.actor_generate(
                    states,
                    max_seq_len = max_seq_len,
                    eos_token = eos_token,
                    temperature = temperature,
                )
                
                actions = action_return.actions
                sequence = action_return.sequence
                mask = action_return.mask
                prompt_mask = action_return.prompt_mask
                action_logits = action_return.action_logits

                action_logits = shift(action_logits, shift = 1, dim = -2) # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token

                action_prob = action_logits.softmax(dim = -1)

                action_len = actions.shape[-1]

                action_log_prob = einx.get_at('b n [l], b n -> b n', action_prob, sequence)
                action_log_prob = action_log_prob[:, -action_len:]

                # get reward as given by supervised trained reward model

                sequence = torch.cat((states, actions), dim = 1)

                prompt_length = states.shape[1]
                prompt_mask = torch.arange(sequence.shape[-1], device = device) < prompt_length
                prompt_mask = repeat(prompt_mask, 'n -> b n', b = action_sample_times + 1)

                mask = default(mask, lambda: torch.ones(sequence.shape, dtype = torch.bool, device = device))

                rewards = self.reward_model(
                    sequence,
                    prompt_mask = prompt_mask,
                    mask = mask,
                )

                rewards = rewards.float()

                # rewards are normalized for use in FlowRL trajectory balance
                # normalize rewards across the batch

                normalized_rewards = (rewards - rewards.mean()) / rewards.std().clamp(min = 1e-8)

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

        print('FlowRL training complete')
