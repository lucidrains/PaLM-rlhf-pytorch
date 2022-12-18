from pathlib import Path
from tqdm import tqdm
from collections import deque, namedtuple
from random import randrange

from beartype import beartype
from typing import List, Optional, Callable, Deque

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange

from palm_rlhf_pytorch.palm_rlhf_pytorch import (
    PaLM,
    ActorWithValueHead,
    RewardModel
)

from palm_rlhf_pytorch.utils import masked_mean

# data

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])
AuxMemory = namedtuple('Memory', ['state', 'target_value', 'old_values'])

class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

def create_dataloader(data, batch_size, shuffle = True, **kwargs):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = shuffle, **kwargs)

# helper functions

def exists(val):
    return val is not None

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / t.std().clamp(min = eps)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def get_log_prob(prob, indices, dim = -1):
    return log(prob.gather(dim, indices))

def get_entropy(prob, dim = -1, mask = None):
    entropies = (prob * log(prob)).sum(dim = -1)
    return masked_mean(entropies, mask = mask).mean()

def kl_div_with_maybe_mask(prob1, prob2, mask = None):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob2) - log(prob1))).sum(dim = -1)

    if not exists(mask):
        return kl_divs.mean()

    return masked_mean(kl_divs, mask).mean()

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))

# rlhf trainer

class RLHFTrainer(nn.Module):
    def __init__(
        self,
        *,
        prompts: Optional[List[str]] = None,
        prompts_path: Optional[str] = None,
        prompt_token_ids: Optional[torch.Tensor] = None,
        tokenizer: Callable = None,
        palm: PaLM,
        reward_model: RewardModel,
        actor_critic: Optional[ActorWithValueHead] = None,
        actor_lr = 1e-4,
        critic_lr = 1e-4,
        betas = (0.9, 0.999),
        lam = 0.95,
        gamma = 0.99,
        eps_clip = 0.2,
        value_clip = 0.4,
        beta_s = .01,
        pad_value = 0.
    ):
        super().__init__()

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

        if not exists(actor_critic):
            actor_critic = ActorWithValueHead(palm = palm, pooled_values = True).to(palm.device)

        self.actor_critic = actor_critic

        self.reward_model = reward_model.eval()

        # optimizers

        self.actor_optim = Adam(actor_critic.actor_parameters(), lr = actor_lr, betas = betas)
        self.critic_optim = Adam(actor_critic.critic_parameters(), lr = critic_lr, betas = betas)

        # ppo hyperparams

        self.lam = lam
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.beta_s = beta_s

    def save_actor_critic(self, filepath = './checkpoint.pt'):
        torch.save(self.actor_critic.state_dict(), filepath)

    def load_actor_critic(self, filepath = './checkpoint.pt'):
        state_dict = torch.load(filepath)
        self.actor_critic.load_state_dict(state_dict)

    @property
    def device(self):
        return next(self.parameters()).device

    def learn(
        self,
        memories: Deque[Memory],
        aux_memories: Deque[AuxMemory],
        next_state
    ):
        # retrieve and prepare data from memory for training

        states = []
        actions = []
        old_log_probs = []
        rewards = []
        masks = []
        values = []

        for mem in memories:
            states.append(mem.state)
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)
            rewards.append(mem.reward)
            masks.append(1 - float(mem.done))
            values.append(mem.value)

        # calculate generalized advantage estimate

        self.eps_clip = ep
        next_state = torch.from_numpy(next_state).to(device)
        next_value = self.critic(next_state).detach()
        values = values + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # convert values to torch tensors

        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_values = to_torch_tensor(values[:-1])
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = torch.tensor(returns).float().to(device)

        # store state and target values to auxiliary memory buffer for later training

        aux_memory = AuxMemory(states, rewards, old_values)
        aux_memories.append(aux_memory)

        # prepare dataloader for policy phase training

        dl = create_shuffled_dataloader([states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

        # policy phase training, similar to original PPO

        for _ in range(self.epochs):
            for states, actions, old_log_probs, rewards, old_values in dl:
                action_probs, _ = self.actor(states)
                values = self.critic(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                # calculate clipped surrogate objective, classic PPO loss

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

                update_network_(policy_loss, self.opt_actor)

                # calculate value loss and update value network separate from policy network

                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)

    def learn_aux(
        self,
        aux_memories: Deque[AuxMemory]
    ):
        # gather states and target values into one tensor

        states = []
        rewards = []
        old_values = []
        for state, reward, old_value in aux_memories:
            states.append(state)
            rewards.append(reward)
            old_values.append(old_value)

        states = torch.cat(states)
        rewards = torch.cat(rewards)
        old_values = torch.cat(old_values)

        # get old action predictions for minimizing kl divergence and clipping respectively

        old_action_probs, _ = self.actor(states)
        old_action_probs.detach_()

        # prepared dataloader for auxiliary phase training

        dl = create_shuffled_dataloader([states, old_action_probs, rewards, old_values], self.minibatch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network, while making sure the policy network does not change the action predictions (kl div loss)

        for epoch in range(self.epochs_aux):
            for states, old_action_probs, rewards, old_values in tqdm(dl, desc=f'auxiliary epoch {epoch}'):
                action_probs, policy_values = self.actor(states)
                action_logprobs = action_probs.log()

                # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                policy_loss = aux_loss + loss_kl

                update_network_(policy_loss, self.opt_actor)

                # paper says it is important to train the value network extra during the auxiliary phase

                values = self.critic(states)
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)

    def train(
        self,
        num_episodes = 50000,
        max_timesteps = 500,
        update_timesteps = 5000,
        num_policy_updates_per_aux = 32,
        max_batch_size = 16,
        max_seq_len = 2048,
        eos_token = None,
        temperature = 1.
    ):
        device = self.device
        time = 0
        num_policy_updates = 0

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

                # get predicted sequence

                action, action_prob, value = self.actor_critic.generate(
                    state,
                    return_action_probs = True,
                    return_value = True,
                )

                action_log_prob = get_log_prob(action_prob, action)

                # get reward as given by supervised trained reward model

                sequence = torch.cat((state, actions), dim = 0)

                prompt_length = len(state)
                prompt_mask = torch.arange(sequence.shape[-1], device = device) < prompt_length

                sequence = rearrange('n -> 1 n')
                prompt_mask = rearrange('n -> 1 n')

                reward = self.reward_model(sequence, prompt_mask)

                # store memory for learning

                memories.append(Memory(
                    state,
                    action,
                    action_log_prob,
                    reward,
                    value
                ))

                # learn from the stored memories

                if time % update_timesteps == 0:
                    self.learn(memories, aux_memories)

                    num_policy_updates += 1
                    memories.clear()

                    if num_policy_updates % num_policy_updates_per_aux == 0:
                        self.learn_aux(aux_memories)
                        aux_memories.clear()                    

        print('rlhf training complete')
