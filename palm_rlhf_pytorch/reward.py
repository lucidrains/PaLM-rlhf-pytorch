import copy
from pathlib import Path

from tqdm import tqdm
from beartype import beartype

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from palm_rlhf_pytorch.utils import masked_mean, gumbel_sample
from palm_rlhf_pytorch.palm import PaLM

# helper functions

def exists(val):
    return val is not None

def default(val, default_val):
    return val if exists(val) else default_val

# Reward Model - PaLM with a scalar head

class RewardModel(Module):
    @beartype
    def __init__(
        self,
        palm: PaLM,
        dropout = 0.1,
        num_binned_output = 0.,
        use_lora = True,
        lora_r = 8,
        reward_lora_scope = 'reward',
        sample_from_bins = None,
        sample_temperature = 1.
    ):
        super().__init__()

        self.palm = copy.deepcopy(palm)
        self.palm.set_dropout(dropout)

        self.reward_lora_scope = reward_lora_scope if use_lora else None

        if exists(self.reward_lora_scope):
            self.palm.add_finetune_params(reward_lora_scope, lora_r = lora_r)

        dim = palm.dim

        self.binned_output = num_binned_output > 1

        self.prompt_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.response_embed = nn.Parameter(torch.zeros(1, 1, dim))

        if self.binned_output:
            self.to_pred = nn.Linear(dim, num_binned_output)
        else:
            self.to_pred = nn.Sequential(
                nn.Linear(dim, 1, bias = False),
                Rearrange('... 1 -> ...')
            )

        self.sample_from_bins = default(sample_from_bins, self.binned_output)
        self.sample_temperature = sample_temperature

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    def finetune_parameters(self):
        return [
            *self.to_pred.parameters(),
            *(self.palm.finetune_parameters(self.reward_lora_scope) if exists(self.reward_lora_scope) else self.palm.parameters())
        ]

    def forward(
        self,
        x,
        mask = None,
        prompt_mask = None,
        prompt_lengths = None,
        labels = None,
        disable_lora = False
    ):

        assert not (exists(prompt_mask) and exists(prompt_lengths))

        # derive prompt mask from prompt lengths

        if exists(prompt_lengths):
            batch, seq_len = x.shape
            arange = torch.arange(seq_len, device = x.device)
            prompt_mask = repeat(arange, 'n -> b n', b = batch) < rearrange(prompt_lengths, 'b -> b 1')

        # reward model should have an understanding of which section is prompt, and which section is response

        extra_embed = None

        if exists(prompt_mask):
            extra_embed = torch.where(
                rearrange(prompt_mask, 'b n -> b n 1'),
                self.prompt_embed,
                self.response_embed
            )

        # get embeddings from palm

        embeds = self.palm(
            x,
            extra_embed = extra_embed,
            return_only_embedding = True,
            disable_lora = disable_lora,
            finetune_scope = self.reward_lora_scope
        )

        pooled = masked_mean(embeds, mask, dim = 1)
        pred = self.to_pred(pooled)

        is_inferencing = not exists(labels)

        if (
            self.sample_from_bins and
            self.binned_output and
            is_inferencing
        ):
            pred = gumbel_sample(pred, temperature = self.sample_temperature, dim = -1)

        if is_inferencing:
            return pred

        if not self.binned_output:
            return F.mse_loss(pred, labels)

        return F.cross_entropy(pred, labels)
