import copy
from pathlib import Path

from tqdm import tqdm
from beartype import beartype
from beartype.typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from palm_rlhf_pytorch.utils import masked_mean, gumbel_sample
from palm_rlhf_pytorch.palm import PaLM

# helper functions

def exists(val):
    return val is not None

# Reward Model - PaLM with a scalar head

@beartype
class RewardModel(nn.Module):
    def __init__(
        self,
        palm: PaLM,
        dropout = 0.1,
        num_binned_output = 0.,
        use_lora = True,
        lora_r = 8,
        reward_lora_scope = 'reward',
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
        labels = None,
        sample = False,
        sample_temperature = 1.,
        disable_lora = False
    ):
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

        if sample and self.binned_output:
            assert not exists(labels)
            pred = gumbel_sample(pred, temperature = sample_temperature, dim = -1)

        if not exists(labels):
            return pred

        if not self.binned_output:
            return F.mse_loss(pred, labels)

        return F.cross_entropy(pred, labels)
