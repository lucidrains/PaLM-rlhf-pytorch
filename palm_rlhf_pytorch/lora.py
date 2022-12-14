import torch
from torch import nn

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# LoRA - https://arxiv.org/abs/2106.09685

class LoRA(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        r = 8,
        alpha = None
    ):
        super().__init__()
        alpha = default(alpha, r)
        self.scale = alpha / r

        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.zeros(r, dim_out))

    @property
    def weight(self):
        return (self.A @ self.B) * self.scale

    def forward(self, x):
        return x @ self.weight
