import torch
from torch import nn

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return ((val,) * length) if not isinstance(val, tuple) else val

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

class FusedLoRA(nn.Module):
    def __init__(
        self,
        dim,
        dim_outs,
        Rs = 8,
        alphas = None
    ):
        super().__init__()
        assert isinstance(dim_outs, tuple)
        num_dims_out = len(dim_outs)
        Rs = cast_tuple(Rs, num_dims_out)
        alphas = cast_tuple(alphas, num_dims_out)

        self.loras = nn.ModuleList([])
        for dim_out, r, alpha in zip(dim_outs, Rs, alphas):
            self.loras.append(LoRA(dim, dim_out, r = r, alpha = alpha))

    @property
    def weight(self):
        return torch.cat(tuple(lora.weight for lora in self.loras), dim = -1)

    def forward(self, x):
        return x @ self.weight
