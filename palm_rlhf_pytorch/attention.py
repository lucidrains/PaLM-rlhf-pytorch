import torch
from torch import nn, einsum
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version

from einops import rearrange

# constants

Config = namedtuple('FlashConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attention(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash_attn = False
    ):
        super().__init__()
        self.dropout = dropout
        self.causal = causal
        self.attn_dropout = nn.Dropout(dropout)

        assert version.parse(torch.__version__) >= version.parse('2.0.0'), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.use_flash_attn = use_flash_attn

        self.register_buffer("mask", None, persistent=False)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len = *q.shape, k.shape[-2]

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
        v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        device_str = 'cuda' if torch.cuda.is_available() and q.is_cuda else 'cpu'
        device = torch.device(device_str)

        try:
            if device_str == 'cuda':
                device_properties = torch.cuda.get_device_properties(device)

                if device_properties.major == 8 and device_properties.minor == 0:
                    print_once('A100 GPU detected, using flash attention')
                    config = Config(True, False, False)
                else:
                    print_once('Non-A100 GPU detected, using math or mem efficient attention')
                    config = Config(False, True, True)
            else:
                print_once('CPU detected, using default context manager settings')
                config = Config(True, True, True)

        except RuntimeError as error:
            print(f'An error occurred: {error}.')

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = self.causal
            )

        return out

    def forward(self, q, k, v, mask = None):
        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash_attn:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out
