from contextlib import ExitStack, contextmanager, nullcontext

import torch
from torch import einsum, nn
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from palm_rlhf_pytorch.utils import top_p, top_k
from palm_rlhf_pytorch.lora import LoRA

def exists(val):
    return val is not None

@contextmanager
def multi_context(*cms):
    with ExitStack() as stack:
        yield [stack.enter_context(cls()) for cls in cms]

def eval_decorator(model):
    @contextmanager
    def inner():
        was_training = model.training
        model.eval()
        yield
        model.train(was_training)
    return inner

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        y = self.fn(x, **kwargs)

        if not y.requires_grad and not x.requires_grad:
            return x.add_(y)

        return y + x

# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4, lora=True, lora_r=8):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)

        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        # fine tuning parameters

        self.qkv_lora = self.attn_out_lora = None

        if lora:
            self.qkv_lora = nn.ModuleList([
                LoRA(dim, dim_out, r=lora_r) for dim_out in self.fused_dims[:-1]
            ])

            self.attn_out_lora = LoRA(attn_inner_dim, dim, r=lora_r) if lora else None

        # parallel feedforward tail

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, disable_lora = False):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        if exists(self.qkv_lora) and not disable_lora:
            lq, lk, lv = tuple(lora(x) for lora in self.qkv_lora)
            q, k, v = (q + lq), (k + lk), (v + lv)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        attn_out = self.attn_out(out)

        ff_out = self.ff_out(ff)

        if exists(self.attn_out_lora) and not disable_lora:
            attn_out = attn_out + self.attn_out_lora(out)

        return attn_out + ff_out

# transformer


class PaLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        lora=True,
        lora_r=8,
    ):
        super().__init__()
        self.dim = dim
        self.lora = lora

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, lora=lora, lora_r=lora_r)))

        self.norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        self.to_logits.weight = self.token_emb.weight

        nn.init.normal_(self.token_emb.weight, std=0.02)

    # researcher train palm parameters first
    # before finetuning

    def palm_parameters(self):
        if not self.lora:
            return self.parameters()

        return set(self.parameters()) - set(self.finetune_parameters())

    def finetune_parameters(self):
        assert self.lora, 'lora not present on this palm'

        loras = []

        for layer in self.layers:
            loras.extend(layer.fn.attn_out_lora.parameters())

        return set(loras)

    # generate function

    def generate(
        self,
        prime,
        seq_len,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        trainable = False,
        **kwargs
    ):
        n, out = prime.shape[-1], prime.clone()

        context = multi_context(eval_decorator(self), torch.no_grad) if not trainable else nullcontext

        with context:
            for _ in range(seq_len):
                logits = self.forward(out, **kwargs)

                filtered_logits = filter_logits_fn(logits[:, -1], thres = filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

                sample = torch.multinomial(probs, 1)
                out = torch.cat((out, sample), dim=-1)

        return out[:, n:]

    def forward(
        self,
        x,
        return_loss = False,
        disable_lora = False,
        return_embedding = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for layer in self.layers:
            x = layer(x, disable_lora = disable_lora) + x

        x = self.norm(x)

        if return_embedding:
            return x

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels)

# Reward Model - PaLM with a scalar head

class RewardModel(nn.Module):
    def __init__(
        self,
        palm: PaLM
    ):
        super().__init__()
        self.palm = palm

        self.to_pred = nn.Sequential(
            nn.Linear(palm.dim, 1, bias = False),
            Rearrange('... 1 -> ...')
        )

    def forward(self, x):
        embeds = self.palm(x, return_embedding = True)

        pooled = reduce(embeds, 'b n d -> b d', 'mean')
        return self.to_pred(pooled)

# PaLM with actor and critic heads

class ActorWithValueHead(nn.Module):
    def __init__(
        self,
        palm: PaLM
    ):
        super().__init__()
        self.palm = palm

        self.value_head = nn.Sequential(
            nn.Linear(palm.dim, 1),
            Rearrange('... 1 -> ...')
        )

    def forward(self, x):
        embeds = self.palm(x, return_embedding = True)

        actions = self.palm.to_logits(embeds)
        values = self.value_head(embeds)

        return actions, values
