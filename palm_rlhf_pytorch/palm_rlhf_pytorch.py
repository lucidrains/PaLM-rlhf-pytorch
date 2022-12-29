import math
import copy
from pathlib import Path
from collections import namedtuple

from tqdm import tqdm
from beartype import beartype
from beartype.typing import Tuple, Optional

import torch
from torch import einsum, nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from palm_rlhf_pytorch.utils import top_p, top_k, masked_mean, gumbel_sample
from palm_rlhf_pytorch.lora import LoRA

# functions and decorators

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    return t

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
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

        if not any([t.requires_grad for t in (x, y)]):
            return x.add_(y)

        return y + x

# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale_base = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t, scale = 1.):
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        causal = True,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        xpos_scale_base = 512
    ):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.causal = causal

        self.rotary_emb = RotaryEmbedding(dim_head, scale_base = xpos_scale_base)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)

        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # parallel feedforward tail

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)
        self.register_buffer("pos_emb_scale", None, persistent=False)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if exists(self.pos_emb) and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n], self.pos_emb_scale[:n]

        pos_emb, scale = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        self.register_buffer("pos_emb_scale", scale, persistent=False)
        return pos_emb, scale

    def forward(
        self,
        x,
        finetune_modules = None
    ):
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

        # finetune loras

        lora_q = lora_k = lora_v = lora_o = None

        if exists(finetune_modules):
            lora_q, lora_k, lora_v, lora_o = finetune_modules
            q = q + lora_q(x)
            k = k + lora_k(x)
            v = v + lora_v(x)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        # rotary embeddings with xpos decay for better length extrapolation

        positions, scale = self.get_rotary_embedding(n, device)

        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        attn_out = self.attn_out(out)

        ff_out = self.ff_out(ff)

        if exists(lora_o):
            attn_out = attn_out + lora_o(out)

        return attn_out + ff_out

# transformer

@beartype
class PaLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        causal = True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        lora_r = 8,
        rotary_xpos_scale_base = 512,
        finetune_scopes = tuple(),
        cross_entropy_ignore_index = 0
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.num_tokens = num_tokens

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            block = Residual(ParallelTransformerBlock(
                dim = dim,
                causal = causal,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                xpos_scale_base = rotary_xpos_scale_base
            ))

            self.layers.append(block)

        self.norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        self.to_logits.weight = self.token_emb.weight

        nn.init.normal_(self.token_emb.weight, std=0.02)

        # fine tuning related

        self.lora_r = lora_r
        self.finetune_modules = nn.ModuleDict({})

        for scope in finetune_scopes:
            self.add_finetune_params(scope)

        # loss related

        self.cross_entropy_ignore_index = cross_entropy_ignore_index

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    def set_dropout(self, dropout):
        for module in self.layers.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        return self

    def add_finetune_params(self, scope, lora_r = None):
        assert scope not in self.finetune_modules, f'finetune scope {scope} already found'
        dim, dim_head, heads, r, device = self.dim, self.dim_head, self.heads, default(lora_r, self.lora_r), self.device

        q_inner_dim = heads * dim_head
        kv_inner_dim = dim_head

        lora_modules = nn.ModuleList([])

        for _ in range(len(self.layers)):
            lora_modules.append(nn.ModuleList([
                LoRA(dim, q_inner_dim, r = r),   # queries
                LoRA(dim, kv_inner_dim, r = r),  # keys
                LoRA(dim, kv_inner_dim, r = r),  # values
                LoRA(q_inner_dim, dim, r = r)    # wo
            ]))

        self.finetune_modules[scope] = lora_modules.to(device)

    def remove_finetune_params(self, scope):
        assert scope in self.finetune_modules, f'finetune scope {scope} not found'
        return self.finetune_modules.pop(scope)

    @torch.no_grad()
    def merge_finetune_params(self, scope):
        """ in the case one wants to merge the fine-tuned actor LORA parameters and do multiple rounds of fine tuning off different reward models """

        assert scope in self.finetune_modules, f'finetune scope {scope} not found'

        lora_modules = self.finetune_modules.pop(scope)

        for layer, (lora_q, lora_k, lora_v, lora_o) in zip(self.layers, lora_modules):
            block = layer.fn

            fused_attn_ff_weight = block.fused_attn_ff_proj.weight
            attn_out_weight = block.attn_out.weight

            fused_proj_out_dim = fused_attn_ff_weight.shape[0]

            lora_qkv_weight, _ = pack([lora_q.weight, lora_k.weight, lora_v.weight], 'i *')
            lora_qkv_weight = F.pad(lora_qkv_weight, (0, fused_proj_out_dim - lora_qkv_weight.shape[1]))

            lora_qkv_weight = rearrange(lora_qkv_weight, 'i o -> o i')
            lora_o_weight = rearrange(lora_o.weight, 'i o -> o i')

            fused_attn_ff_weight.add_(lora_qkv_weight)
            attn_out_weight.add_(lora_o_weight)

    # researcher train palm parameters first
    # before finetuning

    def palm_parameters(self):
        return set(self.parameters()) - set(self.finetune_modules.parameters())

    def finetune_parameters(self, scope = 'default'):
        assert scope in self.finetune_modules, f'finetune parameters of scope {scope} not found'
        return self.finetune_modules[scope].parameters()

    # generate function

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        seq_len,
        prompt = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        trainable = False,
        pad_value = 0.,
        eos_token = None,
        return_seq_without_prompt = True,
        use_tqdm = False,
        **kwargs
    ):
        if not exists(prompt):
            prompt = torch.randint(0, self.num_tokens, (1, 1))
            prompt = prompt.to(self.device)
            return_seq_without_prompt = False

        prompt, leading_dims = pack([prompt], '* n')

        n, out = prompt.shape[-1], prompt.clone()

        wrapper_fn = identity if not use_tqdm else tqdm
        sample_num_times = max(1, seq_len - prompt.shape[-1])

        for _ in wrapper_fn(range(sample_num_times)):
            logits, embeds = self.forward(out, return_logits_with_embedding = True, **kwargs)
            logits, embeds = logits[:, -1], embeds[:, -1]

            if exists(filter_logits_fn):
                logits = filter_logits_fn(logits, thres = filter_thres)

            sample = gumbel_sample(logits, temperature = temperature, dim = -1)
            out, _ = pack([out, sample], 'b *')

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, pad_value)
                    break

        out, = unpack(out, leading_dims, '* n')

        if return_seq_without_prompt:
            out = out[..., n:]

        return out

    def forward(
        self,
        x,
        return_loss = False,
        disable_lora = False,
        finetune_scope = None,
        extra_embed = None,
        return_only_embedding = False,
        return_logits_with_embedding = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        if exists(extra_embed):
            x = x + extra_embed

        if exists(finetune_scope) and not disable_lora:
            assert finetune_scope in self.finetune_modules
            finetune_modules = self.finetune_modules[finetune_scope]
        else:
            finetune_modules = ((None,) * len(self.layers))

        for layer, finetune_modules in zip(self.layers, finetune_modules):
            x = layer(x, finetune_modules = finetune_modules)

        embeds = self.norm(x)

        if return_only_embedding:
            return embeds

        logits = self.to_logits(x)

        ret = (logits, embeds) if return_logits_with_embedding else logits

        if not return_loss:
            return ret

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels, ignore_index = self.cross_entropy_ignore_index)

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

# PaLM with actor and critic heads

PPOActionCriticReturn = namedtuple('PPOActionCriticReturn', [
    'actions',
    'sequence',
    'mask',
    'prompt_mask',
    'action_logits',
    'values'
])

@beartype
class ActorCritic(nn.Module):
    def __init__(
        self,
        palm: PaLM,
        critic_palm: Optional[PaLM] = None,
        pooled_values = False,
        actor_lora = True,
        critic_lora = True,
        actor_lora_r = 8,
        critic_lora_r = 8,
        actor_lora_scope = 'actor',
        critic_lora_scope = 'critic',
        actor_dropout = 0.,
        critic_dropout = 0.
    ):
        super().__init__()
        self.actor_palm = palm

        self.critic_palm = critic_palm

        if not exists(self.critic_palm):
            self.critic_palm = copy.deepcopy(palm)

        self.actor_palm.set_dropout(actor_dropout)
        self.critic_palm.set_dropout(critic_dropout)

        self.actor_lora = actor_lora
        self.critic_lora = critic_lora

        self.actor_lora_scope = actor_lora_scope if actor_lora else None
        self.critic_lora_scope = critic_lora_scope if critic_lora else None

        if self.actor_lora:
            self.actor_palm.add_finetune_params(actor_lora_scope, lora_r = actor_lora_r)

        if self.critic_lora:
            self.critic_palm.add_finetune_params(critic_lora_scope, lora_r = critic_lora_r)

        self.pooled_values = pooled_values
        self.value_head = nn.Sequential(
            nn.Linear(palm.dim, 1),
            Rearrange('... 1 -> ...')
        )

        nn.init.zeros_(self.value_head[0].bias)
        nn.init.orthogonal_(self.value_head[0].weight, gain = math.sqrt(2))

    def actor_parameters(self):
        if not self.actor_lora:
            return self.actor_palm.parameters()

        return [
            *self.actor_palm.finetune_parameters(self.actor_lora_scope)
        ]

    def critic_parameters(self):
        if not self.actor_lora:
            return [*self.critic_palm.parameters(), *self.value_head.parameters()]

        return [
            *self.critic_palm.finetune_parameters(self.critic_lora_scope),
            *self.value_head.parameters()
        ]

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        state,
        max_seq_len,
        eos_token = None,
        return_values = False,
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

        mask = None
        if exists(eos_token):
            mask = ((sequence == eos_token).cumsum(dim = -1) == 0)
            mask = F.pad(mask, (1, -1), value = True) # include eos token

        action_logits, value = self.forward(
            sequence,
            mask = mask,
            return_values = return_values
        )        

        return PPOActionCriticReturn(
            actions,
            sequence,
            mask,
            prompt_mask,
            action_logits,
            value
        )

    def forward(
        self,
        x,
        mask = None,
        return_values = True
    ):
        action_logits = self.actor_palm(
            x,
            finetune_scope = self.actor_lora_scope
        )

        if not return_values:
            return action_logits, None

        critic_embeds = self.critic_palm(
            x,
            return_only_embedding = True,
            finetune_scope = self.critic_lora_scope
        )

        if self.pooled_values:
            critic_embeds = masked_mean(critic_embeds, mask, dim = 1)

        values = self.value_head(critic_embeds)

        return action_logits, values
