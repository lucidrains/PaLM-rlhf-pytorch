from beartype import beartype

import torch
from torch import nn

from palm_rlhf_pytorch.palm import PaLM
from palm_rlhf_pytorch.utils import eval_decorator, top_k

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

@beartype
class PaLMEncDec(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        enc_depth = None,
        dec_default_start_token_id = None,
        causal = True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        lora_r = 8,
        rotary_xpos_scale_base = 512,
        finetune_scopes = tuple(),
        cross_entropy_ignore_index = 0,
    ):
        super().__init__()
        self.dim = dim

        enc_depth = default(enc_depth, depth)

        palm_kwargs = dict(
            dim = dim,
            num_tokens = num_tokens,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            lora_r = lora_r,
            finetune_scopes = finetune_scopes
        )

        self.encoder = PaLM(
            depth = enc_depth,
            causal = False,
            **palm_kwargs
        )

        self.decoder = PaLM(
            depth = depth,
            causal = True,
            cross_attend = True,
            rotary_xpos_scale_base = rotary_xpos_scale_base,
            cross_entropy_ignore_index = cross_entropy_ignore_index,
            default_start_token_id = dec_default_start_token_id,
            **palm_kwargs
        )

        self.encoder.token_emb = self.decoder.token_emb

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    def set_dropout(self, dropout):
        self.encoder.set_dropout(dropout)
        self.decoder.set_dropout(dropout)
        return self

    def add_finetune_params(self, scope, lora_r = None):
        self.encoder.add_finetune_params(scope, lora_r = lora_r)
        self.decoder.add_finetune_params(scope, lora_r = lora_r)

    def remove_finetune_params(self, scope):
        self.encoder.remove_finetune_params(scope)
        self.decoder.remove_finetune_params(scope)

    @torch.no_grad()
    def merge_finetune_params(self, scope):
        self.encoder.merge_finetune_params(scope)
        self.decoder.merge_finetune_params(scope)

    def palm_parameters(self):
        return set(self.decoder.palm_parameters()) | set(self.encoder.palm_parameters())

    def finetune_parameters(self, scope = 'default'):
        return set(self.decoder.finetune_parameters(scope = scope)) | set(self.encoder.finetune_parameters(scope = scope))

    # generate function

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        seq_len,
        prompt,
        prompt_mask = None,
        decoder_prompt = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        pad_value = 0.,
        eos_token = None,
        return_seq_without_prompt = True,
        use_tqdm = False,
        extra_prompt_embed = None,
        **kwargs
    ):
        prompt_embed, prompt_mask = self.encode_prompt(
            prompt,
            prompt_mask,
            extra_prompt_embed = extra_prompt_embed,
            **kwargs
        )

        generated = self.decoder.generate(
            seq_len,
            prompt = decoder_prompt,
            temperature = temperature,
            filter_logits_fn = filter_logits_fn,
            filter_thres = filter_thres,
            pad_value = pad_value,
            eos_token = eos_token,
            return_seq_without_prompt = return_seq_without_prompt,
            use_tqdm = use_tqdm,
            context = prompt_embed,
            context_mask = prompt_mask,
            **kwargs
        )

        return generated

    def encode_prompt(
        self,
        prompt,
        prompt_mask = None,
        disable_lora = False,
        finetune_scope = None,
        extra_prompt_embed = None
    ):
        # treat any token ids that are negative as tokens to mask out

        if not exists(prompt_mask):
            prompt_mask = prompt >= 0
            prompt = prompt.masked_fill(~prompt_mask, 0)

        prompt_embed = self.encoder(
            prompt,
            mask = prompt_mask,
            disable_lora = disable_lora,
            finetune_scope = finetune_scope,
            extra_embed = extra_prompt_embed,
            return_only_embedding = True
        )

        return prompt_embed, prompt_mask

    def forward(
        self,
        prompt,
        *,
        decoder_seq = None,
        prompt_mask = None,
        return_loss = False,
        disable_lora = False,
        finetune_scope = None,
        extra_embed = None,
        extra_prompt_embed = None,
        return_only_embedding = False,
        return_logits_with_embedding = False
    ):

        prompt_embed, prompt_mask = self.encode_prompt(
            prompt,
            prompt_mask,
            disable_lora = disable_lora,
            finetune_scope = finetune_scope,
            extra_prompt_embed = extra_prompt_embed
        )

        return self.decoder(
            prompt = decoder_seq,
            context = prompt_embed,
            context_mask = prompt_mask,
            return_loss = return_loss,
            disable_lora = disable_lora,
            finetune_scope = finetune_scope,
            extra_embed = extra_embed,
            return_only_embedding = return_only_embedding,
            return_logits_with_embedding = return_logits_with_embedding
        )
