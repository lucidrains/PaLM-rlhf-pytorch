# Free Process Rewards without Process Labels 
# Yuan et al.  https://arxiv.org/abs/2412.01981 - paper that led to Prime

from __future__ import annotations
from copy import deepcopy

import torch
from torch.nn import Module
from torch.nn.functional import logsigmoid

from einops import rearrange

# helpers

def exists(v):
    return v is not None

def get_logprob_at(logits, seq):
    log_probs = logits.log_softmax(dim = -1)
    seq = rearrange(seq, '... -> ... 1')
    log_prob = log_probs.gather(-1, seq)
    return rearrange(log_prob, '... 1 -> ...')

class ImplicitPRM(Module):
    """ PRM stands for process reward model, an openai paper that shows that rewarding the steps a model takes to its outcome is better than only rewarding based on final answer or outcome. basically same as when a teacher gives you some credit for showing your steps on an exam """

    def __init__(
        self,
        model: Module,
        ref_model: Module | None = None,
        beta = 0.1
    ):
        super().__init__()
        self.model = model

        # only drawback to this technique is needing a reference model

        if not exists(ref_model):
            ref_model = deepcopy(model)

        self.ref_model = ref_model
        ref_model.requires_grad_(False) # insurance

        self.beta = beta

    def parameters(self):
        return self.model.parameters() # only main model is trained

    def forward(
        self,
        seq,
        labels = None
    ):
        source_seq, target_seq = seq[:, :-1], seq[:, 1:]

        mask = target_seq >= 0 # assume any token ids < 0 to be padding

        model_logits = self.model(source_seq)
        ref_model_logits = self.ref_model(source_seq)

        log_prob = get_logprob_at(model_logits, target_seq)
        ref_log_prob = get_logprob_at(ref_model_logits, target_seq)

        # main formula is DPO-like, and has some connection with Q-learning https://arxiv.org/abs/2404.12358 . it is all connected

        implicit_rewards = self.beta * (log_prob - ref_log_prob)

        # zero out rewards in padding

        implicit_rewards = implicit_rewards.masked_fill(~mask, 0.)

        # early return if not training, as in Prime with alternating model and prm training

        if not exists(labels):
            return implicit_rewards

        labels = rearrange(labels, 'b -> b 1')

        # otherwise use the cross entropy formulation from their paper (eq 5)

        loss = (
            labels * logsigmoid(implicit_rewards) +
            (1. - labels) * logsigmoid(-implicit_rewards)  # (1. - sigmoid(x)) == sigmoid(-x)
        )

        return loss[mask].mean()

# make it easy for others to copy paste into another project

if __name__ == '__main__':
    from palm_rlhf_pytorch import PaLM

    palm = PaLM(
        num_tokens = 256,
        dim = 64,
        depth = 2
    )

    ref_palm = PaLM(
        num_tokens = 256,
        dim = 64,
        depth = 2
    )

    implicit_prm = ImplicitPRM(
        palm,
        ref_model = ref_palm
    )

    # mock data

    seq = torch.randint(0, 256, (2, 1024))
    labels = torch.randint(0, 2, (2,))

    loss = implicit_prm(seq, labels)
    loss.backward()

    # after much training

    implicit_rewards = implicit_prm(seq) # Float[2, 1024]

    # there you go, free process reward model
    # now you can use this dense reward for rlhf, beam search whatever
