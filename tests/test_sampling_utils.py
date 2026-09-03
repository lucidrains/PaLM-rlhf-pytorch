"""Regression tests for probability-based logit filtering."""

import torch

from palm_rlhf_pytorch.utils import top_p


def test_top_p_restores_vocab_order_after_sorting():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]])
    original = logits.clone()

    filtered = top_p(logits, thres=0.5)

    expected = torch.tensor(
        [[4.0, -float("inf"), -float("inf"), -float("inf")],
         [-float("inf"), 4.0, -float("inf"), -float("inf")]],
    )
    assert torch.equal(filtered, expected)
    assert torch.equal(logits, original)


def test_top_p_keeps_gradients_on_original_token_positions():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], requires_grad=True)

    filtered = top_p(logits, thres=0.5)
    torch.where(torch.isfinite(filtered), filtered, torch.zeros_like(filtered)).sum().backward()

    assert torch.equal(logits.grad, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

