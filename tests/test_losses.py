import math

import torch

from drishyam.losses import brier, ce_soft, question_loss, rps


def test_ce_brier_hand_computed():
    logits = torch.tensor([[2.0, 0.0]])
    gold = torch.tensor([[1.0, 0.0]])
    p1 = math.exp(2.0) / (math.exp(2.0) + 1.0)  # 0.8808
    ce = -math.log(p1)
    brier_v = (p1 - 1.0) ** 2 + ((1 - p1) - 0.0) ** 2
    assert torch.allclose(ce_soft(logits, gold), torch.tensor([ce]), atol=1e-5)
    assert torch.allclose(brier(logits, gold), torch.tensor([brier_v]), atol=1e-5)
    # spec: choice/noul loss = CE + 0.5*Brier
    expected = ce + 0.5 * brier_v
    assert torch.allclose(question_loss(logits, gold, "noul"), torch.tensor([expected]), atol=1e-5)


def test_rps_hand_computed():
    logits = torch.zeros(1, 4)
    gold = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    cp = torch.tensor([0.25, 0.5, 0.75]).cumsum(0)  # uniform softmax cumsum (first 3 steps)
    cg = torch.tensor([0.1, 0.3, 0.6])
    rps_v = float((((cp - cg) ** 2).sum()) / 3)
    assert torch.allclose(rps(logits, gold), torch.tensor([rps_v]), atol=1e-5)
    ce = -1.0 * math.log(0.25)
    expected = ce + 0.5 * rps_v  # spec: score loss = CE + 0.5*RPS
    assert torch.allclose(question_loss(logits, gold, "score"), torch.tensor([expected]), atol=1e-4)


def test_choice_order_shuffled_loss_invariant():
    """CE+Brier over options is invariant to a consistent permutation of options+gold."""
    torch.manual_seed(0)
    logits = torch.randn(1, 3)
    gold = torch.tensor([[0.2, 0.5, 0.3]])
    perm = [2, 0, 1]
    got = question_loss(logits[:, perm], gold[:, perm], "choice")
    base = question_loss(logits, gold, "choice")
    assert torch.allclose(got, base, atol=1e-6)
