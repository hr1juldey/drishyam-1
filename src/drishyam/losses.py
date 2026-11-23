"""Strictly proper scoring losses (spec §3): CE + 0.5*Brier | CE + 0.5*RPS."""

import torch
import torch.nn.functional as F


def ce_soft(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Soft-target cross entropy over options; gold rows sum to 1."""
    return -(gold * F.log_softmax(logits, dim=-1)).sum(-1)


def brier(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    return ((F.softmax(logits, dim=-1) - gold) ** 2).sum(-1)


def rps(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Ranked Probability Score over the cumulative distribution (ordinal levels)."""
    k = gold.size(-1)
    return ((F.softmax(logits, dim=-1).cumsum(-1) - gold.cumsum(-1)) ** 2).sum(-1) / (k - 1)


def question_loss(logits: torch.Tensor, gold: torch.Tensor, primitive: str) -> torch.Tensor:
    """Per-question scalar: noul/choice -> CE + 0.5*Brier; score -> CE + 0.5*RPS (spec §3)."""
    pen = rps if primitive == "score" else brier
    return ce_soft(logits, gold) + 0.5 * pen(logits, gold)
