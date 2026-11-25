"""RLCD reward math: every term is strictly proper — gold must maximize it."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishyam.train_gate2_rlcd import reward_of, rps, spherical


def test_proper_scoring():
    torch.manual_seed(0)
    logp = F.log_softmax(torch.randn(8, 3), dim=-1)
    gold = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    mask = torch.zeros(8, dtype=torch.bool)

    r = reward_of(logp, gold, mask)
    # gold answer must score strictly higher than any wrong answer, every row
    for i in range(8):
        for j in range(3):
            if j != gold[i]:
                # perturb: push prob mass to the wrong option, recompute
                pert = torch.log_softmax(
                    torch.stack([logp[i].exp() * 0.05,
                                 F.one_hot(torch.tensor(j), 3).float() * 0.95]).sum(0).log(),
                    dim=-1)
                r_pert = reward_of(pert[None], gold[i][None], mask[i][None])
                assert r_pert.item() < r[i].item(), (i, j)

    # spherical: at the gold one-hot point it is maximal (=1 per unit mass)
    p = torch.tensor([[0.7, 0.2, 0.1]])
    g = torch.tensor([0])
    assert spherical(p, g).item() > spherical(
        torch.tensor([[0.2, 0.7, 0.1]]), g).item()

    # RPS ordinal: distribution closer to gold CDF scores higher
    p_far = torch.tensor([[0.05, 0.05, 0.05, 0.05, 0.8]])  # mass at L5
    p_near = torch.tensor([[0.8, 0.05, 0.05, 0.05, 0.05]])  # mass at L1
    g1 = torch.tensor([0])
    assert rps(p_near, g1).item() > rps(p_far, g1).item()

    # score rows get the RPS bonus; non-score rows don't
    logp5 = F.log_softmax(torch.randn(4, 5), dim=-1)
    gold5 = torch.tensor([0, 1, 2, 3])
    m5 = torch.tensor([True, False, True, False])
    r_all = reward_of(logp5, gold5, m5)
    r_none = reward_of(logp5, gold5, torch.zeros(4, dtype=torch.bool))
    assert torch.allclose(r_all[[1, 3]], r_none[[1, 3]])  # non-score unchanged
    assert not torch.allclose(r_all[[0, 2]], r_none[[0, 2]])  # score rows differ
