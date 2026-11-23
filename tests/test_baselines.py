import numpy as np

from drishyam.baselines import majority_dist, metrics_rows


def test_majority_dist_hand_computed():
    golds = [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8]]  # option0 mass 2.0, option1 1.0
    dist = majority_dist(golds)
    assert np.allclose(dist, [2 / 3, 1 / 3])


def test_metrics_rows_hand_computed():
    rows = [
        {"prim": "choice", "probs": np.array([0.8, 0.2]), "gold": np.array([1.0, 0.0])},
        {"prim": "choice", "probs": np.array([0.6, 0.4]), "gold": np.array([0.5, 0.5])},
        {"prim": "score", "probs": np.array([0.1, 0.1, 0.7, 0.1]),
         "gold": np.array([0, 0, 1.0, 0])},
    ]
    m = metrics_rows(rows)
    # all three correct: q1 0.8>0.2=idx0=gold idx0; q2 tie argmax=0 == gold argmax 0; q3 idx2==idx2
    assert abs(m["acc"] - 1.0) < 1e-9
    # brier q1: (0.8-1)^2+(0.2-0)^2 = 0.08 ; q2: 2*(0.1)^2 = 0.02 ; q3: 3*(0.01)+0.09 = 0.12
    assert abs(m["brier"] - (0.08 + 0.02 + 0.12) / 3) < 1e-9
    # score MAE: |2-2| / 1 score question = 0
    assert m["score_mae"] == 0.0
    assert "ece15" in m and 0.0 <= m["ece15"] <= 1.0


def test_metrics_flatten_identical():
    r0 = {"prim": "noul", "probs": np.array([0.3, 0.7]), "gold": np.array([0.2, 0.8])}
    one = metrics_rows([r0])
    flat = metrics_rows([{**r0, "probs": r0["probs"].reshape(1, -1)[0]}])
    assert abs(one["acc"] - flat["acc"]) < 1e-12
