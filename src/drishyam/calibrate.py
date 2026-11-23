"""G1-C1: post-hoc per-primitive temperature scaling, fitted on VAL NLL only (spec §2)."""

import json
from pathlib import Path

import numpy as np

ART = Path("artifacts")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """softmax(log p / T) — log-prob is the exact softmax inverse (constant shifts cancel)."""
    return _softmax(np.log(np.clip(probs, 1e-9, 1.0)) / t)


def fit_temperature(probs: np.ndarray, golds: np.ndarray, primitive: str) -> float:
    """Grid + parabolic refine on soft-gold NLL; primitive param keeps the table explicit."""
    del primitive  # fitted per-primitive by the caller; asserted in fit_all
    logits = np.log(np.clip(np.atleast_2d(probs), 1e-9, 1.0))
    g = np.atleast_2d(golds)

    def nll(t: float) -> float:
        return float(-np.mean([gi @ np.log(_softmax(li / t))
                               for li, gi in zip(logits, g, strict=False)]))

    grid = np.concatenate([np.linspace(0.2, 5.0, 97), [1.0]])
    ts = sorted(set(grid.tolist()))
    vals = [nll(t) for t in ts]
    best = int(np.argmin(vals))
    lo = ts[max(0, best - 1)]
    hi = ts[min(len(ts) - 1, best + 1)]
    fine = np.linspace(lo, hi, 41)
    fvals = [nll(t) for t in fine]
    return float(fine[int(np.argmin(fvals))])


def fit_all(npz_path: Path = ART / "val_logits.npz") -> dict:
    d = np.load(npz_path, allow_pickle=True)
    probs, golds, prims = d["probs"], d["gold"], d["prims"]
    table = {}
    for prim in ("noul", "choice", "score"):
        sel = prims == prim
        if not sel.any():
            continue
        p = np.stack([np.asarray(x, dtype=float) for x in probs[sel]])
        g = np.stack([np.asarray(x, dtype=float) for x in golds[sel]])
        table[prim] = {"T": fit_temperature(p, g, prim), "n": int(sel.sum())}
    return table


def main() -> None:
    table = fit_all()
    ART.mkdir(exist_ok=True)
    (ART / "temperatures.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
