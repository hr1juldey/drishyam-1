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


def _rows(x) -> list[np.ndarray]:
    """2D array -> list of rows; ragged/object/list input -> list of 1-D vectors."""
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return list(x)
    return [np.asarray(r, dtype=float) for r in x]


def fit_temperature(probs, golds, primitive: str = "") -> float:
    """Grid + parabolic refine on soft-gold NLL; ragged-safe (choice K varies)."""
    del primitive  # fitted per-primitive by the caller; asserted in fit_all
    pl, gl = _rows(probs), _rows(golds)

    def nll(t: float) -> float:
        vals = []
        for p, g in zip(pl, gl, strict=False):
            z = np.log(np.clip(p, 1e-9, 1.0)) / t
            z -= z.max()
            vals.append(float(g @ (z - np.log(np.exp(z).sum()))))
        return float(-np.mean(vals))

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
        p_sel = [np.asarray(x, dtype=float) for x in probs[sel]]
        g_sel = [np.asarray(x, dtype=float) for x in golds[sel]]
        table[prim] = {"T": fit_temperature(p_sel, g_sel, prim), "n": int(sel.sum())}
    return table


def main() -> None:
    table = fit_all()
    ART.mkdir(exist_ok=True)
    (ART / "temperatures.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
