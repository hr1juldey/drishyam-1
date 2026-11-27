"""Embed SROIE trainval receipts -> fp16 page bank for v5-A domain adaptation.

Page order = sroie_trainval.load_pool()'s sorted, gold-complete keys — the
same enumeration the label rows' page_key indices point into. Convention is
byte-identical to eval_sroie's test-side embed (JPEG -> fp32 [0,1] ->
bilinear 896x896 -> frozen tower -> projector, fp32; bank stored fp16).
The tower is frozen across all runs, so bank content is checkpoint-invariant.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PARQUET = ROOT / "runs/hub_ref/sroie/jsdnrs_ref/data/train-00000-of-00001.parquet"
OUT_DIR = ROOT / "artifacts/embeddings_sroie"


def load_pool_keys() -> list[str]:
    from drishyam.sroie_trainval import load_pool
    return [r["key"] for r in load_pool()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/g2-row-sft-v4/best.pt",
                    help="tower source (tower is frozen; bank is ckpt-invariant)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="re-embed even if a consistent bank manifest already exists")
    a = ap.parse_args()

    manifest = OUT_DIR / "MANIFEST.json"
    if manifest.exists() and not a.force:
        m = json.loads(manifest.read_text())
        shape = np.load(OUT_DIR / "emb_000000.npy", mmap_mode="r").shape
        if tuple(shape) == tuple(m["shape"]) and m["n"] == len(load_pool_keys()):
            print(f"bank already consistent ({m['n']} pages) — skipping embed (use --force to redo)",
                  flush=True)
            return

    from drishyam.sroie_trainval import load_pool
    pool = load_pool()
    want = [r["key"] for r in pool]

    df = pd.read_parquet(TRAIN_PARQUET)
    bytes_by_key = {str(r["key"]): bytes(r["image"]["bytes"]) for _, r in df.iterrows()}
    missing = [k for k in want if k not in bytes_by_key]
    assert not missing, f"trainval keys missing from parquet: {missing[:5]}"
    receipts = [{"key": k, "image_bytes": bytes_by_key[k]} for k in want]

    from drishyam.eval_jevbench import load_model
    from drishyam.eval_sroie import embed_receipts
    model, _tok, _fl, _sl = load_model(ROOT / a.ckpt, a.device)
    model.float()
    t0 = time.time()
    embs = embed_receipts(model, receipts, a.device)
    print(f"embedded {len(embs)} receipts in {time.time() - t0:.1f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = [r["key"] for r in receipts]
    arr = np.stack([embs[k].numpy() for k in order]).astype(np.float16)
    np.lib.format.open_memmap(OUT_DIR / "emb_000000.npy", mode="w+",
                              dtype=np.float16, shape=arr.shape)[...] = arr
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps({
        "keys": order, "n": len(order), "shape": list(arr.shape), "ckpt": a.ckpt,
        "convention": "JPEG->fp32[0,1]->bilinear 896x896->tower->projector fp32, bank fp16",
    }, indent=2))
    print(f"bank: {OUT_DIR / 'emb_000000.npy'} shape={arr.shape}")


if __name__ == "__main__":
    main()
