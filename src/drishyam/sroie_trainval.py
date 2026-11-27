"""SROIE trainval -> verification training rows (v5-A domain adaptation).

Source: runs/hub_ref/sroie/jsdnrs_ref/data/train-00000-of-00001.parquet
(626 real scanned receipts, public Task-3 KIE gold; key-verified 0 overlap
with the 361-receipt official test pool used by eval_sroie).

Rows mirror the eval protocol EXACTLY (same template, same 50/50, same seed
family): per receipt x {total,date,company} x {TRUE gold-verbatim, FALSE
cross-receipt draw never equal to gold}, rng keyed per (key, field).

One receipt missing any gold field is dropped entirely (gold completeness
law: a verification claim needs a verbatim gold string).

Carve: deterministic sorted-keys split, seed 2027 -> train/val receipts.
Output: gate2_labels_sroie_train.jsonl / gate2_labels_sroie_val.jsonl with
page_key = "embeddings_sroie/emb_000000.npy:{idx}" (idx = receipt order in
the SAME sorted enumeration the embed script uses — the two files are a pair).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "runs/hub_ref/sroie/jsdnrs_ref/data/train-00000-of-00001.parquet"
OUT_DIR = ROOT / "artifacts"
FIELDS = ("total", "date", "company")
CARVE_SEED = 2027
CLAIM_SEED = 2026  # same seed family as eval_sroie (different receipt ids)
BANK_REL = "artifacts/embeddings_sroie/emb_000000.npy"  # ROOT-relative (PageCache law)


def load_pool() -> list[dict]:
    """Receipts with complete gold for all three fields, sorted by key."""
    df = pd.read_parquet(PARQUET)
    pool = []
    for i in range(len(df)):
        r = df.iloc[i]
        ent = r["entities"] or {}
        gold = {f: str(ent.get(f, "")).strip() for f in FIELDS}
        if any(not v for v in gold.values()):
            continue
        pool.append({"key": str(r["key"]), "gold": gold})
    pool.sort(key=lambda x: x["key"])
    return pool


def draw_false(pool: list[dict], field: str, own_key: str, rng_seed: int) -> str:
    """Cross-receipt value for `field`, deterministic, never equal to own gold."""
    h = hashlib.sha256(f"{rng_seed}:{own_key}:{field}".encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    own = next(r["gold"][field] for r in pool if r["key"] == own_key)
    for _ in range(64):
        other = pool[rng.randrange(len(pool))]
        v = other["gold"][field]
        if other["key"] != own_key and v != own:
            return v
    raise ValueError(f"no false draw possible for {own_key}/{field}")


def build_rows(pool: list[dict], full_pool: list[dict], bank_rel: str = BANK_REL) -> list[dict]:
    """Verification label rows: 6 per receipt, exact 50/50, one question each.

    page idx is ALWAYS the receipt's position in full_pool's sorted enumeration
    (the embed bank's order) — train and val carves share one bank.
    False-claim draws come from full_pool: a val receipt's gold can appear as a
    FALSE claim in train rows, which can only UNDERSTATE val scores (safe
    direction); true val golds are never trained as positives.
    """
    key_idx = {r["key"]: i for i, r in enumerate(full_pool)}
    rows = []
    for r in pool:
        for field in FIELDS:
            gold = r["gold"][field]
            for claim, answer in ((gold, "Yes"),
                                  (draw_false(full_pool, field, r["key"], CLAIM_SEED), "No")):
                ctype = "true" if answer == "Yes" else "false"
                rows.append({
                    "row_id": f"{r['key']}:{field}:{ctype}",
                    "page_key": f"{bank_rel}:{key_idx[r['key']]}",
                    "family": "sroie",
                    "seed": 0,
                    "questions": [{
                        "qid": f"{r['key']}:{field}:{ctype}",
                        "primitive": "noul",
                        "text": question_text(field, claim),
                        "options": ["No", "Yes"],
                    }],
                    "gold": [[1.0, 0.0] if answer == "No" else [0.0, 1.0]],
                })
    return rows


def question_text(field: str, claimed: str) -> str:
    """MUST match eval_sroie.question_template — training form = eval form."""
    from drishyam.eval_sroie import question_template
    return question_template(field, claimed)


def carve(pool: list[dict], n_val: int = 100) -> tuple[list[dict], list[dict]]:
    keys = sorted(r["key"] for r in pool)
    rng = random.Random(CARVE_SEED)
    val_keys = set(rng.sample(keys, n_val))
    train = [r for r in pool if r["key"] not in val_keys]
    val = [r for r in pool if r["key"] in val_keys]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=100)
    a = ap.parse_args()

    pool = load_pool()
    assert len({r["key"] for r in pool}) == len(pool)
    train, val = carve(pool, a.n_val)
    tr_rows = build_rows(train, pool)
    va_rows = build_rows(val, pool)
    for name, rows in (("sroie_train", tr_rows), ("sroie_val", va_rows)):
        n_true = sum(r["gold"][0][1] == 1.0 for r in rows)
        n_false = len(rows) - n_true
        assert n_true == n_false, f"{name}: not 50/50 ({n_true}/{n_false})"
        assert len({r["row_id"] for r in rows}) == len(rows)
        with (OUT_DIR / f"gate2_labels_{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} rows ({n_true} true / {n_false} false), "
              f"{len({r['page_key'] for r in rows})} pages")
    print(f"pool {len(pool)} receipts -> train {len(train)} / val {len(val)}; "
          f"bank page order = sorted keys of each carve")


if __name__ == "__main__":
    main()
