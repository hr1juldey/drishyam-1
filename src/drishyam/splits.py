"""G1-D2: stratified 240/60 per seen workflow; unseen test-only (its train never loaded)."""

import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from drishyam.data_inspect import REPO, SEEN, UNSEEN, parquet_sha, q_prim
from drishyam.schemas import SplitSpec

OUT = Path("splits")


def stratify_240_60(n: int, keys: list[str], seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_key: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)
    train, val = [], []
    for idxs in by_key.values():
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * 60 / n))
        val += idxs[:n_val]
        train += idxs[n_val:]
    return sorted(train), sorted(val)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    spec = SplitSpec(
        version="v1", seed=42, parquet_sha256={}, train={}, val={},
        test_indices=[], unseen_workflow=UNSEEN, unseen_test_indices=[],
    )
    for wf in [*SEEN, UNSEEN]:
        # byte hash only — the unseen workflow's train rows are NEVER deserialized
        spec.parquet_sha256[f"{wf}/train"] = parquet_sha(wf, "train")
        spec.parquet_sha256[f"{wf}/test"] = parquet_sha(wf, "test")
    for wf in SEEN:
        ds = load_dataset(REPO, wf, split="train")
        keys = ["+".join(sorted(q_prim(m) for m in json.loads(r["questions"]).values()))
                for r in ds]
        tr, va = stratify_240_60(len(ds), keys, seed=42)
        spec.train[wf], spec.val[wf] = tr, va
        print(wf, "train", len(tr), "val", len(va),
              "val-key-dist", dict(Counter(keys[i] for i in va)))
    spec.test_indices = list(range(100))
    spec.unseen_test_indices = list(range(100))
    assert all(set(spec.train[wf]).isdisjoint(spec.val[wf]) for wf in SEEN)
    assert spec.unseen_workflow not in spec.train and spec.unseen_workflow not in spec.val
    (OUT / "split_v1.json").write_text(spec.model_dump_json(indent=2))
    print("SPLITS WRITTEN: 3x(240/60/100) + unseen 100 (train never loaded)")


if __name__ == "__main__":
    main()
