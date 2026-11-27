#!/usr/bin/env python
"""Gate: SROIE trainval bank vs label rows consistency (v5-A pre-train gate).

Fails loudly unless: bank exists with the manifest's shape, 625 pages, and
every label row's page_key index maps to the right receipt key.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "artifacts/embeddings_sroie/emb_000000.npy"
MANIFEST = ROOT / "artifacts/embeddings_sroie/MANIFEST.json"

import numpy as np  # noqa: E402

m = json.load(open(MANIFEST))
mm = np.load(BANK, mmap_mode="r")
assert mm.shape[0] == m["n"] == 625, f"bank/manifest mismatch: {mm.shape} vs n={m['n']}"
assert list(mm.shape[1:]) == [256, 640], f"bad page shape: {mm.shape}"
keys = m["keys"]
checked = 0
for f in ("gate2_labels_sroie_train.jsonl", "gate2_labels_sroie_val.jsonl"):
    for line in (ROOT / "artifacts" / f).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        path, idx = r["page_key"].rsplit(":", 1)
        assert path == "artifacts/embeddings_sroie/emb_000000.npy", r["page_key"]
        i = int(idx)
        assert 0 <= i < len(keys), (r["row_id"], i)
        assert keys[i] == r["row_id"].split(":")[0], (r["row_id"], keys[i])
        checked += 1
print(f"bank/labels consistent: {mm.shape[0]} pages, {checked} rows validated")
sys.exit(0)
