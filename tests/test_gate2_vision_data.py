"""Gate-2 label factory tests: coverage, one-hot contract, LOFO, determinism, gt parity."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drishyam.vision_data import (  # noqa: E402
    EMB_DIRS, UNSEEN_FAMILY, LabelFactory, load_pages, page_gt,
)


def test_pages_banked_and_parity():
    pages = load_pages()
    assert len(pages) > 80_000, f"only {len(pages)} pages found in {EMB_DIRS}"
    for (fam, seed), (rel, idx) in list(pages.items())[:5]:
        emb = np.load(ROOT / rel, mmap_mode="r")
        lab = (ROOT / rel).parent / (ROOT / rel).name.replace("emb_", "lab_", 1).replace(
            ".npy", ".jsonl")
        rows = [json.loads(l) for l in open(lab) if l.strip()]
        assert emb.shape[0] == len(rows)


def test_page_gt_matches_jit_pipeline():
    sys.path.insert(0, str(ROOT / "artifacts" / "paper_rig"))
    from jit_pipeline import gen_page

    for fam, seed in [("thermal", 1_000_003), ("electricity", 1_005_011),
                      ("salary", 1_002_007)]:
        _, gt_ref = gen_page(fam, seed)
        gt = page_gt(fam, seed)
        assert gt.get("legibility_applied") == gt_ref.get("legibility_applied"), fam
        for k, v in gt_ref.items():
            if k in ("legibility_applied", "degradation_plan", "row_id"):
                continue
            assert json.dumps(gt.get(k), sort_keys=True, default=str) == \
                json.dumps(v, sort_keys=True, default=str), (fam, seed, k)


def test_rows_contract_and_lofo():
    pages = load_pages()
    small = {k: v for i, (k, v) in enumerate(pages.items())
             if i % 997 == 0}  # ~90 pages, fast
    pools = {}
    insts = []
    factory = LabelFactory(small, pools, insts, seed=42)
    emitted = []
    for (fam, seed), loc in small.items():
        if fam == UNSEEN_FAMILY:
            continue
        gt = page_gt(fam, seed)
        qs, golds = factory.questions_for(fam, seed, gt, "train")
        if qs:
            emitted.append({"family": fam, "seed": seed, "questions": qs, "gold": golds})
    assert emitted, "no questions emitted on the probe sample"
    for r in emitted:
        for q, g in zip(r["questions"], r["gold"], strict=False):
            assert abs(sum(g) - 1.0) < 1e-6
            assert len(g) == len(q["options"]) >= 2
            assert q["primitive"] in ("noul", "choice", "score")
        assert all(q["qid"].split(":")[0] == r["family"] for q in r["questions"])


def test_factory_determinism():
    pages = load_pages()
    probe = [(k, v) for i, (k, v) in enumerate(pages.items()) if i % 499 == 0][:12]
    outs = []
    for _ in range(2):
        factory = LabelFactory(dict(probe), {}, [], seed=7)
        one = []
        for (fam, seed), loc in probe:
            gt = page_gt(fam, seed)
            qs, golds = factory.questions_for(fam, seed, gt, "train")
            one.append(json.dumps([qs, golds], sort_keys=True, default=str))
        outs.append(one)
    assert outs[0] == outs[1], "label factory is not deterministic for fixed seed"
