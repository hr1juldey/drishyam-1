#!/usr/bin/env python
"""Hermes-harness teacher interface.

The teacher is NEVER wired directly to an API. Instead the harness emits
work-order files + a labeling contract; Hermes subagents (the GLM provider
Hermes itself runs on) read them, label the cases, and write draft files;
the deterministic referee (same alias table as the bake-off) scores the
drafts. All question types — choice, noul (yes/no), score — go through the
same harness; there is no direct path for any of them.

Modes:
  prepare --cases 20 --batch-size 5   write work orders + CONTRACT.md
  score    --manifest .../manifest.json  validate + score drafts -> metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bakeoff import (  # noqa: E402
    answers_to_gold_keys,
    build_alias_map,
    few_shot_examples,
    mean_kl,
    sample_cases,
)

HARNESS_DIR = ROOT / "drafts" / "harness"

CONTRACT_PREAMBLE = """# LABELING CONTRACT (read fully before labeling)

You are labeling operational decision cases for a training dataset. For each
case you receive a STATE (a JSON scenario) and 5 QUESTIONS. For each question
you output a probability distribution over its option keys, summing to 1.

## Output format (exact)
One JSON object per case, all cases collected in one file:
{"cases": [{"case_id": "<as given>", "answers": {"<qid>": {"<key>": p, ...}}}]}

## Key rules
1. Answer keys: use the option key as given in the question's criteria (for
   numeric ranks '0'..'4' use the NUMBER; for named options use the NAME).
   Never include description text in keys.
2. Every question must list ALL its option keys with probabilities.
3. Probabilities must sum to 1.0 (within 0.02) and every entry must be >= 0.
4. Use your judgment on the MERITS of the case. These are gray decisions —
   expressing uncertainty is correct. But do not flip a yes/no to its opposite
   confidently when the evidence is mixed: if a binary is genuinely close,
   say so in the distribution (e.g. {"true": 0.55, "false": 0.45}).
5. Strictness calibration comes from the worked examples below — match the
   EXAMPLES' level of strictness, not a stricter or laxer reading.
6. Do not consult anything outside the contract and the work order. Label from
   the case itself.

## Worked examples (style + strictness anchors; do NOT copy their answers)
"""


def prepare(n_cases: int, batch_size: int, seed: int) -> Path:
    cases = sample_cases(n_cases, seed)
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    contract = HARNESS_DIR / "CONTRACT.md"
    ex = few_shot_examples(seed)
    parts = [CONTRACT_PREAMBLE]
    for i, e in enumerate(ex, 1):
        parts.append(f"EXAMPLE {i} STATE:\n{e['state']}\n")
        parts.append(f"EXAMPLE {i} QUESTIONS:\n{json.dumps(e['questions'], ensure_ascii=False)}\n")
        ans_json = json.dumps({"answers": e["answers"]}, ensure_ascii=False)
        parts.append(f"EXAMPLE {i} ANSWERS (style anchor):\n{ans_json}\n")
    contract.write_text("\n".join(parts))

    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed,
                "n_cases": len(cases), "batch_size": batch_size,
                "labeler": "hermes-subagent (inherited provider)",
                "batches": [], "cases": []}
    for b in range(0, len(cases), batch_size):
        chunk = cases[b:b + batch_size]
        wo = [{"case_id": f"case_{b + j:03d}", "workflow": c["workflow"],
               "state": c["state"], "questions": c["questions"]}
              for j, c in enumerate(chunk)]
        wo_path = HARNESS_DIR / f"wo_{b // batch_size:02d}.json"
        wo_path.write_text(json.dumps(wo, ensure_ascii=False, indent=1))
        manifest["batches"].append(wo_path.name)
        for j, c in enumerate(chunk):
            manifest["cases"].append({"case_id": f"case_{b + j:03d}",
                                      "workflow": c["workflow"]})
    (HARNESS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"contract: {contract} ({contract.stat().st_size} B)")
    for name in manifest["batches"]:
        p = HARNESS_DIR / name
        print(f"work order: {p} ({p.stat().st_size} B, "
          f"{len(json.loads(p.read_text()))} cases)")
    print(f"manifest: {HARNESS_DIR / 'manifest.json'}")
    return HARNESS_DIR / "manifest.json"


def score(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    cases = sample_cases(manifest["n_cases"], manifest["seed"])
    drafts: dict = {}
    for name in manifest["batches"]:
        dp = manifest_path.parent / name.replace("wo_", "draft_")
        d = json.loads(dp.read_text())
        for row in d["cases"]:
            drafts[row["case_id"]] = row["answers"]
    kls, agree, by_type, ok = [], [], {}, 0
    for i, c in enumerate(cases):
        cid = f"case_{i:03d}"
        if cid not in drafts:
            continue
        amap = build_alias_map(c["questions"])
        norm = answers_to_gold_keys(drafts[cid], c["gold"], alias=amap)
        if norm is None:
            continue
        ok += 1
        qtypes = {qid: q.get("type", "?") for qid, q in c["questions"].items()}
        kls.append(mean_kl(norm, c["gold"]))
        all_am = True
        for qid, g in c["gold"].items():
            slot = by_type.setdefault(qtypes.get(qid, "?"), {"n": 0, "agree": 0})
            am_ok = max(norm[qid], key=norm[qid].get) == max(g, key=g.get)
            slot["n"] += 1
            slot["agree"] += int(am_ok)
            all_am &= am_ok
        agree.append(all_am)
    rep = {"n_cases": manifest["n_cases"], "ok_cases": ok,
           "mean_soft_kl": float(np.mean(kls)) if kls else float("nan"),
           "argmax_agree_frac": float(np.mean(agree)) if agree else float("nan"),
           "by_question_type": {k: {"n": v["n"], "agree_frac": v["agree"] / v["n"]}
                                for k, v in sorted(by_type.items())},
           "labeler": manifest["labeler"], "harness": True}
    out = manifest_path.parent.parent.parent / "artifacts" / "teacher_audit" / "harness_glm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))
    print(f"saved {out}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["prepare", "score"], required=True)
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default=str(HARNESS_DIR / "manifest.json"))
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.cases, a.batch_size, a.seed)
    else:
        score(Path(a.manifest))


if __name__ == "__main__":
    main()
