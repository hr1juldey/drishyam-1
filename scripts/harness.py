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

CONTRACT_V2_EXTRA = """
## Calibration rules (match the dataset's graders, evidence-first)
1. The criteria descriptions are AUTHORITATIVE: interpret the state THROUGH
   them. For score/rank questions read EVERY level's description and pick the
   level whose description matches the OBSERVED facts — never the worst case
   imaginable. If evidence is partial for a level, drop one rank.
2. When the facts reconcile and policy was followed, lean toward the benign
   outcome (approve / close_benign / answer_directly) — the worked examples
   show this house style. Escalate (hold / manual_review / investigate /
   contain) only when a SPECIFIC fact violates policy or cannot be verified.
   Do not add caution beyond what the evidence supports.
3. Distribution shape: when evidence points one way, lead with 0.55-0.8 and
   put the remainder on the single most plausible alternative (a small tail on
   other keys is fine). Reserve >0.9 for conclusive evidence (IDs verbatim in
   prior lists, policy explicitly broken). Use near-uniform only when the case
   is truly undecidable, not when you merely lack one detail.
4. Yes/no questions: answer what the evidence SHOWS, not what would be safer.
   Genuinely close calls stay 0.5-0.65; never 0.9+ against the evidence.
5. Multi-option choices: weigh ALL named options before distributing — the
   examples show approve/hold/manual_review/reject and
   close_benign/investigate/contain/escalate usage patterns.
"""


def contract_text(version: str, examples: list[dict]) -> str:
    parts = [CONTRACT_PREAMBLE]
    for i, e in enumerate(examples, 1):
        parts.append(f"EXAMPLE {i} STATE:\n{e['state']}\n")
        parts.append(f"EXAMPLE {i} QUESTIONS:\n{json.dumps(e['questions'], ensure_ascii=False)}\n")
        ans_json = json.dumps({"answers": e["answers"]}, ensure_ascii=False)
        parts.append(f"EXAMPLE {i} ANSWERS (style anchor):\n{ans_json}\n")
    if version == "v2":
        parts.append(CONTRACT_V2_EXTRA)
    return "\n".join(parts)


def few_shot_examples_v2(seed: int = 42) -> list[dict]:
    """Worked examples covering ALL THREE workflows (v1 covered two)."""
    from datasets import load_dataset

    from drishyam.data_inspect import REPO
    from drishyam.splits import SEEN

    picks = [(SEEN[0], 1), (SEEN[1], 2), (SEEN[2], 3), (SEEN[1], 5)]
    out = []
    for wf, i in picks:
        r = load_dataset(REPO, wf, split="test")[i]
        gold = json.loads(r["gold"])
        answers = {q: g["probabilities"] for q, g in gold.items()}
        questions = json.loads(r["questions"])
        questions = {
            qid: {k: v for k, v in q.items() if k != "criteria"}
            | {"options": q.get("criteria", {})}
            for qid, q in questions.items()
        }
        out.append({"state": r["state"], "questions": questions, "answers": answers})
    return out


def prepare(n_cases: int, batch_size: int, seed: int, out_dir: Path,
            contract: str = "v1") -> Path:
    cases = sample_cases(n_cases, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = (few_shot_examples_v2(seed) if contract == "v2" else few_shot_examples(seed))
    contract_path = out_dir / f"CONTRACT_{contract.upper()}.md"
    contract_path.write_text(contract_text(contract, ex))

    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed,
                "n_cases": len(cases), "batch_size": batch_size,
                "contract": contract, "labeler": "hermes-subagent (inherited provider)",
                "batches": [], "cases": []}
    for b in range(0, len(cases), batch_size):
        chunk = cases[b:b + batch_size]
        wo = [{"case_id": f"case_{b + j:03d}", "workflow": c["workflow"],
               "state": c["state"], "questions": c["questions"]}
              for j, c in enumerate(chunk)]
        wo_path = out_dir / f"wo_{b // batch_size:03d}.json"
        wo_path.write_text(json.dumps(wo, ensure_ascii=False, indent=1))
        manifest["batches"].append(wo_path.name)
        for j, c in enumerate(chunk):
            manifest["cases"].append({"case_id": f"case_{b + j:03d}",
                                      "workflow": c["workflow"]})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"contract: {contract_path} ({contract_path.stat().st_size} B)")
    print(f"{len(manifest['batches'])} work orders, {len(cases)} cases -> {out_dir}")
    return out_dir / "manifest.json"


def score(manifest_path: Path, draft_prefix: str = "draft_",
          limit: int | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    cases = sample_cases(manifest["n_cases"], manifest["seed"])
    n_batches = (len(manifest["batches"]) if limit is None
                 else -(-limit // manifest["batch_size"]))
    drafts: dict = {}
    for name in manifest["batches"][:n_batches]:
        dp = manifest_path.parent / (draft_prefix + name.replace("wo_", ""))
        if not dp.exists():
            continue
        d = json.loads(dp.read_text())
        for row in d["cases"]:
            drafts[row["case_id"]] = row["answers"]
    kls, agree, by_type, by_wf, ok = [], [], {}, {}, 0
    wf_of = {c["case_id"]: c["workflow"] for c in manifest["cases"]}
    for i, c in enumerate(cases):
        cid = f"case_{i:03d}"
        if limit is not None and i >= limit:
            break
        if cid not in drafts:
            continue
        amap = build_alias_map(c["questions"])
        norm = answers_to_gold_keys(drafts[cid], c["gold"], alias=amap)
        if norm is None:
            continue
        ok += 1
        wf_slot = by_wf.setdefault(wf_of.get(cid, "?"), {"n": 0, "agree": 0})
        qtypes = {qid: q.get("type", "?") for qid, q in c["questions"].items()}
        kls.append(mean_kl(norm, c["gold"]))
        all_am = True
        for qid, g in c["gold"].items():
            slot = by_type.setdefault(qtypes.get(qid, "?"), {"n": 0, "agree": 0})
            am_ok = max(norm[qid], key=norm[qid].get) == max(g, key=g.get)
            slot["n"] += 1
            slot["agree"] += int(am_ok)
            wf_slot["agree"] += int(am_ok)
            all_am &= am_ok
        wf_slot["n"] += 1
        agree.append(all_am)
    def frac(slot): return slot["agree"] / slot["n"] if slot["n"] else float("nan")
    rep = {"n_cases": manifest["n_cases"], "scored_limit": limit, "ok_cases": ok,
           "draft_prefix": draft_prefix, "contract": manifest.get("contract"),
           "mean_soft_kl": float(np.mean(kls)) if kls else float("nan"),
           "argmax_agree_frac": float(np.mean(agree)) if agree else float("nan"),
           "by_question_type": {k: {"n": v["n"], "agree_frac": frac(v)}
                                for k, v in sorted(by_type.items())},
           "by_workflow": {k: {"n": v["n"], "agree_frac": frac(v)}
                           for k, v in sorted(by_wf.items())},
           "labeler": manifest["labeler"], "harness": True}
    tag = f"{manifest_path.parent.name}_{draft_prefix.strip('_')}"
    if limit is not None:
        tag += f"_first{limit}"
    out = ROOT / "artifacts" / "teacher_audit" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))
    print(f"saved {out}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Hermes-harness teacher interface")
    ap.add_argument("--mode", choices=["prepare", "score"], required=True)
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contract", choices=["v1", "v2"], default="v1")
    ap.add_argument("--out-dir", default=str(ROOT / "drafts" / "harness"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--draft-prefix", default="draft_")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.cases, a.batch_size, a.seed, Path(a.out_dir), contract=a.contract)
    else:
        mpath = Path(a.manifest or (Path(a.out_dir) / "manifest.json"))
        score(mpath, draft_prefix=a.draft_prefix, limit=a.limit)


if __name__ == "__main__":
    main()
