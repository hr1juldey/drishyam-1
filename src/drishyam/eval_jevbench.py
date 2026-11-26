"""JevBench public-231 evaluator — the external text-state benchmark (TEXT track).

Protocol (runs/hub_ref/jfbench_research.md §1, §3, §4):
  The runnable public set is 231 tasks: datasets/public/easy.jsonl (48),
  original.jsonl (72, tier "standard"), hard.jsonl (111). One plain-text state +
  one typed question per task; the judge tier and all held-out tiers are sealed.
  Score = plain accuracy over the tasks ("public-231 accuracy"), the number open
  models report (Jev 86.6, Kev-0.6B 66.7, bev-decider 65.8, Laya 58.4).
Task format (verified against the downloaded files, 2026-11-26):
  {id, family, group, split: "public", state, question: {type, instructions,
  criteria}, labels, expected}
    choice: labels = option keys, expected = one key -> options = labels
    noul:   labels = ["no","yes"], expected in {"yes","no"} -> options ["No","Yes"]
    score:  labels = index strings, expected = int, criteria = ordered levels
            -> options = criteria levels, gold = expected
Local copies (MIT-licensed) live at runs/hub_ref/jevbench_public/; point --data-dir
elsewhere for fixtures. Contamination note (research §5): bev-decision's
counterfactual_15k batch was screened against these 231 states; training on the
public split must be disclosed per JevBench policy.

fp32 eval law (evaluate.py/eval_gate2.py): model.float() keeps logits
deterministic across batch shapes. State limit defaults to 2,048 tokens
(bev-decider's protocol); hard-tier states run up to ~6k tokens — raise
--state-max-tokens (ktcapraz used 8,192) to score them untruncated.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from drishyam.calibrate import apply_temperature
from drishyam.model import DecoupledDecision, attach_lora, build_backbone
from drishyam.tokenize import add_slot_token, collate_questions
from drishyam.train_gate2 import MODEL_ID
from drishyam.train_text_row import state_batch_ids

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_FILES = ("easy.jsonl", "original.jsonl", "hard.jsonl")
FILE_TIER = {"easy.jsonl": "easy", "original.jsonl": "standard", "hard.jsonl": "hard"}
FULL_COUNTS = {"easy": 48, "standard": 72, "hard": 111}  # research §3; informational
NOUL_OPTIONS = ["No", "Yes"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def task_to_case(r: dict, tier: str) -> dict:
    """One public JSONL task -> trainer/eval case (one typed question)."""
    q = r["question"]
    qtype = q["type"]
    labels, expected = r["labels"], r["expected"]
    if qtype == "choice":
        options = [str(x) for x in labels]
        if expected not in labels:
            raise ValueError(f"{r['id']}: expected {expected!r} not in labels")
        gold_idx = labels.index(expected)
    elif qtype == "noul":
        if tuple(sorted(str(x).lower() for x in labels)) != ("no", "yes"):
            raise ValueError(f"{r['id']}: noul labels must be no/yes, got {labels!r}")
        if str(expected).lower() not in ("yes", "no"):
            raise ValueError(f"{r['id']}: noul expected must be yes/no, got {expected!r}")
        options = list(NOUL_OPTIONS)
        gold_idx = 1 if str(expected).lower() == "yes" else 0
    elif qtype == "score":
        criteria = q.get("criteria")
        options = (
            [str(x) for x in criteria] if isinstance(criteria, list) else [str(x) for x in labels]
        )
        gold_idx = (
            expected
            if isinstance(expected, int) and not isinstance(expected, bool)
            else labels.index(str(expected))
        )
        if not 0 <= gold_idx < len(options):
            raise ValueError(f"{r['id']}: score gold {gold_idx} out of range")
    else:
        raise ValueError(f"{r['id']}: unknown question type {qtype!r}")
    gold = [1.0 if i == gold_idx else 0.0 for i in range(len(options))]
    from drishyam.schemas import Primitive, Question

    question = Question(
        qid=r["id"], primitive=Primitive(qtype), text=q["instructions"], options=options
    )
    return {
        "case_id": r["id"],
        "tier": tier,
        "family": r.get("family", ""),
        "state": _state_text(r["state"]),
        "questions": [question],
        "gold": [gold],
        "expected_raw": expected,
        "labels_raw": labels,
    }


def _state_text(state: object) -> str:
    """Public-set states are str OR dict-of-documents (35 hard tasks) — canonical
    text form is sorted-keys JSON: lossless + deterministic for the tokenizer."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def load_public(data_dir: str | Path) -> list[dict]:
    """All public JSONLs -> task cases with their tier attached."""
    data_dir = Path(data_dir)
    cases = []
    for fname in PUBLIC_FILES:
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — download the public JSONLs to {data_dir} (research §1)"
            )
        tier = FILE_TIER[fname]
        for line in path.read_text().splitlines():
            if line.strip():
                cases.append(task_to_case(json.loads(line), tier))
    counts = {t: sum(1 for c in cases if c["tier"] == t) for t in FILE_TIER.values()}
    if counts != FULL_COUNTS:
        print(
            f"NOTE: tier counts {counts} != full public set {FULL_COUNTS} (fixture subset?)",
            flush=True,
        )
    return cases


def predict_rows(model, tok, cases, state_len, fixed_len, temps=None, device="cuda"):
    """Text-state forward_row predictions; one question per task, fp32 probs."""
    temps = temps or {}
    rows = []
    for case in cases:
        s_ids, s_mask = state_batch_ids(tok, [case["state"]], state_len, device)
        state_embs = model.bb.get_input_embeddings()(s_ids)
        qs = case["questions"]
        dec_ids, dec_mask, slot_pos, _ = collate_questions(tok, qs, fixed_len, device)
        rc = torch.zeros(dec_ids.size(0), dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model.forward_row(state_embs, s_mask, dec_ids, dec_mask, slot_pos, rc)
        for logit, q, g in zip(out, qs, case["gold"], strict=False):
            probs = torch.softmax(logit.float(), -1).cpu().numpy()
            t = temps.get(q.primitive.value, {}).get("T", 1.0)
            probs = apply_temperature(probs, t)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "tier": case["tier"],
                    "family": case["family"],
                    "qid": q.qid,
                    "prim": q.primitive.value,
                    "probs": probs,
                    "gold": np.asarray(g, dtype=float),
                    "options": q.options,
                    "expected_raw": case["expected_raw"],
                    "correct": bool(int(probs.argmax()) == int(np.argmax(g))),
                }
            )
    return rows


def accuracy(rows) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(r["probs"].argmax() == r["gold"].argmax()) for r in rows]))


def summarize(rows) -> dict:
    """Public-231 plain accuracy, overall + per tier + per question type."""
    out = {"n_tasks": len(rows), "acc": accuracy(rows)}
    for tier in ("easy", "standard", "hard"):
        sel = [r for r in rows if r["tier"] == tier]
        out[f"acc_{tier}"] = accuracy(sel)
        out[f"n_{tier}"] = len(sel)
    for prim in ("choice", "noul", "score"):
        sel = [r for r in rows if r["prim"] == prim]
        out[f"acc_{prim}"] = accuracy(sel)
        out[f"n_{prim}"] = len(sel)
    out["brier"] = (
        float(np.mean([float(((r["probs"] - r["gold"]) ** 2).sum()) for r in rows]))
        if rows
        else float("nan")
    )
    return out


def load_model(ckpt_path: Path, device: str = "cuda"):
    """Checkpoint -> (model fp32-eval-ready, tokenizer, fixed_len, state_len)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bb = build_backbone(MODEL_ID)
    slot_id = add_slot_token(tok, bb)
    bb = attach_lora(bb)
    model = DecoupledDecision(bb, slot_id).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    fixed_len = int(ck["fixed_len"])
    state_len = int(ck.get("state_len", 1024))
    return model, tok, fixed_len, state_len


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/text-row-sft/best.pt")
    ap.add_argument("--data-dir", default="runs/hub_ref/jevbench_public")
    ap.add_argument(
        "--state-max-tokens",
        type=int,
        default=2048,
        help="bev-decider's eval protocol; raise to 8192 for hard tier",
    )
    ap.add_argument("--temps", default="", help="optional per-primitive T json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap tasks (smoke)")
    ap.add_argument("--out", default="artifacts/jevbench_public231_table.json")
    a = ap.parse_args()

    device = a.device
    temps = json.loads((ROOT / a.temps).read_text()) if a.temps else {}
    cases = load_public(ROOT / a.data_dir)
    if a.limit:
        cases = cases[: a.limit]
    tier_counts = {t: sum(1 for c in cases if c["tier"] == t) for t in FILE_TIER.values()}
    print(f"tasks={len(cases)} by_tier={tier_counts}", flush=True)

    model, tok, fixed_len, ck_state_len = load_model(ROOT / a.ckpt, device)
    model.float()  # fp32 eval law (see module docstring)
    state_len = a.state_max_tokens or ck_state_len
    rows = predict_rows(model, tok, cases, state_len, fixed_len, temps, device)
    table = {
        "ckpt": a.ckpt,
        "data_dir": a.data_dir,
        "state_max_tokens": state_len,
        "git_sha": git_sha(),
        "per_task": [
            {
                "id": r["case_id"],
                "tier": r["tier"],
                "type": r["prim"],
                "pred": r["options"][int(r["probs"].argmax())],
                "expected": r["expected_raw"],
                "correct": bool(r["probs"].argmax() == r["gold"].argmax()),
            }
            for r in rows
        ],
        **summarize(rows),
    }
    table["cfg_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in table.items() if k != "per_task"}, sort_keys=True, default=str
        ).encode()
    ).hexdigest()[:12]
    (ROOT / a.out).write_text(json.dumps(table, indent=2))
    print(json.dumps({k: v for k, v in table.items() if k != "per_task"}, indent=2))

    import mlflow

    from drishyam.tracking import mlflow_run

    with mlflow_run("jevbench-public231") as run:
        mlflow.set_tag("gate", "3")
        mlflow.set_tag("track", "text-row")
        mlflow.log_params({"ckpt": a.ckpt, "state_max_tokens": state_len, "n_tasks": len(cases)})
        for k, v in table.items():
            if k.startswith("acc") or k in ("brier", "n_tasks"):
                mlflow.log_metric(k, float(v))
        mlflow.log_artifact(str(ROOT / a.out))
        print("mlflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
