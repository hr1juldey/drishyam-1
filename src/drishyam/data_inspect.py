"""G1-D1: verify counts vs card; discover question field names; snapshot official scorer."""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from drishyam.schemas import Primitive, Question

REPO = "LocalLLaMA/typed-decisions"
SEEN = ["invoice_processing", "security_incidents", "customer_service"]
UNSEEN = "agent_trace_observability"  # held-out zero-shot workflow (user-confirmed scope)
WORKFLOWS = [*SEEN, UNSEEN]
PRIMS = {"noul", "choice", "score"}
ART = Path("artifacts")
SCORER_URL = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/raw/main/bench/score_with_jobs.py"


def parquet_sha(cfg: str, split: str) -> str:
    p = hf_hub_download(REPO, f"{cfg}/{split}-00000-of-00001.parquet", repo_type="dataset")
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def q_prim(meta: dict) -> str:
    """questions is a DICT qid -> {type, instructions, criteria} (discovered 2026-11-23)."""
    t = meta.get("type")
    if t in PRIMS:
        return t
    raise SystemExit(f"UNKNOWN question type keys={list(meta)} sample={json.dumps(meta)[:300]}")


def to_questions(row: dict) -> list:
    """Normalise a dataset row into schema.Question list in dict insertion order."""
    out = []
    for qid, meta in json.loads(row["questions"]).items():
        prim = q_prim(meta)
        crit = meta.get("criteria")
        if prim == "score":
            options = [f"{i}: {c}" for i, c in enumerate(crit)]
        elif prim == "choice":
            options = [f"{name}: {desc}" for name, desc in crit.items()]
        else:  # noul — criteria may be {'false': desc, 'true': desc} or None
            options = [
                f"false: {crit['false']}" if isinstance(crit, dict) else "false",
                f"true: {crit['true']}" if isinstance(crit, dict) else "true",
            ]
        out.append(Question(
            qid=qid, primitive=Primitive(prim), text=meta["instructions"], options=options,
        ))
    return out


def to_gold(row: dict) -> dict[str, list[float]]:
    """gold[qid] -> distribution vector aligned to to_questions option order (sums to 1)."""
    gold_json = json.loads(row["gold"])
    questions = json.loads(row["questions"])
    out: dict[str, list[float]] = {}
    for qid, meta in questions.items():
        probs = gold_json[qid]["probabilities"]
        if meta["type"] == "score":
            keys = [str(i) for i in range(len(meta["criteria"]))]
        elif meta["type"] == "choice":
            keys = list(meta["criteria"].keys())
        else:
            keys = ["false", "true"]
        vec = [float(probs[k]) for k in keys]
        s = sum(vec)
        if abs(s - 1.0) > 1e-2:
            raise SystemExit(f"gold distribution for {qid} sums to {s}: {vec}")
        out[qid] = [v / s for v in vec]
    return out


def main() -> None:
    ART.mkdir(exist_ok=True)
    card: dict = {
        "counts": {},
        "primitives": {},
        "samples": {},
        "parquet_sha256": {},
        "question_field_sample": None,
    }
    for wf in WORKFLOWS:
        for split in ("train", "test"):
            ds = load_dataset(REPO, wf, split=split)
            card["counts"][f"{wf}/{split}"] = len(ds)
            card["parquet_sha256"][f"{wf}/{split}"] = parquet_sha(wf, split)
        if wf == UNSEEN:
            rows = load_dataset(REPO, wf, split="test")  # test only; train rows never deserialized
        else:
            rows = load_dataset(REPO, wf, split="train")
        pc: Counter = Counter()
        for row in rows:
            for meta in json.loads(row["questions"]).values():
                pc[q_prim(meta)] += 1
        card["primitives"][wf] = dict(pc)
        test = load_dataset(REPO, wf, split="test")
        card["samples"][wf] = [
            {k: (row[k][:400] if isinstance(row[k], str) else row[k])
             for k in ("state", "questions", "gold")}
            for row in test.select(range(3))
        ]
    card["question_field_sample"] = next(iter(
        json.loads(load_dataset(REPO, WORKFLOWS[0], split="test")[0]["questions"]).items()
    ))[1]
    print(json.dumps(card["counts"], indent=1))
    print(json.dumps(card["primitives"], indent=1))
    (ART / "data_card.json").write_text(json.dumps(card, indent=2))
    sp = subprocess.run(
        ["curl", "-sL", SCORER_URL, "-o", str(ART / "bench_scorer_snapshot.py")], check=False
    )
    (ART / "bench_scorer_snapshot.sha256").write_text(
        hashlib.sha256((ART / "bench_scorer_snapshot.py").read_bytes()).hexdigest()
    )
    print(f"scorer snapshot exit={sp.returncode}")


if __name__ == "__main__":
    main()
