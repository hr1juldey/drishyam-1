"""G1-D1: verify counts vs card; discover question field names; snapshot official scorer."""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

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


def q_prim(q: dict) -> str:
    for key in ("primitive", "type", "kind"):  # field name discovered HERE, defensively
        if key in q and q[key] in PRIMS:
            return q[key]
    raise SystemExit(f"UNKNOWN question primitive keys={list(q)} sample={json.dumps(q)[:300]}")


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
        rows = load_dataset(REPO, wf, split="test" if wf == UNSEEN else "train")
        # unseen primitive counts come from TEST only — its train rows are never deserialized
        pc: Counter = Counter()
        for row in rows:
            for q in json.loads(row["questions"]):
                pc[q_prim(q)] += 1
        card["primitives"][wf] = dict(pc)
        test = load_dataset(REPO, wf, split="test")
        card["samples"][wf] = [
            {k: (row[k][:400] if isinstance(row[k], str) else row[k])
             for k in ("state", "questions", "gold")}
            for row in test.select(range(3))
        ]
    card["question_field_sample"] = json.loads(
        load_dataset(REPO, WORKFLOWS[0], split="test")[0]["questions"]
    )[0]
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
