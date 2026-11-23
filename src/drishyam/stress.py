"""G1-D3: deterministic jitter suite (n=300) + multi-question suite (n=100 states x 5)."""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

from drishyam.data_inspect import REPO
from drishyam.splits import SEEN


def jitter_state(state: str, rng: random.Random) -> str:
    words = [w.capitalize() if rng.random() < 0.30 else w for w in state.split(" ")]
    return " ".join(words) + "\n" * rng.randint(1, 3)  # case-only + whitespace: meaning preserved


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(seed: int):
    rng = random.Random(seed)
    test = {wf: load_dataset(REPO, wf, split="test") for wf in SEEN}
    jrows, mrows = [], []
    for wf in SEEN:
        for i, row in enumerate(test[wf]):
            qs = json.loads(row["questions"])
            jrows.append({
                "case_id": f"{wf}/test/{i}", "workflow": wf,
                "clean_state": row["state"], "jittered_state": jitter_state(row["state"], rng),
                "questions": qs,
            })
            if len(mrows) < 100:
                mrows.append({
                    "case_id": f"{wf}/test/{i}", "workflow": wf,
                    "qids": list(qs.keys()),
                })
    return jrows, mrows


def main() -> None:
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    jrows, mrows = build(1337)
    (out / "jitter_suite.jsonl").write_text("".join(json.dumps(r) + "\n" for r in jrows))
    (out / "multiq_suite.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mrows))
    h1 = sha((out / "jitter_suite.jsonl").read_bytes())
    j2, _ = build(1337)  # determinism proof: regenerate, byte-compare
    h2 = sha(("".join(json.dumps(r) + "\n" for r in j2)).encode())
    assert h1 == h2, "jitter suite not deterministic"
    print(f"jitter n={len(jrows)} multiq n={len(mrows)} deterministic=OK sha={h1[:12]}")


if __name__ == "__main__":
    main()
