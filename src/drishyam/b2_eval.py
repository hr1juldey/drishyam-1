"""G1-B2: zero-shot eval of the EXTERNAL reference model through ITS OWN serving engine.

ktcapraz/t5gemma2-decision-270m-v2 ships decisions/engine.py (CPU, threads) — we run it
as-shipped (layout=shared per its decision_config) on the same test/unseen/jitter/
invariance protocols. Fair comparison = their interface, our suites.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B2_SNAPSHOT = sorted((Path.home() / ".cache/huggingface/hub").glob(
    "models--ktcapraz--t5gemma2-decision-270m-v2/snapshots/*"))[0]


def load_engine(layout="shared"):
    sys.path.insert(0, str(B2_SNAPSHOT))
    import os

    os.environ["T5D_READOUT"] = "final"
    from decisions.engine import Engine, Timer

    return Engine(str(B2_SNAPSHOT), threads=6, max_state_tokens=8192,
                  layout=layout, local_attn=256), Timer


def raw_request(row_or_case, questions_json=None) -> dict:
    """Dataset row (raw) or CaseDataset case -> their request body (native format)."""
    qs = questions_json if questions_json is not None else json.loads(row_or_case["questions"])
    return {"state": row_or_case["state"], "questions": qs}


def align_probs(answer: dict, keys: list[str]) -> np.ndarray:
    if "probabilities" in answer:
        p = answer["probabilities"]
        return np.array([float(p[k]) for k in keys], dtype=float)
    if "noul" in answer:  # their tree mode: scalar p(true); keys order is [false, true]
        return np.array([1.0 - float(answer["noul"]), float(answer["noul"])])
    raise KeyError(f"no probabilities in answer: {list(answer)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=0, help="smoke cap per suite")
    ap.add_argument("--out", default="artifacts/b2_report.json")
    a = ap.parse_args()
    from datasets import load_dataset

    from drishyam.baselines import metrics_rows
    from drishyam.data_inspect import REPO
    from drishyam.splits import SEEN, UNSEEN

    engine, Timer = load_engine()

    def decide_rows(requests: list[dict]) -> list[dict]:
        from decisions.format import parse_request

        rows = []
        for req in requests:
            state, questions = parse_request(req)
            answers, _, _ = engine.decide(state, questions, Timer())
            for q in questions:
                probs = align_probs(answers[q.qid], q.keys)
                gold_json = req["_gold"][q.qid]["probabilities"]
                vec = np.array([float(gold_json[k]) for k in q.keys])
                rows.append({"case_id": req["_case_id"], "qid": q.qid, "prim": q.type,
                             "probs": probs, "gold": vec / vec.sum()})
        return rows

    def build_requests(wf: str, split: str, n: int, jitter_map=None) -> list[dict]:
        ds = load_dataset(REPO, wf, split=split)
        reqs = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            qs = json.loads(row["questions"])
            state = row["state"]
            if jitter_map is not None:
                state = jitter_map[(wf, f"{wf}/test/{i}")]
            reqs.append({"state": state, "questions": qs,
                         "_gold": json.loads(row["gold"]), "_case_id": f"{wf}/{split}/{i}"})
        return reqs

    cap = a.max_cases
    report = {}

    # ---- seen test ----
    reqs = []
    for wf in SEEN:
        reqs += build_requests(wf, "test", cap or 100)
    report["test"] = metrics_rows(decide_rows(reqs))

    # ---- unseen workflow (zero-shot) ----
    reqs = build_requests(UNSEEN, "test", cap or 100)
    report["unseen_test"] = metrics_rows(decide_rows(reqs))

    # ---- jitter flip ----
    jmap = {}
    for line in (ROOT / "artifacts/jitter_suite.jsonl").read_text().splitlines():
        r = json.loads(line)
        jmap[(r["workflow"], r["case_id"])] = r["jittered_state"]
    flips = {}
    for wf in SEEN:
        clean_reqs = build_requests(wf, "test", cap or 100)
        jitter_reqs = build_requests(wf, "test", cap or 100, jitter_map=jmap)
        rc = decide_rows(clean_reqs)
        rj = decide_rows(jitter_reqs)
        flips[wf] = float(np.mean([x["probs"].argmax() != y["probs"].argmax()
                                   for x, y in zip(rc, rj, strict=False)]))
    report["jitter_flip_rate"] = float(np.mean(list(flips.values())))
    report["jitter_per_workflow"] = flips

    # ---- multi-question invariance + speedup (Q1 alone vs Q1 in full batch, their engine) ----
    from decisions.format import parse_request

    deltas, speedups = [], []
    wf0 = SEEN[0]
    ds = load_dataset(REPO, wf0, split="test")
    n_inv = min(cap or 25, 25)  # their tree mode is slow — bounded sample, recorded as such
    for i in range(n_inv):
        row = ds[i]
        qs = json.loads(row["questions"])
        q1_body = {"state": row["state"], "questions": {k: qs[k] for k in list(qs)[:1]}}
        full_body = {"state": row["state"], "questions": qs}
        s1, q1q = parse_request(q1_body)
        sf, qfq = parse_request(full_body)
        t0 = time.perf_counter()
        a1, _, _ = engine.decide(s1, q1q, Timer())
        t1 = time.perf_counter()
        af, _, _ = engine.decide(sf, qfq, Timer())
        t2 = time.perf_counter()
        k = q1q[0].keys
        deltas.append(float(np.abs(align_probs(a1[q1q[0].qid], k)
                                   - align_probs(af[q1q[0].qid], k)).max()))
        speedups.append((t2 - t1) / max(t1 - t0, 1e-9))
    report["multiq"] = {
        "n_states": len(deltas),
        "max_delta_full_prec": float(np.max(deltas)),
        "max_delta_4dp": float(np.max(np.round(deltas, 4))),
        "bitwise_equal_frac": float(np.mean([d == 0.0 for d in deltas])),
        "batched_vs_single_q1_wall_ratio": float(np.median(speedups)),
        "note": "ratio>1 means batching is SLOWER for their engine at this sample",
    }

    (ROOT / a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
