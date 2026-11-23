"""G1-V1: evaluation suites — test/unseen/jitter/multiq-invariance/state-ablation + gate table."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from drishyam.baselines import metrics_rows
from drishyam.calibrate import apply_temperature
from drishyam.dataset import CaseDataset
from drishyam.model import DecoupledDecision, attach_lora, build_backbone
from drishyam.tokenize import add_slot_token, collate_questions

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/t5gemma-2-270m-270m"


def load_model(ckpt_path: Path, device: str = "cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bb = build_backbone(MODEL_ID)
    slot_id = add_slot_token(tok, bb)
    bb = attach_lora(bb)
    model = DecoupledDecision(bb, slot_id).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, int(ck["fixed_len"])


def predict_rows(model, tok, cases, fixed_len, temperatures: dict,
                 device="cuda", state_override=None, blank_state=False):
    """Each case: ONE cached H_enc, its questions as one padded fixed-shape decoder batch."""
    rows = []
    for case in cases:
        state = case["state"] if state_override is None else state_override
        s_ids, s_mask = state_batch(tok, [state], device)
        dec_ids, dec_mask, slot_pos, _ = collate_questions(
            tok, case["questions"], fixed_len, device)
        with torch.no_grad():
            out = model(s_ids, s_mask, dec_ids, dec_mask, slot_pos,
                        blank_state=blank_state)
        for i, (logit, q) in enumerate(zip(out, case["questions"], strict=False)):
            probs = torch.softmax(logit.float(), -1).cpu().numpy()
            t = temperatures.get(q.primitive.value, {}).get("T", 1.0)
            probs = apply_temperature(probs, t)
            rows.append({
                "case_id": case["case_id"], "qid": q.qid, "prim": q.primitive.value,
                "probs": probs, "gold": np.asarray(case["gold"][i], dtype=float),
            })
    return rows


def state_batch(tok, states, device):
    enc = [tok(s, add_special_tokens=True, truncation=True, max_length=1024)["input_ids"]
           for s in states]
    longest = max(len(e) for e in enc)
    ids = torch.tensor([e + [tok.pad_token_id] * (longest - len(e)) for e in enc], device=device)
    mask = torch.tensor([[1] * len(e) + [0] * (longest - len(e)) for e in enc], device=device)
    return ids, mask


def flip_rate(rows_a, rows_b) -> float:
    assert len(rows_a) == len(rows_b)
    return float(np.mean([a["probs"].argmax() != b["probs"].argmax()
                          for a, b in zip(rows_a, rows_b, strict=False)]))


def max_prob_delta(rows_a, rows_b) -> float:
    """Ragged-safe max |Δp| over matching option sets (multiq invariance metric)."""
    worst = 0.0
    for a, b in zip(rows_a, rows_b, strict=False):
        assert a["probs"].shape == b["probs"].shape
        worst = max(worst, float(np.abs(a["probs"] - b["probs"]).max()))
    return worst


def timed_cuda(fn, warmup=3, reps=20) -> float:
    """Median wall ms over reps with proper sync (GPU-box timing law)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def multiq_suite(model, tok, cases, fixed_len, device="cuda"):
    """For n=100 states: Q1 alone vs Q1 batched with the rest, off the SAME cached H_enc.
    Reports 4dp-rounded max |Δp| (DECISIONS §4) and cached-vs-reencode speedup."""
    deltas, speedups = [], []
    for case in cases:
        qs = case["questions"]
        if len(qs) < 2:
            continue
        s_ids, s_mask = state_batch(tok, [case["state"]], device)

        def q1_alone(qs=qs, s_ids=s_ids, s_mask=s_mask):
            di, dm, pos, _ = collate_questions(tok, qs[:1], fixed_len, device)
            with torch.no_grad():
                return model(s_ids, s_mask, di, dm, pos)

        def q1_batched(qs=qs, s_ids=s_ids, s_mask=s_mask):
            di, dm, pos, _ = collate_questions(tok, qs, fixed_len, device)
            with torch.no_grad():
                return model(s_ids, s_mask, di, dm, pos)[0]

        def reencode_all(qs=qs, s_ids=s_ids, s_mask=s_mask):
            outs = []
            for q in qs:
                di, dm, pos, _ = collate_questions(tok, [q], fixed_len, device)
                with torch.no_grad():
                    outs.append(model(s_ids, s_mask, di, dm, pos)[0])
            return outs[0]

        a = [torch.softmax(t.float(), -1).cpu().numpy() for t in q1_alone()]
        b = [torch.softmax(t.float(), -1).cpu().numpy() for t in q1_batched()]
        deltas.append(float(np.abs(a[0] - b[0]).max()))
        speedups.append(timed_cuda(reencode_all) / max(timed_cuda(q1_batched), 1e-9))
    return {
        "n_states": len(deltas),
        "max_delta_full_prec": float(np.max(deltas)),
        "max_delta_4dp": float(np.max(np.round(deltas, 4))),
        "bitwise_equal_frac": float(np.mean([d == 0.0 for d in deltas])),
        "median_speedup": float(np.median(speedups)),
    }


def ablation_suite(model, tok, cases, fixed_len, temperatures, device="cuda"):
    real = predict_rows(model, tok, cases, fixed_len, temperatures, device)
    blank = predict_rows(model, tok, cases, fixed_len, temperatures, device,
                         blank_state=True)
    return {
        "state_ablation_flip_rate": flip_rate(real, blank),
        "real_acc": metrics_rows(real)["acc"],
        "blank_acc": metrics_rows(blank)["acc"],
        "acc_drop": metrics_rows(real)["acc"] - metrics_rows(blank)["acc"],
        "real_brier": metrics_rows(real)["brier"],
        "blank_brier": metrics_rows(blank)["brier"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/g1-t1/best.pt")
    ap.add_argument("--out", default="artifacts/gate_table.json")
    a = ap.parse_args()
    from drishyam.splits import SEEN, UNSEEN

    temperatures = json.loads((ROOT / "artifacts/temperatures.json").read_text()) \
        if (ROOT / "artifacts/temperatures.json").exists() else {}
    model, tok, fixed_len = load_model(ROOT / a.ckpt)

    test, unseen = [], []
    for wf in SEEN:
        ds = CaseDataset(wf, "test", list(range(100)))
        test += [ds[i] for i in range(len(ds))]
    ds = CaseDataset(UNSEEN, "test", list(range(100)))
    unseen += [ds[i] for i in range(len(ds))]
    mrows = [json.loads(line) for line in
             (ROOT / "artifacts/multiq_suite.jsonl").read_text().splitlines()]
    # multiq rows carry (case_id, workflow, qids) — rebuild cases from the dataset
    mq_cases = []
    for r in mrows[:100]:
        idx = int(r["case_id"].rsplit("/", 1)[1])
        ds = CaseDataset(r["workflow"], "test", [idx])
        mq_cases.append(ds[0])

    table = {
        "ckpt": a.ckpt,
        "temperatures": temperatures,
        "in_domain_test": metrics_rows(predict_rows(model, tok, test, fixed_len, temperatures)),
        "unseen_workflow": metrics_rows(predict_rows(model, tok, unseen, fixed_len, temperatures)),
        "multiq": multiq_suite(model, tok, mq_cases, fixed_len),
        "state_ablation": ablation_suite(model, tok, unseen, fixed_len, temperatures),
    }

    # jitter flip: clean vs jittered on the same 300 test cases
    jmap = {}
    for line in (ROOT / "artifacts/jitter_suite.jsonl").read_text().splitlines():
        r = json.loads(line)
        jmap[(r["workflow"], r["case_id"])] = r["jittered_state"]
    flips = []
    for wf in SEEN:
        ds = CaseDataset(wf, "test", list(range(100)))
        cs = [ds[i] for i in range(len(ds))]
        rc = predict_rows(model, tok, cs, fixed_len, temperatures)
        rj = predict_rows(model, tok, [dict(c, state=jmap[(wf, c["case_id"])]) for c in cs],
                          fixed_len, temperatures)
        flips.append(flip_rate(rc, rj))
    table["jitter_flip_rate"] = float(np.mean(flips))

    (ROOT / a.out).write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
