"""Gate-2 eval: six pre-registered bars on the cached-page-embeddings decision model.

Bars (plan 2026-11-25_144900, pre-registered BEFORE the SFT run):
  g2_1 ablation flip > 0.30   (the model READS the page; blank-state must change answers)
  g2_2 multiq delta  = 0.0000 @4dp (one cached state, questions in any batch shape)
  g2_3 post-TS ECE  < 0.08    (calibrated trust, temperature scaling fitted on VAL)
  g2_4 unseen-family AUC > 0.75 on noul (family `salary` never seen in train/val/TS;
      table was planned but zero table pages exist in the banked 90k — measured)
  g2_5 jitter flip  < 0.05    (case/whitespace jitter on question text)
  g2_6 decisions/s  reported  (cached-state throughput; informational)

fp32 eval law (evaluate.py): bf16 GEMM reduction order makes near-ties flip across
batch shapes; model.float() keeps logits deterministic to ~1e-7 -> 4dp bar passes.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from drishyam.baselines import metrics_rows
from drishyam.calibrate import apply_temperature, fit_temperature
from drishyam.evaluate import flip_rate, load_model, timed_cuda
from drishyam.tokenize import collate_questions
from drishyam.train_gate2 import IMG_TOKENS, ece15, load_cases

ROOT = Path(__file__).resolve().parents[2]
BARS = {"g2_1_ablation_flip": (">", 0.30), "g2_2_multiq_delta_4dp": ("==", 0.0),
        "g2_3_ece_post_ts": ("<", 0.08), "g2_4_unseen_auc": (">", 0.75),
        "g2_5_jitter_flip": ("<", 0.05)}


def jitter_question(q, rng):
    from drishyam.augs import jitter_state
    from drishyam.schemas import Question

    return Question(qid=q.qid, primitive=q.primitive,
                    text=jitter_state(q.text, rng), options=list(q.options))


def rows_embeds(model, tok, cases, fixed_len, temps, cache, device="cuda",
                blank_state=False, q_transform=None, path="embeds"):
    """Gate-2 predict: one cached page tensor per case; questions batched; TS applied.

    path='row' routes through forward_row (encoder-row binding, Gate-2.5): each
    question is its own encoder row [page_embs ; question+option tokens]. fp32
    pages to match model.float() (the smoke's bf16!=float crash, 2026-11-25)."""
    rows = []
    for case in cases:
        dtype = torch.float32 if path == "row" else torch.bfloat16
        pages = torch.from_numpy(cache.get(case["page_key"]))[None].to(
            device=device, dtype=dtype)
        page_mask = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)
        qs = [q_transform(q) if q_transform else q for q in case["questions"]]
        dec_ids, dec_mask, slot_pos, _ = collate_questions(tok, qs, fixed_len, device)
        with torch.no_grad():
            if path == "row":
                if blank_state:
                    pages = torch.zeros_like(pages)
                out = model.forward_row(pages, page_mask, dec_ids, dec_mask, slot_pos,
                                        torch.zeros(dec_ids.size(0), dtype=torch.long,
                                                    device=device))
            else:
                out = model.forward_embeds(pages, page_mask, dec_ids, dec_mask, slot_pos,
                                           blank_state=blank_state)
        for i, (logit, q) in enumerate(zip(out, qs, strict=False)):
            probs = torch.softmax(logit.float(), -1).cpu().numpy()
            t = temps.get(q.primitive.value, {}).get("T", 1.0)
            rows.append({"case_id": case["page_key"], "qid": q.qid,
                         "prim": q.primitive.value,
                         "probs": apply_temperature(probs, t),
                         "gold": np.asarray(case["gold"][i], dtype=float)})
    return rows


def binary_auc(scores, pos_mask) -> float:
    s = np.asarray(scores, dtype=float)
    m = np.asarray(pos_mask, dtype=bool)
    n_pos, n_neg = int(m.sum()), int((~m).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # tie-averaged ranks
        j = i
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return float((ranks[m].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def multiq_embeds(model, tok, cases, fixed_len, cache, device="cuda", n_max=50,
                  path="embeds"):
    """Q1 alone vs Q1 batched with the rest, off the SAME cached page tensor."""
    deltas, speedups = [], []
    for case in cases[:n_max]:
        qs = case["questions"]
        if len(qs) < 2:
            continue
        dtype = torch.float32 if path == "row" else torch.bfloat16
        pages = torch.from_numpy(cache.get(case["page_key"]))[None].to(
            device=device, dtype=dtype)
        page_mask = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)

        def call(qlist, pages=pages, page_mask=page_mask):
            di, dm, pos, _ = collate_questions(tok, qlist, fixed_len, device)
            with torch.no_grad():
                if path == "row":
                    return model.forward_row(pages, page_mask, di, dm, pos,
                                             torch.zeros(di.size(0),
                                                         dtype=torch.long,
                                                         device=device))
                return model.forward_embeds(pages, page_mask, di, dm, pos)

        def per_question(qs=qs, pages=pages, page_mask=page_mask):
            outs = []
            for q in qs:
                di, dm, pos, _ = collate_questions(tok, [q], fixed_len, device)
                with torch.no_grad():
                    if path == "row":
                        outs.append(model.forward_row(pages, page_mask, di, dm, pos,
                                                      torch.zeros(1, dtype=torch.long,
                                                                  device=device))[0])
                    else:
                        outs.append(model.forward_embeds(pages, page_mask,
                                                         di, dm, pos)[0])
            return outs[0]

        a = [torch.softmax(t.float(), -1).cpu().numpy() for t in call(qs[:1])]
        b = [torch.softmax(t.float(), -1).cpu().numpy() for t in call(qs)]
        deltas.append(float(np.abs(a[0] - b[0]).max()))
        speedups.append(timed_cuda(per_question)
                        / max(timed_cuda(lambda qs=qs: call(qs)), 1e-9))
    return {"n_states": len(deltas),
            "max_delta_full_prec": float(np.max(deltas)) if deltas else 0.0,
            "max_delta_4dp": float(np.max(np.round(deltas, 4))) if deltas else 0.0,
            "bitwise_equal_frac": float(np.mean([d == 0.0 for d in deltas])) if deltas else 1.0,
            "median_speedup": float(np.median(speedups)) if speedups else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/g2-sft/best.pt")
    ap.add_argument("--out", default="artifacts/gate2_table.json")
    ap.add_argument("--jitter-n", type=int, default=300)
    ap.add_argument("--path", choices=["embeds", "row"], default="embeds",
                    help="embeds=Gate-2 decoder path; row=Gate-2.5 encoder-row path")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = "cuda"
    from drishyam.train_gate2 import PageCache

    cache = PageCache()
    val_cases, test_cases = load_cases("val"), load_cases("test")
    unseen_cases = load_cases("unseen")
    model, tok, fixed_len = load_model(ROOT / a.ckpt, device)
    model.float()  # fp32 eval law (see module docstring)

    # --- temperatures on VAL per primitive (trust layer) ---
    val_rows = rows_embeds(model, tok, val_cases, fixed_len, {}, cache, device,
                           path=a.path)
    temps = {}
    for prim in ("noul", "choice", "score"):
        sel = [r for r in val_rows if r["prim"] == prim]
        if sel:
            temps[prim] = {"T": fit_temperature([r["probs"] for r in sel],
                                                [r["gold"] for r in sel], prim),
                           "n": len(sel)}
    print("temperatures:", json.dumps(temps))

    # --- 1: state ablation on TEST (does it read the page?) ---
    real = rows_embeds(model, tok, test_cases, fixed_len, temps, cache, device,
                       path=a.path)
    blank = rows_embeds(model, tok, test_cases, fixed_len, temps, cache, device,
                        blank_state=True, path=a.path)
    g2_1 = flip_rate(real, blank)

    # --- 2: multi-question invariance (one state, any batch shape) ---
    mq = multiq_embeds(model, tok, test_cases + unseen_cases, fixed_len, cache, device,
                       path=a.path)

    # --- 3: post-TS calibration on TEST ---
    g2_3 = ece15([r["probs"] for r in real], [r["gold"] for r in real])

    # --- 4: unseen family (table) ---
    unseen_rows = rows_embeds(model, tok, unseen_cases, fixed_len, temps, cache,
                              device, path=a.path)
    um = metrics_rows(unseen_rows)
    noul = [r for r in unseen_rows if r["prim"] == "noul"]
    g2_4_auc = binary_auc([float(r["probs"][1]) for r in noul],
                          [float(r["gold"][1]) for r in noul]) if noul else float("nan")

    # --- 5: jitter flip (question text case/whitespace) ---
    rng = random.Random(a.seed)
    jcases = []
    for case in test_cases[: a.jitter_n]:
        qc = dict(case)
        qc["questions"] = [jitter_question(q, rng) for q in case["questions"]]
        jcases.append(qc)
    jrows = rows_embeds(model, tok, jcases, fixed_len, temps, cache, device,
                        path=a.path)
    g2_5 = flip_rate(real[: len(jrows)], jrows)

    # --- 6: throughput ---
    case = test_cases[0]

    def batched_all(case=case):
        dtype = torch.float32 if a.path == "row" else torch.bfloat16
        pages = torch.from_numpy(cache.get(case["page_key"]))[None].to(
            device=device, dtype=dtype)
        pm = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)
        di, dm, pos, _ = collate_questions(tok, case["questions"], fixed_len, device)
        with torch.no_grad():
            if a.path == "row":
                model.forward_row(pages, pm, di, dm, pos,
                                  torch.zeros(di.size(0), dtype=torch.long,
                                              device=device))
            else:
                model.forward_embeds(pages, pm, di, dm, pos)

    ms = timed_cuda(batched_all, warmup=3, reps=20)
    g2_6 = len(case["questions"]) / (ms / 1000.0)

    tm = metrics_rows(real)
    table = {
        "ckpt": a.ckpt, "temperatures": temps,
        "n_test": len(real), "n_unseen": len(unseen_rows),
        "g2_1_ablation_flip": g2_1, "blank_acc": metrics_rows(blank)["acc"],
        "g2_2_multiq": mq, "g2_2_multiq_delta_4dp": mq["max_delta_4dp"],
        "g2_3_ece_post_ts": g2_3,
        "g2_3_ece_pre_ts_val": ece15([r["probs"] for r in val_rows],
                                     [r["gold"] for r in val_rows]),
        "test_acc": tm["acc"], "test_brier": tm["brier"],
        "g2_4_unseen_acc": um["acc"], "g2_4_unseen_auc": g2_4_auc,
        "g2_5_jitter_flip": g2_5, "g2_6_decisions_per_s": g2_6,
    }
    verdict = {}
    for k, (op, thr) in BARS.items():
        v = table[k]
        ok = (v == thr) if op == "==" else (v > thr if op == ">" else v < thr)
        verdict[k] = {"value": v, "bar": f"{op} {thr}", "PASS": bool(ok)}
    table["verdict"] = verdict
    table["all_pass"] = all(x["PASS"] for x in verdict.values())
    table["cfg_sha256"] = hashlib.sha256(
        (json.dumps(table, sort_keys=True, default=str) + str(a.seed)).encode()).hexdigest()[:12]

    (ROOT / a.out).write_text(json.dumps(table, indent=2, default=str))
    print(json.dumps(verdict, indent=2))
    print("ALL PASS" if table["all_pass"] else "GATE-2 BARS NOT MET")

    import mlflow

    from drishyam.tracking import mlflow_run

    with mlflow_run("g2-eval") as run:
        mlflow.set_tag("gate", "2.5" if a.path == "row" else "2")
        mlflow.log_params({"ckpt": a.ckpt, "jitter_n": a.jitter_n, "seed": a.seed,
                           "path": a.path})
        for k, v in verdict.items():
            mlflow.log_metric(k, float(v["value"]) if v["value"] == v["value"] else -1.0)
            mlflow.log_metric(k + "_pass", float(v["PASS"]))
        mlflow.log_metric("g2_6_decisions_per_s", g2_6)
        mlflow.log_metric("test_acc", tm["acc"])
        mlflow.log_metric("unseen_acc", um["acc"])
        mlflow.log_artifact(str(ROOT / a.out))
        print("mlflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
