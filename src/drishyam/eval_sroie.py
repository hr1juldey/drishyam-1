"""SROIE receipt verification — first in-distribution PUBLIC image benchmark entry.

Protocol (v1, defined in docs/BENCHMARK_LEDGER.md and recorded verbatim in the
output table): per receipt x per field (total, date, company) x {TRUE, FALSE}.
TRUE quotes the gold value verbatim in the question; FALSE is a deterministic
perturbation drawn from another receipt's same-field gold value (seed 2026,
never equal to this receipt's gold) -> exactly 50/50 balanced No/Yes rows on
the noul head.

Pool: the OFFICIAL SROIE 400-image test pool, identifiable via the HF
extension jsdnrs/ICDAR2019-SROIE: its test split carries 361 receipts with
Task-3 KIE gold (347 original + 14 recovered; the remaining 39 images of the
400 have no public Task-3 gold anywhere). The pinned snapshot
Voxel51/scanned_receipts (712 receipts) was schema-verified and used as the
contamination check: it contains ZERO official-test receipt IDs (it is
trainval-side), so evaluating on it would NOT have been the official test
pool — per the pre-declared split rule ("official 400-test pool where
identifiable") the test split of the extension is the eval set. No SROIE data
was used in any drishyam training run; these receipts are unseen.

Embed path (precompute_embed.py convention, proven on real images by
eval_imajev.py): JPEG bytes -> float32 [0,1] -> bilinear stretch 896x896 ->
frozen vision tower -> projector -> [256,640]; each receipt encoded ONCE, all
6 verification rows of a receipt forwarded batched through forward_row, fp32
(model.float()). Raw softmax (no temperature scaling): gate2 TS temps are
fitted on our synthetic val distribution and documented not to transfer OOD.
"""
import argparse
import hashlib
import io
import json
import os
import random
import subprocess
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # cache id google/t5gemma-2-270m-270m only

import numpy as np
import torch
from PIL import Image

from drishyam.tokenize import SLOT, question_text

ROOT = Path(__file__).resolve().parents[2]
SROIE_DIR = ROOT / "runs" / "hub_ref" / "sroie"
TEST_PARQUET = SROIE_DIR / "jsdnrs_ref" / "data" / "test-00000-of-00001.parquet"
VOXEL_SAMPLES = SROIE_DIR / "hf_snapshot" / "samples.json"
NOUL_OPTIONS = ["No", "Yes"]  # our noul eval convention (gold TRUE -> "Yes")
FIELDS = ("total", "date", "company")  # canonical field order (protocol)
SEED = 2026
EMBED_SIZE = (896, 896)
PAGE_TOKENS = 256
PAGE_DIM = 640
EMBED_BATCH = 16
ROW_BATCH_RECEIPTS = 36  # 36 receipts * 6 rows = 216 decoder rows per forward
PROTOCOL = (
    "SROIE receipt verification v1 (protocol defined here; recorded for comparability, "
    "precedent: bev-decider's self-defined 5k protocol). Pool: the OFFICIAL SROIE "
    "400-image test pool as identifiable = the 361 test receipts with public Task-3 KIE "
    "gold (347 original + 14 recovered), taken from HF extension jsdnrs/ICDAR2019-SROIE "
    "test split; the remaining 39/400 images have no public Task-3 gold anywhere. The "
    "pinned source Voxel51/scanned_receipts (712 receipts, schema-verified: per-receipt "
    "company/date/address/total + image + OCR boxes) is trainval-side and contains ZERO "
    "official-test IDs (verified by receipt-ID join, recorded) — used only as the "
    "contamination check and sample inspection. Rows: per receipt x field (total, date, "
    "company) x {TRUE, FALSE}: TRUE quotes the gold value verbatim (outer whitespace "
    "stripped only) in the question; FALSE is a deterministic perturbation drawn from "
    "another receipt's same-field gold value (rng seeded per (receipt_id, field) under "
    "seed 2026, candidate pool excludes any value equal to this receipt's gold), exactly "
    "one TRUE + one FALSE -> 50/50 balanced. Question template: 'Does the {field} on "
    "this receipt read \"{value}\"? Answer No or Yes.' Options No/Yes (noul head). Eval: "
    "each receipt image encoded once (JPEG -> fp32 [0,1] -> bilinear stretch 896x896 -> "
    "frozen tower -> projector, precompute_embed.py convention), all 6 rows per receipt "
    "forwarded batched, fp32, raw softmax (no TS: gate2 temperatures are fitted on our "
    "synthetic val and documented not to transfer OOD). Metrics: accuracy (argmax), AUC "
    "of P(Yes) vs gold Yes/No, 15-bin ECE on max-prob confidence (house ece15 "
    "convention)."
)


# --- pool loading (CPU, no torch) -------------------------------------------


def load_official_test(parquet_path: Path | None = None) -> list[dict]:
    """Official test pool -> [{key, gold{...}, image_bytes}] sorted by key.

    Source: jsdnrs/ICDAR2019-SROIE test split (the official 400-test pool
    receipts that carry Task-3 KIE gold: 361). Requires runs-local parquet.
    """
    import pandas as pd

    path = Path(parquet_path) if parquet_path else TEST_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — snapshot the test split first "
            "(hf_hub_download jsdnrs/ICDAR2019-SROIE data/test-00000-of-00001.parquet)"
        )
    df = pd.read_parquet(path)
    out = []
    for _, r in df.iterrows():
        gold_fields = ("total", "date", "company", "address")
        out.append({
            "key": str(r["key"]),
            "gold": {f: str(r["entities"].get(f, "")).strip() for f in gold_fields},
            "image_bytes": bytes(r["image"]["bytes"]),
        })
    out.sort(key=lambda x: x["key"])
    return out


def voxel51_overlap_check(receipts: list[dict], samples_path: Path | None = None) -> dict:
    """Contamination check against the pinned Voxel51 snapshot (schema-verified)."""
    import os

    path = Path(samples_path) if samples_path else VOXEL_SAMPLES
    vox = json.loads(path.read_text())["samples"]
    voxel_ids = {os.path.basename(s["filepath"]).rsplit(".", 1)[0] for s in vox}
    ours = {r["key"] for r in receipts}
    return {
        "voxel51_receipts": len(voxel_ids),
        "official_test_receipts": len(ours),
        "overlap": len(voxel_ids & ours),
    }


# --- verification-row construction (pure, CPU-testable) ----------------------


def question_template(field: str, value: str) -> str:
    return f'Does the {field} on this receipt read "{value}"? Answer No or Yes.'


def false_claim_pool(receipts: list[dict], field: str, gold: str) -> list[str]:
    """Sorted unique same-field values from OTHER receipts, minus this gold."""
    seen = set()
    g = gold.strip()
    for r in receipts:
        v = r["gold"].get(field, "").strip()
        if v and v != g:
            seen.add(v)
    return sorted(seen)


def draw_false_claim(receipts: list[dict], field: str, receipt: dict, seed: int = SEED) -> str:
    """Deterministic wrong value: another receipt's same-field gold, never ours."""
    pool = false_claim_pool(receipts, field, receipt["gold"][field])
    if not pool:
        raise ValueError(f"{receipt['key']}/{field}: no false-claim candidate pool")
    rng = random.Random(f"{seed}:{receipt['key']}:{field}")
    return rng.choice(pool)


def build_verification_rows(receipts: list[dict], seed: int = SEED) -> list[dict]:
    """Per receipt x field x {TRUE, FALSE} rows, 50/50 balanced by construction.

    Row order: receipt key sorted, fields in FIELDS order, TRUE then FALSE —
    so any prefix of the row list stays answer-balanced within each receipt.
    """
    rows: list[dict] = []
    for r in sorted(receipts, key=lambda x: x["key"]):
        for field in FIELDS:
            gold = r["gold"][field].strip()
            if not gold:
                raise ValueError(f"{r['key']}/{field}: empty gold, cannot build claims")
            wrong = draw_false_claim(receipts, field, r, seed)
            rows.append({
                "row_id": f"{r['key']}:{field}:true",
                "receipt_key": r["key"],
                "field": field,
                "claim_type": "true",
                "claimed": gold,
                "gold": gold,
                "answer": "Yes",
            })
            rows.append({
                "row_id": f"{r['key']}:{field}:false",
                "receipt_key": r["key"],
                "field": field,
                "claim_type": "false",
                "claimed": wrong,
                "gold": gold,
                "answer": "No",
            })
    return rows


def row_case(row: dict):
    """Verification row -> pydantic Question (schema-gated)."""
    from drishyam.schemas import Primitive, Question

    return Question(
        qid=row["row_id"],
        primitive=Primitive.NOUL,
        text=question_template(row["field"], row["claimed"]),
        options=list(NOUL_OPTIONS),
    )


# --- embed + forward (GPU) ---------------------------------------------------


def image_bytes_tensor(raw: bytes) -> "torch.Tensor":
    """JPEG bytes -> [3,896,896] float32 in [0,1] (precompute convention)."""
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return torch.nn.functional.interpolate(
        t, size=EMBED_SIZE, mode="bilinear", align_corners=False, antialias=True
    )[0]


def vision_modules(bb):
    """(tower, projector) under possible peft wrapping — BFS, fixed chains crash
    on T5Gemma2+peft (eval_imajev.py lesson)."""
    queue, seen = [bb], set()
    while queue:
        node = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        if hasattr(node, "vision_tower") and hasattr(node, "multi_modal_projector"):
            return node.vision_tower, node.multi_modal_projector
        for hop in ("model", "encoder", "base_model"):
            child = getattr(node, hop, None)
            if child is not None and hasattr(child, "parameters"):
                queue.append(child)
    raise AttributeError(f"cannot locate vision modules under {type(bb).__name__}")


def embed_receipts(model, receipts: list[dict], device: str) -> dict[str, "torch.Tensor"]:
    """Receipt key -> [256,640] fp32 page embedding; each image encoded once."""
    tower, proj = vision_modules(model.bb)
    out: dict[str, torch.Tensor] = {}
    for s in range(0, len(receipts), EMBED_BATCH):
        chunk = receipts[s : s + EMBED_BATCH]
        px = torch.stack([image_bytes_tensor(r["image_bytes"]) for r in chunk]).to(device)
        with torch.inference_mode():
            emb = proj(tower(px).last_hidden_state)
        for r, e in zip(chunk, emb.float().cpu(), strict=False):
            out[r["key"]] = e
    return out


def predict_rows(model, tok, rows: list[dict], page_embs: dict, fixed_len: int,
                 device: str, receipts_per_batch: int = ROW_BATCH_RECEIPTS) -> list[np.ndarray]:
    """All verification rows, batched by receipt: each forward carries
    receipts_per_batch page embeddings once and their 6 rows each."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    slot_id = tok.convert_tokens_to_ids(SLOT)
    probs_all: list[np.ndarray] = [None] * len(rows)  # type: ignore[list-item]
    for s in range(0, len(rows), receipts_per_batch * 6):
        chunk = rows[s : s + receipts_per_batch * 6]
        keys = sorted({r["receipt_key"] for r in chunk})
        key_idx = {k: i for i, k in enumerate(keys)}
        pages = torch.stack([page_embs[k] for k in keys]).to(device=device, dtype=torch.float32)
        page_mask = torch.ones(pages.size(0), PAGE_TOKENS, dtype=torch.long, device=device)
        q_texts = [question_text(row_case(r)) for r in chunk]
        raws = [tok(t, add_special_tokens=True)["input_ids"] for t in q_texts]
        dec_ids = torch.tensor([r + [pad] * (fixed_len - len(r)) for r in raws], device=device)
        dec_mask = torch.tensor(
            [[1] * len(r) + [0] * (fixed_len - len(r)) for r in raws], device=device)
        slot_pos = [(torch.tensor(r, device=device) == slot_id).nonzero().flatten() for r in raws]
        for i, r in enumerate(chunk):
            if len(slot_pos[i]) != len(NOUL_OPTIONS):
                raise ValueError(f"{r['row_id']}: slot count != {len(NOUL_OPTIONS)} options")
        row_case_idx = torch.tensor([key_idx[r["receipt_key"]] for r in chunk],
                                    dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model.forward_row(pages, page_mask, dec_ids, dec_mask, slot_pos, row_case_idx)
        for i, (logit, r) in enumerate(zip(out, chunk, strict=False)):
            probs = torch.softmax(logit.float(), -1).cpu().numpy()
            if len(probs) != len(NOUL_OPTIONS):
                raise ValueError(f"{r['row_id']}: got {len(probs)} slot logits")
            probs_all[s + i] = probs
    return probs_all


# --- metrics (house conventions: ece15 from eval_gate2, tie-averaged AUC) -----


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


def ece15(probs: list, golds: list) -> float:
    conf = np.array([p.max() for p in probs])
    ok = np.array([float(p.argmax() == g.argmax()) for p, g in zip(probs, golds, strict=False)])
    bins = np.linspace(0.0, 1.0, 16)
    n = len(conf)
    e = 0.0
    for b in range(15):
        lo, hi = bins[b], bins[b + 1]
        m = (conf >= lo) & (conf < hi if b < 14 else conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def metric_block(rows: list[dict], probs: list[np.ndarray]) -> dict:
    golds = [np.asarray([1.0, 0.0] if r["answer"] == "No" else [0.0, 1.0]) for r in rows]
    correct = [bool(p.argmax() == g.argmax()) for p, g in zip(probs, golds, strict=False)]
    p_yes = [float(p[1]) for p in probs]
    yes_mask = [g[1] == 1.0 for g in golds]
    return {
        "n": len(rows),
        "n_true": sum(r["claim_type"] == "true" for r in rows),
        "n_false": sum(r["claim_type"] == "false" for r in rows),
        "acc": float(np.mean(correct)),
        "acc_true": float(np.mean([c for c, r in zip(correct, rows, strict=False)
                                   if r["claim_type"] == "true"])),
        "acc_false": float(np.mean([c for c, r in zip(correct, rows, strict=False)
                                    if r["claim_type"] == "false"])),
        "auc_p_yes": binary_auc(p_yes, yes_mask),
        "ece15": ece15(probs, golds),
    }


# --- plumbing ----------------------------------------------------------------


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_rows(rows: list[dict]) -> str:
    """Digest of the exact verification-row set (reproducibility record)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()[:16]


def main() -> None:
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="runs/g2-row-sft/best.pt")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="cap receipts (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/sroie_table.json")
    ap.add_argument("--dump-probs", default="",
                    help="write per-row P(Yes)+gold to this JSON so TS re-scoring "
                         "reuses the forward pass (step-2 calibration path)")
    a = ap.parse_args()

    receipts = load_official_test()
    if a.limit:
        receipts = receipts[: a.limit]
    overlap = voxel51_overlap_check(receipts)
    assert overlap["overlap"] == 0, f"test-pool leakage into pinned snapshot: {overlap}"
    rows = build_verification_rows(receipts, a.seed)
    n_true = sum(r["claim_type"] == "true" for r in rows)
    print(f"receipts={len(receipts)} rows={len(rows)} true={n_true} false={len(rows) - n_true} "
          f"voxel51_overlap={overlap['overlap']}", flush=True)

    from drishyam.eval_jevbench import load_model

    model, tok, fixed_len, _state_len = load_model(ROOT / a.ckpt, a.device)
    model.float()  # fp32 eval law
    t0 = time.time()
    page_embs = embed_receipts(model, receipts, a.device)
    t_embed = time.time() - t0
    print(f"embedded {len(page_embs)} receipts in {t_embed:.1f}s "
          f"(fp32 tower, {EMBED_SIZE[0]}px [0,1] convention)", flush=True)

    t0 = time.time()
    probs_all = predict_rows(model, tok, rows, page_embs, fixed_len, a.device)
    if a.dump_probs:
        (ROOT / a.dump_probs).write_text(json.dumps({
            "row_ids": [r["row_id"] for r in rows],
            "p_yes": [float(p[1]) for p in probs_all],
            "gold_argmax": [1 if r["answer"] == "Yes" else 0 for r in rows],
        }))
        print(f"probs dumped: {a.dump_probs}", flush=True)
    t_fwd = time.time() - t0
    print(f"forwarded {len(probs_all)} rows in {t_fwd:.1f}s", flush=True)

    per_field = {}
    for field in FIELDS:
        idx = [i for i, r in enumerate(rows) if r["field"] == field]
        per_field[field] = metric_block([rows[i] for i in idx], [probs_all[i] for i in idx])
    overall = metric_block(rows, probs_all)

    table = {
        "run_kind": "SROIE receipt verification v1 — first in-distribution PUBLIC "
                    "image benchmark entry (protocol self-defined on public gold, "
                    "recorded for comparability)",
        "benchmark": "sroie-verification",
        "ckpt": a.ckpt,
        "git_sha": git_sha(),
        "seed": a.seed,
        "protocol": PROTOCOL,
        "question_template": question_template("{field}", "{value}"),
        "fields": list(FIELDS),
        "pool": {
            "official_test_pool": 400,
            "scored_receipts": len(receipts),
            "pool_note": "361/400 official test receipts have public Task-3 KIE gold "
                         "(347 original + 14 recovered, HF jsdnrs/ICDAR2019-SROIE test "
                         "split); remaining 39 have no public gold anywhere",
            "voxel51_contamination_check": overlap,
            "rows_sha256_16": sha256_rows(rows),
        },
        "n_rows": len(rows),
        "balance": {"true": overall["n_true"], "false": overall["n_false"]},
        "embed_convention": "JPEG bytes -> fp32 [0,1] -> bilinear stretch 896x896 -> "
                            "frozen tower -> projector (precompute_embed.py convention); "
                            "each receipt encoded once, its 6 rows batched",
        "no_temperature_scaling": "raw softmax; gate2 TS temps are synthetic-val fits, "
                                  "documented not to transfer OOD",
        "timing_s": {"embed": round(t_embed, 1), "forward": round(t_fwd, 1)},
        "per_field": per_field,
        "overall": overall,
    }
    (ROOT / a.out).write_text(json.dumps(table, indent=2))
    print(json.dumps({k: table[k] for k in
                      ("benchmark", "ckpt", "n_rows", "balance", "per_field", "overall")},
                     indent=2))

    import mlflow

    from drishyam.tracking import mlflow_run

    with mlflow_run("sroie-verification-v1") as run:
        mlflow.set_tag("gate", "2.5")
        mlflow.set_tag("benchmark", "sroie-verification")
        mlflow.set_tag("kind", "public-benchmark")
        mlflow.log_params({"ckpt": a.ckpt, "seed": a.seed, "n_receipts": len(receipts),
                           "n_rows": len(rows), "fixed_len": fixed_len})
        mlflow.log_metrics({
            "overall_acc": overall["acc"],
            "overall_auc_p_yes": overall["auc_p_yes"],
            "overall_ece15": overall["ece15"],
            **{f"{f}_acc": per_field[f]["acc"] for f in FIELDS},
            **{f"{f}_auc": per_field[f]["auc_p_yes"] for f in FIELDS},
            **{f"{f}_ece": per_field[f]["ece15"] for f in FIELDS},
        })
        mlflow.log_artifact(str(ROOT / a.out))
        print("mlflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
