"""ImajevBench v2.0-lite STOCK evaluator — photo-domain OOD gap measurement.

Purpose (user-authorized GPU run, 2026-11-26): measure the OOD domain gap of the
Gate-2.5 row champion (runs/g2-row-sft/best.pt, trained ONLY on synthetic
document pages) on ImajevBench's locally scoreable splits (dev 173 +
calibration 81; test is public-input/private-label and stays untouched).
Zero trained-family overlap (deleg_56fcabed contact-sheet verification): the
13 scene kinds (menus/signs/shelves/nutrition/parcels/bins/timetable/
thermostat) share no family with our receipts/statements/cards/bills training
distribution. OOD is EXPECTED — this run quantifies it, it does not tune it.

Mapping (their schema -> our typed rows, schema verified deleg_56fcabed):
  boolean -> noul  (options No/Yes, our fixed eval convention; gold True -> Yes)
  choice  -> choice (options = their option values in their order)
  ordinal -> score  (options = level descriptions, values 1|2|3 kept aligned)
  gold null = Unknown reference -> abstention target (their UNKNOWN key)

Abstention rule (fitted, not trained): predict Unknown iff max prob < tau.
tau is fitted on the calibration split to maximize THEIR official typed
correctness (their shipped scoring.py: gold null correct iff abstained,
otherwise correct iff answered value type-matches gold); dev is reported at
that tau with the full sweep curve. tau=0 row = raw argmax (no abstention).

Embed path: visual/joint items encode their JPEG assets through our frozen
vision tower + projector exactly as precompute_embed.py did for training
pages: float32 [0,1] -> bilinear stretch to 896x896 -> tower -> projector ->
[256,640]. (Not the Gemma3 processor's [-1,1] normalization — the gate2
training cache was produced tower-on-[0,1]; we stay in that convention.)
Multi-image items use the mean of their page embeddings (keeps the trained
256-token shape). Text-track items have no page: blank-page (zeros), the
model's own ablation state — documented prior-only behavior for a vision-row
checkpoint. request.state text is prepended as a "State: ... | " segment and
truncated to the checkpoint's fixed_len; question+options always fit (the
question text itself is shortened word-wise as a fallback, counted in the
table). fp32 eval law: model.float(), fp32 tower, one forward per item.

Score: their dependency-free scoring module (runs/hub_ref/imajev/runs/
imajev-2b-released/source/src__imajev_bench__scoring.py) is loaded verbatim
and produces typed correctness, group-bootstrap CI, Brier/NLL/fixed-10-bin
ECE and selective metrics. Probabilities expose our softmax over the typed
options plus __unknown__ carrying 1.0 when abstaining, 0.0 otherwise (sums
to 1.0, passes their validator).
"""

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # cache id google/t5gemma-2-270m-270m only

import numpy as np
import torch
from PIL import Image

from drishyam.tokenize import SLOT, question_text

ROOT = Path(__file__).resolve().parents[2]
IMAJEV_DIR = ROOT / "runs" / "hub_ref" / "imajev"
RECORDS = IMAJEV_DIR / "records" / "records-public.jsonl"
SCORING_SRC = (
    IMAJEV_DIR / "runs" / "imajev-2b-released" / "source" / "src__imajev_bench__scoring.py"
)
NOUL_OPTIONS = ["No", "Yes"]  # our noul eval convention (gold True -> "Yes")
BOOLEAN_KEYS = ["false", "true"]  # their candidate keys for booleans, in option order
EMBED_SIZE = (896, 896)  # precompute_embed.py convention (bilinear stretch)
PAGE_TOKENS = 256
PAGE_DIM = 640
OOD_LABEL = (
    "OOD gap measurement — ImajevBench is photo-domain (menus/signs/shelves/"
    "parcels); our model is document-domain; zero trained-family overlap "
    "(verified deleg_56fcabed)"
)


def load_scoring(path: Path | None = None):
    """Import THEIR shipped scoring module verbatim (dependency-free)."""
    src = Path(path) if path else SCORING_SRC
    if not src.exists():
        raise FileNotFoundError(f"official scorer missing: {src}")
    spec = importlib.util.spec_from_file_location("imajev_scoring", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def state_text(record: dict) -> str:
    """request.state -> canonical sorted-keys JSON (lossless, deterministic)."""
    st = (record.get("request") or {}).get("state")
    if st is None:
        return ""
    return json.dumps(st, sort_keys=True, ensure_ascii=False)


def scene_kind(record: dict) -> str:
    for src in (record.get("provenance") or {}).get("image_sources") or []:
        if src.get("scene_kind"):
            return str(src["scene_kind"])
    return "text_only"


def record_to_case(record: dict) -> dict:
    """Their single-field record -> our typed case (pure, no torch).

    gold None is kept as gold_idx None (Unknown reference, abstention target);
    option_keys fixes the option order so the probability vector maps back
    onto THEIR candidate keys losslessly.
    """
    field = (record.get("request") or {}).get("fields", [{}])[0]
    kind = field.get("type")
    gold = record.get("gold")
    if kind == "boolean":
        prim = "noul"
        options = list(NOUL_OPTIONS)
        option_keys = list(BOOLEAN_KEYS)  # option i -> their candidate key
        gold_idx = None if gold is None else {False: 0, True: 1}[bool(gold)]
    elif kind == "choice":
        prim = "choice"
        options = [str(o["value"]) for o in field["options"]]
        if len(set(options)) != len(options):
            raise ValueError(f"{record.get('id')}: duplicate choice option values")
        option_keys = list(options)  # values ARE their candidate keys
        gold_idx = None if gold is None else options.index(str(gold))
    elif kind == "ordinal":
        prim = "score"
        options = [str(lv.get("description") or lv["value"]) for lv in field["levels"]]
        values = [int(lv["value"]) for lv in field["levels"]]
        if len(set(values)) != len(values):
            raise ValueError(f"{record.get('id')}: duplicate ordinal level values")
        option_keys = [str(v) for v in values]
        gold_idx = None if gold is None else values.index(int(gold))
    else:
        raise ValueError(f"{record.get('id')}: unsupported field type {kind!r}")
    from drishyam.schemas import Primitive, Question

    question = Question(
        qid=str(record.get("id")),
        primitive=Primitive(prim),
        text=str(field["question"]),
        options=options,
    )
    return {
        "case_id": str(record.get("id")),
        "record": record,
        "track": str(record.get("track")),
        "family": str(record.get("family")),
        "scene_kind": scene_kind(record),
        "prim": prim,
        "question": question,
        "option_keys": option_keys,
        "gold_idx": gold_idx,
        "state": state_text(record),
        "asset_paths": [str(IMAJEV_DIR / im["path"]) for im in (record.get("images") or [])],
    }


def load_records(splits: tuple[str, ...]) -> list[dict]:
    if not RECORDS.exists():
        raise FileNotFoundError(f"{RECORDS} missing — dataset snapshot not in repo")
    rows = [json.loads(line) for line in RECORDS.read_text().splitlines() if line.strip()]
    sel = [r for r in rows if r.get("split") in splits]
    if not sel:
        raise ValueError(f"no records for splits {splits}")
    return sel


def build_row_text(case: dict, tok, fixed_len: int) -> tuple[str, dict]:
    """(text, flags): question+options suffix always fits fixed_len; the
    'State: ... | ' segment gets the remaining token budget (longest state
    prefix via binary search). If the suffix alone overflows, the question
    text is shortened word-wise (flagged, never silent)."""
    q = case["question"]
    q_text, question_truncated = q.text, False
    suffix = question_text(q)

    def tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=True)["input_ids"])

    while tokens(suffix) > fixed_len and q_text.split():
        q_text = " ".join(q_text.split()[:-1])
        question_truncated = True
        suffix = question_text(q.model_copy(update={"text": q_text}))
    if tokens(suffix) > fixed_len:
        raise ValueError(f"{case['case_id']}: options alone exceed fixed_len")
    st = case["state"]
    if st and tokens(f"State: {st} | {suffix}") <= fixed_len:
        return f"State: {st} | {suffix}", {"question": False, "state": False}
    if st:
        lo, hi = 0, len(st)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tokens(f"State: {st[:mid]} | {suffix}") <= fixed_len:
                lo = mid
            else:
                hi = mid - 1
        if lo > 0:
            return f"State: {st[:lo]} | {suffix}", {"question": question_truncated, "state": True}
    return suffix, {"question": question_truncated, "state": bool(st)}


def jpeg_tensor(path: str) -> "torch.Tensor":
    """JPEG file -> [3,896,896] float32 in [0,1] (precompute convention:
    /255, bilinear stretch, antialiased)."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return torch.nn.functional.interpolate(
        t, size=EMBED_SIZE, mode="bilinear", align_corners=False, antialias=True
    )[0]


def vision_modules(bb):
    """(vision_tower, multi_modal_projector) from the (possibly peft-wrapped)
    backbone. Raw path is .model.encoder.*; peft inserts one extra .model hop.
    BFS over the module tree (depth-capped) rather than a fixed attribute chain."""
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
    raise AttributeError(
        f"cannot locate vision_tower/multi_modal_projector under {type(bb).__name__}"
    )


def embed_assets(model, paths: list[str], device: str, batch: int = 8) -> dict:
    """Unique asset paths -> [256,640] fp32 page embeddings via OUR frozen
    tower+projector (the exact modules the gate2 cache was built from)."""
    tower, proj = vision_modules(model.bb)
    uniq = sorted(set(paths))
    out = {}
    for s in range(0, len(uniq), batch):
        px = torch.stack([jpeg_tensor(p) for p in uniq[s : s + batch]]).to(device)
        with torch.inference_mode():
            emb = proj(tower(px).last_hidden_state)
        for p, e in zip(uniq[s : s + batch], emb.float().cpu(), strict=False):
            out[p] = e
    return out


def page_for_case(case: dict, emb_cache: dict) -> "torch.Tensor":
    """visual/joint -> mean of asset page embeds (trained 256-token shape);
    text -> blank page (the model's own ablation state; prior-only)."""
    if case["asset_paths"]:
        return torch.stack([emb_cache[p] for p in case["asset_paths"]]).mean(0)
    return torch.zeros(PAGE_TOKENS, PAGE_DIM)


def predict_case(model, tok, case, page, fixed_len, device) -> np.ndarray:
    """One forward per item (fp32) -> softmax probs over our options."""
    row_text, _flags = build_row_text(case, tok, fixed_len)
    raw = tok(row_text, add_special_tokens=True)["input_ids"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dec_ids = torch.tensor([raw + [pad] * (fixed_len - len(raw))], device=device)
    dec_mask = torch.tensor([[1] * len(raw) + [0] * (fixed_len - len(raw))], device=device)
    slot_pos = [
        (torch.tensor(raw, device=device) == tok.convert_tokens_to_ids(SLOT)).nonzero().flatten()
    ]
    n_opts = len(case["question"].options)
    if len(slot_pos[0]) != n_opts:
        raise ValueError(
            f"{case['case_id']}: slot count {len(slot_pos[0])} != options {n_opts}"
        )
    pages = page[None].to(device=device, dtype=torch.float32)
    page_mask = torch.ones(1, pages.size(1), dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.forward_row(
            pages, page_mask, dec_ids, dec_mask, slot_pos,
            torch.zeros(1, dtype=torch.long, device=device),
        )
    return torch.softmax(out[0].float(), -1).cpu().numpy()


def to_prediction(record: dict, case: dict, probs: np.ndarray, tau: float) -> dict:
    """Our softmax probs -> THEIR prediction object at threshold tau
    (abstain iff max prob < tau). Probabilities expose their exact candidate
    keys incl __unknown__ and sum to 1.0."""
    keys = [*case["option_keys"], "__unknown__"]
    mx = float(max(probs))
    vec = dict.fromkeys(keys, 0.0)
    if mx < tau:
        vec["__unknown__"] = 1.0
        return {"status": "abstained", "value": None, "probabilities": vec}
    idx = int(np.argmax(probs))
    if case["prim"] == "noul":
        value = bool(idx == 1)
    elif case["prim"] == "choice":
        value = case["option_keys"][idx]
    else:
        value = int(case["option_keys"][idx])
    for i, k in enumerate(case["option_keys"]):
        vec[k] = float(probs[i])
    return {"status": "answered", "value": value, "probabilities": vec}


def typed_correct(records: list[dict], preds: dict, scorer) -> list[bool]:
    """THEIR _correct (typed, abstention-aware) per record; missing predictions
    count as incorrect, exactly like their predictions.get() semantics."""
    return [bool(scorer._correct(r, preds.get(r["id"]))) for r in records]


def tau_candidates(probs_list) -> list[float]:
    mx = sorted({round(float(p), 8) for p in probs_list})
    return [0.0] + [v + 1e-9 for v in mx] + [1.0]


def sweep(records, cases, probs_all, scorer) -> list[dict]:
    """Full tau curve on one split: typed correctness + coverage per tau."""
    maxes = [float(max(p)) for p in probs_all]
    out = []
    for tau in tau_candidates(maxes):
        preds = {
            c["case_id"]: to_prediction(r, c, p, tau)
            for r, c, p in zip(records, cases, probs_all, strict=False)
        }
        correct = typed_correct(records, preds, scorer)
        answered = sum(pr["status"] == "answered" for pr in preds.values())
        out.append(
            {
                "tau": tau,
                "acc": float(np.mean(correct)),
                "coverage": answered / len(records),
                "n": len(records),
            }
        )
    return out


def best_tau(curve: list[dict]) -> tuple[float, float]:
    """Max typed correctness; ties -> lower tau (more answers, higher coverage)."""
    best = max(curve, key=lambda e: (e["acc"], -e["tau"]))
    return best["tau"], best["acc"]


def predictions_at(records, cases, probs_all, tau: float) -> dict:
    return {
        c["case_id"]: to_prediction(r, c, p, tau)
        for r, c, p in zip(records, cases, probs_all, strict=False)
    }


def official_score(records, preds, scorer, bootstrap: int = 400) -> dict:
    """THEIR score() verbatim (capability/probability/selective/groups/CI)."""
    return scorer.score(records, preds, bootstrap_samples=bootstrap, seed=0)


def aggregate(records, cases, probs_all, tau: float, scorer) -> dict:
    """Typed-correctness blocks (overall/per-track/per-scene_kind) + coverage."""
    preds = predictions_at(records, cases, probs_all, tau)
    correct = typed_correct(records, preds, scorer)

    def block(values: list[bool]) -> dict:
        return {"n": len(values), "acc": float(np.mean(values)) if values else None}

    out = {"overall": block(correct), "n_answered": sum(
        p["status"] == "answered" for p in preds.values())}
    for key in ("track", "scene_kind", "prim"):
        grouped: dict[str, list[bool]] = {}
        for i, c in enumerate(cases):
            grouped.setdefault(c[key], []).append(correct[i])
        out[key] = {k: block(v) for k, v in sorted(grouped.items())}
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="runs/g2-row-sft/best.pt")
    ap.add_argument("--splits", default="dev,calibration")
    ap.add_argument("--tau", type=float, default=None, help="pin tau (skip fitting)")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="cap items per split (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/imajev_table.json")
    a = ap.parse_args()

    device = a.device
    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    scorer = load_scoring()
    records = load_records(splits)
    if a.limit:
        records = [r for s in splits for r in
                   [x for x in records if x["split"] == s][: a.limit]]
    cases = [record_to_case(r) for r in records]
    n_assets = len({p for c in cases for p in c["asset_paths"]})
    print(f"records={len(records)} by_split="
          f"{ {s: sum(1 for r in records if r['split'] == s) for s in splits} } "
          f"assets={n_assets}", flush=True)

    from drishyam.eval_jevbench import load_model

    model, tok, fixed_len, _state_len = load_model(ROOT / a.ckpt, device)
    model.float()  # fp32 eval law
    emb_cache = embed_assets(model, [p for c in cases for p in c["asset_paths"]], device)
    print(f"embedded {len(emb_cache)} assets (fp32 tower, {EMBED_SIZE[0]}px "
          f"[0,1] stretch convention)", flush=True)

    probs_all, flags_all = [], []
    for c in cases:
        page = page_for_case(c, emb_cache)
        probs_all.append(predict_case(model, tok, c, page, fixed_len, device))
        flags_all.append(build_row_text(c, tok, fixed_len)[1])
    print(f"forwards done: {len(probs_all)} items", flush=True)

    by_split = {s: [i for i, r in enumerate(records) if r["split"] == s] for s in splits}

    # --- fit tau on calibration (or first split) to THEIR typed metric ---
    fit_split = "calibration" if "calibration" in by_split else splits[0]
    curves = {s: sweep([records[i] for i in by_split[s]], [cases[i] for i in by_split[s]],
                       [probs_all[i] for i in by_split[s]], scorer) for s in splits}
    if a.tau is not None:
        tau_best = float(a.tau)
        calib_acc = next(e["acc"] for e in curves[fit_split] if abs(e["tau"] - a.tau) < 1e-9)
    else:
        tau_best, calib_acc = best_tau(curves[fit_split])

    table = {
        "run_kind": OOD_LABEL,
        "ckpt": a.ckpt,
        "git_sha": git_sha(),
        "fixed_len": fixed_len,
        "splits_scored": list(splits),
        "test_split": "excluded (public-input/private-label; untouched)",
        "n_records": len(records),
        "embed_convention": "JPEG->fp32 [0,1]->bilinear stretch 896x896->tower->projector "
                            "(precompute_embed.py convention); multi-image=mean; "
                            "text track=blank page (prior-only)",
        "abstention_rule": f"abstain (Unknown) iff max prob < tau; tau fitted on "
                           f"{fit_split} against their typed correctness",
        "tau_best": tau_best,
        f"{fit_split}_acc_at_tau_best": calib_acc,
        "state_truncated_items": sum(f["state"] for f in flags_all),
        "question_truncated_items": sum(f["question"] for f in flags_all),
        "tau_sweep": {s: curves[s] for s in splits},
        "per_primitive": {},
    }
    for s in splits:
        idx = by_split[s]
        rec_s = [records[i] for i in idx]
        case_s = [cases[i] for i in idx]
        prob_s = [probs_all[i] for i in idx]
        off = official_score(rec_s, predictions_at(rec_s, case_s, prob_s, tau_best),
                             scorer, a.bootstrap)
        off0 = official_score(rec_s, predictions_at(rec_s, case_s, prob_s, 0.0),
                              scorer, a.bootstrap)
        pq, pq0, sel = off["probability_quality"], off0["probability_quality"], off["selective"]
        table[s] = {
            "typed": aggregate(rec_s, case_s, prob_s, tau_best, scorer),
            "typed_no_abstention": aggregate(rec_s, case_s, prob_s, 0.0, scorer),
            "official_capability": {
                "overall_score": off["capability"]["overall_score"],
                "tracks": {t: {"family_macro_accuracy": v["family_macro_accuracy"],
                               "raw_accuracy": v["raw"]["accuracy"],
                               "raw_class_balanced_accuracy": v["raw"]["class_balanced_accuracy"],
                               "macro_f1": v["raw"]["macro_f1"]}
                           for t, v in off["capability"]["tracks"].items()},
                "groups_accuracy": off["groups"]["accuracy"],
                "accuracy_ci": off["accuracy_ci"],
            },
            "brier": pq["brier"],
            "nll": pq["nll"],
            "ece": pq["ece"],
            "brier_no_abstention": pq0["brier"],
            "ece_no_abstention": pq0["ece"],
            "selective": {k: sel[k] for k in (
                "answerable_total", "answered", "achieved_coverage", "achieved_error",
                "partial_aurc", "unknown_gold_total", "substantive_answers_on_unknown_gold")},
        }
        table["per_primitive"][s] = table[s]["typed"]["prim"]

    (ROOT / a.out).write_text(json.dumps(table, indent=2))
    print(json.dumps({k: v for k, v in table.items() if k != "tau_sweep"}, indent=2))

    import mlflow

    from drishyam.tracking import mlflow_run

    with mlflow_run("imajev-stock-ood") as run:
        mlflow.set_tag("gate", "2.5")
        mlflow.set_tag("benchmark", "imajevbench-v2-lite")
        mlflow.set_tag("kind", "stock-ood-gap")
        mlflow.log_params({"ckpt": a.ckpt, "tau_best": tau_best, "splits": a.splits,
                           "fixed_len": fixed_len, "n_records": len(records)})
        for s in splits:
            t = table[s]
            mlflow.log_metric(f"{s}_acc_at_tau", t["typed"]["overall"]["acc"])
            mlflow.log_metric(f"{s}_acc_no_abstention",
                              t["typed_no_abstention"]["overall"]["acc"])
            mlflow.log_metric(f"{s}_brier", t["brier"])
            mlflow.log_metric(f"{s}_ece", t["ece"])
            for tr, v in t["typed"]["track"].items():
                mlflow.log_metric(f"{s}_acc_{tr}", v["acc"])
        mlflow.log_artifact(str(ROOT / a.out))
        print("mlflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
