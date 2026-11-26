"""Text-row SFT: the TEXT-state track of the Gate-2.5 encoder-row trainer.

Clone of train_gate2_row.py (row unit, encoder-row binding) with the state side
swapped from cached image page embeddings to tokenized TEXT states:
  (1) Data: bev-decision (bev_data.load_cases) instead of gate2_labels JSONL;
      rows already ARE question rows keyed to one state (no page_key indirection).
  (2) State side: page_embs = emb_layer(state_ids padded to state_len) with a
      token mask — forward_row prepends them to the question tokens and reads
      slots at h[i, p + state_len]. No PageCache, no image tensors.
  (3) No val split exists (dataset card §5): --val-rows rows are carved out of
      the train split by seeded permutation (bev_data.carve_val); the test split
      stays sealed for eval.
  (4) --init defaults to SFT (NOT fresh): runs/g2-row-sft/best.pt carries a
      TRAINED decision head on the same encoder/slot/typed-head machinery —
      diagnostic continuity; fresh stays available.
  (5) --device arg (default cuda) so CPU smokes are possible; vram logging is
      guarded (0.0 when CUDA is absent).
Everything else (lr 2e-5, 4 epochs patience 2, rows-per-batch 16, cosine warmup,
per-step MLflow every 25 steps, pre-val ckpt, resume_flag, best/last +
val_logits.npz + report) is train_gate2_row.py, unchanged.
"""

import argparse
import contextlib
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from drishyam.bev_data import carve_val, load_cases
from drishyam.losses import question_loss
from drishyam.model import DecoupledDecision, attach_lora, build_backbone, trainable_groups
from drishyam.tokenize import add_slot_token, collate_questions, question_text
from drishyam.tracking import mlflow_run
from drishyam.train_gate2 import MODEL_ID, ece15

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def state_batch_ids(tok, states: list[str], state_len: int, device="cpu"):
    """Fixed-shape state tokenization — same invariance law as collate_questions:
    every row carries exactly state_len state tokens so batch shape never changes
    a question's logits. Truncation keeps the head (front) of the state, matching
    bev-decider's protocol (1,024-token training states)."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    raws = [
        tok(s, add_special_tokens=True, truncation=True, max_length=state_len)["input_ids"]
        for s in states
    ]
    ids = torch.tensor([r + [pad] * (state_len - len(r)) for r in raws], device=device)
    mask = torch.tensor([[1] * len(r) + [0] * (state_len - len(r)) for r in raws], device=device)
    return ids, mask


def fixed_len_for_cases(tok, cases, cap: int) -> int:
    """Max question token length over ALL cases; refuses to silently truncate
    (a truncated row loses its trailing <slot> tokens -> slot_pos comes up short
    and the head reads garbage). Fail at startup, not mid-epoch."""
    texts = [question_text(q) for c in cases for q in c["questions"]]
    longest = 0
    for start in range(0, len(texts), 2048):
        chunk = texts[start : start + 2048]
        enc = tok(chunk, add_special_tokens=True)["input_ids"]
        longest = max(longest, max(len(r) for r in enc))
    if longest > cap:
        raise ValueError(
            f"longest question is {longest} tokens > cap {cap}; raise --question-len-cap"
        )
    return max(longest, 8)


def run_rows(
    model,
    tok,
    cases,
    device,
    state_len,
    fixed_len,
    opt=None,
    sched=None,
    seed=0,
    rows_per_batch=16,
    step_log=None,
):
    """One pass over all question rows; row = (case_idx, question_idx). Each batch
    entry carries its own state (row_case indexes into the batch's unique states)."""
    import mlflow

    train = opt is not None
    if train:
        model.train()
        rng = random.Random(seed)
        rows = [(ci, qi) for ci in range(len(cases)) for qi in range(len(cases[ci]["questions"]))]
        rng.shuffle(rows)
    else:
        model.eval()
        rows = [(ci, qi) for ci in range(len(cases)) for qi in range(len(cases[ci]["questions"]))]
    total, count = 0.0, 0
    all_p, all_g = [], []
    gn = 0.0
    t_epoch = time.time()
    by_id = {c["case_id"]: c for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case_id in cases")
    for start in range(0, len(rows), rows_per_batch):
        batch = rows[start : start + rows_per_batch]
        keys = list({cases[ci]["case_id"] for ci, _ in batch})
        pidx = {k: i for i, k in enumerate(keys)}
        s_ids, s_mask = state_batch_ids(tok, [by_id[k]["state"] for k in keys], state_len, device)
        state_embs = model.bb.get_input_embeddings()(s_ids)
        qs = [cases[ci]["questions"][qi] for ci, qi in batch]
        golds = [cases[ci]["gold"][qi] for ci, qi in batch]
        dec_ids, dec_mask, slot_pos, _ = collate_questions(tok, qs, fixed_len, device)
        row_case = torch.tensor([pidx[cases[ci]["case_id"]] for ci, _ in batch], device=device)
        if train:
            opt.zero_grad(set_to_none=True)
        # Val MUST run grad-free (train_gate2_row.py: the autograd pass OOM'd the 3060).
        with contextlib.nullcontext() if train else torch.inference_mode():
            out = model.forward_row(state_embs, s_mask, dec_ids, dec_mask, slot_pos, row_case)
            per_q = [
                question_loss(logit.float(), torch.tensor(g, device=logit.device), q.primitive)
                for logit, q, g in zip(out, qs, golds, strict=False)
            ]
            loss = torch.stack(per_q).mean()
        if train:
            loss.backward()
            gn = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for grp in opt.param_groups for p in grp["params"]], 1.0
                )
            )
            opt.step()
            if sched is not None:
                sched.step()
        total += float(loss.detach()) * len(per_q)
        count += len(per_q)
        with torch.no_grad():
            for logit, g in zip(out, golds, strict=False):
                all_p.append(torch.softmax(logit.float(), -1).cpu().numpy())
                all_g.append(np.asarray(g, dtype=float))
        if train and step_log is not None:
            step_log[0] += 1
            if step_log[0] % 25 == 0:
                rps = count / max(time.time() - t_epoch, 1e-9)
                mlflow.log_metrics(
                    {
                        "train_loss": float(loss.detach()),
                        "grad_norm": gn,
                        "rows_per_s": rps,
                        "vram_mb": vram_mb(),
                    },
                    step=step_log[0],
                )
    model.train()
    return (total / max(count, 1), (np.array(all_p, dtype=object), np.array(all_g, dtype=object)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="default",
        help="bev config: default|hard_50k|numeric_temporal|skills|counterfactual_15k",
    )
    ap.add_argument("--train-limit", type=int, default=0, help="cap train rows (smoke)")
    ap.add_argument(
        "--val-rows", type=int, default=2000, help="val carved from train (no val split exists)"
    )
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--rows-per-batch", type=int, default=16)
    ap.add_argument(
        "--state-max-tokens",
        type=int,
        default=1024,
        help="bev-decider trains at 1,024-token states",
    )
    ap.add_argument("--question-len-cap", type=int, default=192)
    ap.add_argument(
        "--init",
        choices=["fresh", "sft"],
        default="sft",
        help="sft = g2-row-sft best.pt (trained decision head) — DEFAULT",
    )
    ap.add_argument("--sft-ckpt", default="runs/g2-row-sft/best.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/text-row-sft")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = a.device
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ART.mkdir(exist_ok=True)

    all_train, skipped = load_cases(a.config, "train", limit=a.train_limit)
    print(f"loaded {len(all_train)} train cases ({skipped} stateless rows skipped)", flush=True)
    train_cases, val_cases = carve_val(all_train, a.val_rows, seed=a.seed)
    n_train_rows = sum(len(c["questions"]) for c in train_cases)
    prim_counts = Counter(q.primitive.value for c in train_cases for q in c["questions"])
    print(
        f"train_cases={len(train_cases)} train_rows={n_train_rows} "
        f"val_cases={len(val_cases)} by_type={dict(prim_counts)}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    fixed_len = fixed_len_for_cases(tok, train_cases + val_cases, a.question_len_cap)
    print(f"fixed_len={fixed_len} state_len={a.state_max_tokens}", flush=True)

    bb = build_backbone(MODEL_ID)
    slot_id = add_slot_token(tok, bb)
    bb = attach_lora(bb)
    for path in ("vision_tower", "vision_model", "model.vision_tower", "model.vision_model"):
        tgt, ok = bb, True
        for part in path.split("."):
            if hasattr(tgt, part):
                tgt = getattr(tgt, part)
            else:
                ok = False
                break
        if ok and hasattr(tgt, "parameters") and next(tgt.parameters(), None) is not None:
            tgt.to("cpu")
            print(f"vision offloaded to CPU: {path}")
            break
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = DecoupledDecision(bb, slot_id).to(device)
    if a.init == "sft":
        ck = torch.load(ROOT / a.sft_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"init=sft loaded {a.sft_ckpt} (epoch {ck.get('epoch')})", flush=True)

    opt = torch.optim.AdamW(trainable_groups(model, wd=0.01), lr=a.lr)
    steps_per_epoch = (n_train_rows + a.rows_per_batch - 1) // a.rows_per_batch
    total_steps = max(steps_per_epoch * a.epochs, 1)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.10 * total_steps)), total_steps)

    cfg = {
        "lr": a.lr,
        "epochs": a.epochs,
        "seed": a.seed,
        "rows_per_batch": a.rows_per_batch,
        "fixed_len": fixed_len,
        "state_max_tokens": a.state_max_tokens,
        "config": a.config,
        "train_limit": a.train_limit,
        "val_rows": a.val_rows,
        "init": a.init,
        "sft_ckpt": a.sft_ckpt,
        "path": "row",
        "track": "text",
    }
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    start_epoch, best = 0, float("inf")
    last = out / "last.pt"
    if (out / "resume_flag").exists() and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"], ck["best"]
        print(f"RESUMED from epoch {start_epoch} best={best:.4f}")

    patience, bad = 2, 0
    t0 = time.time()
    history = []
    step_log = [0]
    with mlflow_run(Path(a.out).name) as run:
        import mlflow

        mlflow.set_tag("gate", "3")
        mlflow.set_tag("track", "text-row")
        mlflow.log_params(
            {
                **cfg,
                "cfg_hash": cfg_hash,
                "git_sha": git_sha(),
                "model_id": MODEL_ID,
                "resumed_from_epoch": start_epoch,
            }
        )
        for epoch in range(start_epoch, a.epochs):
            tr_loss, _ = run_rows(
                model,
                tok,
                train_cases,
                device,
                a.state_max_tokens,
                fixed_len,
                opt=opt,
                sched=sched,
                seed=a.seed * 1000 + epoch,
                rows_per_batch=a.rows_per_batch,
                step_log=step_log,
            )
            # Pre-val checkpoint: a val-time crash must never eat a trained epoch
            # (train_gate2_row.py — it ate epoch 0 once).
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "fixed_len": fixed_len,
                    "state_len": a.state_max_tokens,
                    "slot_id": slot_id,
                    "cfg_hash": cfg_hash,
                    "pre_val": True,
                },
                last,
            )
            vl, (vp, vg) = run_rows(model, tok, val_cases, device, a.state_max_tokens, fixed_len)
            acc = float(
                np.mean([float(p.argmax() == g.argmax()) for p, g in zip(vp, vg, strict=False)])
            )
            bri = float(
                np.mean([float(((p - g) ** 2).sum()) for p, g in zip(vp, vg, strict=False)])
            )
            ece = ece15(vp, vg)
            peak = vram_mb()
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": vl,
                    "val_acc": acc,
                    "val_brier": bri,
                    "val_ece15": ece,
                    "vram_mb": peak,
                    "lr": sched.get_last_lr()[0],
                }
            )
            mlflow.log_metrics(
                {
                    "train_loss": tr_loss,
                    "val_loss": vl,
                    "val_acc": acc,
                    "val_brier": bri,
                    "val_ece15": ece,
                    "vram_mb": peak,
                },
                step=epoch,
            )
            print(
                f"text-row-ep{epoch:02d} train={tr_loss:.4f} val={vl:.4f} "
                f"acc={acc:.4f} brier={bri:.4f} ece={ece:.4f} vram={peak:.0f}MB "
                f"({(time.time() - t0) / 60:.1f}min)",
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "epoch": epoch + 1,
                    "best": best,
                    "fixed_len": fixed_len,
                    "state_len": a.state_max_tokens,
                    "slot_id": slot_id,
                    "cfg_hash": cfg_hash,
                },
                last,
            )
            if vl < best:
                best = vl
                torch.save(
                    {
                        "model": model.state_dict(),
                        "fixed_len": fixed_len,
                        "state_len": a.state_max_tokens,
                        "slot_id": slot_id,
                        "epoch": epoch,
                        "val_loss": vl,
                    },
                    out / "best.pt",
                )
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"EARLY STOP at epoch {epoch} (patience {patience})")
                    break
        ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        _, (vp, vg) = run_rows(model, tok, val_cases, device, a.state_max_tokens, fixed_len)
        np.savez(
            ART / "text_row_val_logits.npz",
            probs=np.array(vp, dtype=object),
            gold=np.array(vg, dtype=object),
        )
        report = {
            "cfg_hash": cfg_hash,
            "git_sha": git_sha(),
            "seed": a.seed,
            **cfg,
            "n_train_cases": len(train_cases),
            "n_train_rows": n_train_rows,
            "n_val_cases": len(val_cases),
            "by_primitive": dict(prim_counts),
            "epochs_run": len(history),
            "best_val_loss": best,
            "best_epoch": int(np.argmin([h["val_loss"] for h in history])),
            "history": history,
            "peak_vram_mb": vram_mb(),
            "wall_min": (time.time() - t0) / 60,
            "mlflow_run_id": run.info.run_id,
        }
        (ART / "text_row_train_report.json").write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(ART / "text_row_train_report.json"))
    print(
        f"TEXT-ROW SFT DONE best_val={best:.4f} wall={report['wall_min']:.1f}min "
        f"vram={report['peak_vram_mb']:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
