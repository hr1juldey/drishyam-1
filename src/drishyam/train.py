"""G1-T1: LoRA training — proper-scoring losses, anti-shortcut augs, MLflow, resume."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from drishyam.augs import jitter_state, shuffled_options
from drishyam.dataset import load_seen_splits
from drishyam.losses import question_loss
from drishyam.model import DecoupledDecision, attach_lora, build_backbone, trainable_groups
from drishyam.tokenize import add_slot_token, collate_questions, question_text
from drishyam.tracking import mlflow_run

MODEL_ID = "google/t5gemma-2-270m-270m"
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def fixed_len_for(tok, datasets, cap: int = 128) -> int:
    m = 0
    for ds in datasets:
        for i in range(len(ds)):
            for q in ds[i]["questions"]:
                m = max(m, len(tok(question_text(q), add_special_tokens=True)["input_ids"]))
    return min(m, cap)


def state_batch(tok, states, device):
    enc = [tok(s, add_special_tokens=True, truncation=True, max_length=1024)["input_ids"]
           for s in states]
    longest = max(len(e) for e in enc)
    ids = torch.tensor([e + [tok.pad_token_id] * (longest - len(e)) for e in enc], device=device)
    mask = torch.tensor([[1] * len(e) + [0] * (longest - len(e)) for e in enc], device=device)
    return ids, mask


def ece15(probs: list, golds: list) -> float:
    """Ragged per-question ECE over 15 confidence bins."""
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


def run_epoch(model, tok, cases, device, fixed_len, opt=None, sched=None, augment=False, seed=0):
    """One pass; returns (mean_loss, (probs,gold)) over questions."""
    rng = random.Random(seed)
    train = opt is not None
    total, count = 0.0, 0
    all_p, all_g = [], []
    for start in range(0, len(cases), 8):
        chunk = cases[start:start + 8]
        states, qtexts, golds, prims = [], [], [], []
        for ci, case in enumerate(chunk):
            st = jitter_state(case["state"], rng) if augment else case["state"]
            states.append(st)
            for q, g in zip(case["questions"], case["gold"], strict=False):
                if augment and q.primitive.value == "choice":
                    opts, g = shuffled_options(q.options, g, "choice", rng)
                    q = q.model_copy(update={"options": opts})
                qtexts.append((ci, q))
                golds.append(g)
                prims.append(q.primitive.value)
        s_ids, s_mask = state_batch(tok, states, device)
        dec_ids, dec_mask, slot_pos, _ = collate_questions(
            tok, [q for _, q in qtexts], fixed_len, device)
        row_case = torch.tensor([ci for ci, _ in qtexts], device=device)
        if train:
            opt.zero_grad(set_to_none=True)
        out = model(s_ids, s_mask, dec_ids, dec_mask, slot_pos, row_case=row_case)
        per_q = [question_loss(logit.float(), torch.tensor(g, device=logit.device), p)
                 for logit, g, p in zip(out, golds, prims, strict=False)]
        loss = torch.stack(per_q).mean()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for grp in opt.param_groups for p in grp["params"]], 1.0)
            opt.step()
            if sched is not None:
                sched.step()
        total += float(loss.detach()) * len(per_q)
        count += len(per_q)
        with torch.no_grad():
            for logit, g in zip(out, golds, strict=False):
                all_p.append(torch.softmax(logit.float(), -1).cpu().numpy())
                all_g.append(np.asarray(g, dtype=float))
    return (total / max(count, 1),
            (np.array(all_p, dtype=object), np.array(all_g, dtype=object), prims))


def evaluate_split(model, tok, cases, device, fixed_len):
    """Eval mode: each case's questions as ONE padded fixed-shape batch off its cached H_enc."""
    losses, all_p, all_g, prims = [], [], [], []
    model.eval()
    for case in cases:
        s_ids, s_mask = state_batch(tok, [case["state"]], device)
        dec_ids, dec_mask, slot_pos, _ = collate_questions(
            tok, case["questions"], fixed_len, device)
        with torch.no_grad():
            out = model(s_ids, s_mask, dec_ids, dec_mask, slot_pos)
        for logit, g, q in zip(out, case["gold"], case["questions"], strict=False):
            gf = torch.tensor(g, device=logit.device)
            losses.append(float(question_loss(logit.float(), gf, q.primitive.value)))
            all_p.append(torch.softmax(logit.float(), -1).cpu().numpy())
            all_g.append(np.asarray(g, dtype=float))
            prims.append(q.primitive.value)
    model.train()
    p, g = np.array(all_p, dtype=object), np.array(all_g, dtype=object)
    return float(np.mean(losses)), p, g, prims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-cases", type=int, default=0, help="smoke: cap per workflow")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-cases", type=int, default=4,
                    help="8-case batches OOM: merged-attn masks scale with decoder rows")
    ap.add_argument("--out", default="runs/g1-t1")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ART.mkdir(exist_ok=True)

    train_sets, val_sets = load_seen_splits()
    train_cases, val_cases = [], []
    for _, ds in train_sets.items():
        cs = [ds[i] for i in range(len(ds))]
        if a.max_cases:
            cs = cs[:a.max_cases // 3]
        train_cases += cs
    for _, ds in val_sets.items():
        val_cases += [ds[i] for i in range(len(ds))]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    fixed_len = fixed_len_for(tok, list(train_sets.values()) + list(val_sets.values()))
    print(f"fixed_len={fixed_len} train_cases={len(train_cases)} val_cases={len(val_cases)}")

    bb = build_backbone(MODEL_ID)
    slot_id = add_slot_token(tok, bb)
    bb = attach_lora(bb)
    # text-only Gate 1: the vision tower never runs, so it must not hold VRAM
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
    print("gradient checkpointing: on")
    model = DecoupledDecision(bb, slot_id).to(device)
    opt = torch.optim.AdamW(trainable_groups(model, wd=0.01), lr=a.lr)
    steps_per_epoch = (len(train_cases) + a.batch_cases - 1) // a.batch_cases
    total_steps = max(steps_per_epoch * a.epochs, 1)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.10 * total_steps)), total_steps)

    cfg = {"lr": a.lr, "epochs": a.epochs, "seed": a.seed, "batch": a.batch_cases,
           "fixed_len": fixed_len, "max_cases": a.max_cases}
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

    patience, bad = 5, 0
    t0 = time.time()
    history = []
    with mlflow_run(a.out.replace("/", "-")) as run:
        import mlflow

        mlflow.log_params({
            **cfg, "cfg_hash": cfg_hash, "git_sha": git_sha(),
            "model_id": MODEL_ID, "resumed_from_epoch": start_epoch,
        })
        for epoch in range(start_epoch, a.epochs):
            model.train()
            tr_loss, _ = run_epoch(model, tok, train_cases, device, fixed_len,
                                   opt=opt, sched=sched, augment=True,
                                   seed=a.seed * 1000 + epoch)
            vl, vp, vg, vprims = evaluate_split(model, tok, val_cases, device, fixed_len)
            acc = float(np.mean([float(p.argmax() == g.argmax())
                                 for p, g in zip(vp, vg, strict=False)]))
            bri = float(np.mean([float(((p - g) ** 2).sum())
                                 for p, g in zip(vp, vg, strict=False)]))
            ece = ece15(vp, vg)
            peak = torch.cuda.max_memory_allocated() / 1e6
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl,
                            "val_acc": acc, "val_brier": bri, "val_ece15": ece,
                            "vram_mb": peak, "lr": sched.get_last_lr()[0]})
            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            mlflow.log_metric("val_loss", vl, step=epoch)
            mlflow.log_metric("val_acc", acc, step=epoch)
            mlflow.log_metric("val_brier", bri, step=epoch)
            mlflow.log_metric("val_ece15", ece, step=epoch)
            mlflow.log_metric("vram_mb", peak, step=epoch)
            print(f"ep{epoch:02d} train={tr_loss:.4f} val={vl:.4f} acc={acc:.4f} "
                  f"brier={bri:.4f} ece={ece:.4f} vram={peak:.0f}MB", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch + 1, "best": best,
                        "fixed_len": fixed_len, "slot_id": slot_id, "cfg_hash": cfg_hash}, last)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "fixed_len": fixed_len,
                            "slot_id": slot_id, "epoch": epoch, "val_loss": vl}, out / "best.pt")
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"EARLY STOP at epoch {epoch} (patience {patience})")
                    break
        ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        _, vp, vg, vprims = evaluate_split(model, tok, val_cases, device, fixed_len)
        np.savez(ART / "val_logits.npz",
                 probs=np.array(vp, dtype=object), gold=np.array(vg, dtype=object),
                 prims=np.array(vprims))
        report = {
            "cfg_hash": cfg_hash, "git_sha": git_sha(), "seed": a.seed, **cfg,
            "epochs_run": len(history), "best_val_loss": best,
            "best_epoch": int(np.argmin([h["val_loss"] for h in history])),
            "history": history, "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6,
            "wall_min": (time.time() - t0) / 60, "fixed_len": fixed_len,
            "mlflow_run_id": run.info.run_id,
        }
        (ART / "train_report.json").write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(ART / "train_report.json"))
    print(f"T1 DONE best_val={best:.4f} wall={report['wall_min']:.1f}min "
          f"vram={report['peak_vram_mb']:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
