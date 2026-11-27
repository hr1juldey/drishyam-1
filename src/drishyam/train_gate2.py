"""Gate-2 SFT: train the slot decision head + LoRA on cached page embeddings.

Differences from train.py (G1-T1), cited:
  (1) states = cached page tokens [256, 640] via page_key ("emb.npy:idx"), not tokenized text
  (2) forward through forward_embeds (page_mask = ones)
  (3) early stop: patience 3, epochs 6 (90x past the JEV/Laya data band — more epochs buy overfit)
  (4) per-step MLflow logging every 25 steps: loss, grad_norm, decisions_per_s, vram_mb
Everything else (LoRA r16 all-linear, slot token, optimizer groups, cosine warmup,
ece15, best/last ckpt, val_logits.npz for TS) is train.py, unchanged.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from drishyam.losses import question_loss
from drishyam.model import DecoupledDecision, attach_lora, build_backbone, trainable_groups
from drishyam.tokenize import add_slot_token, collate_questions
from drishyam.tracking import mlflow_run

MODEL_ID = "google/t5gemma-2-270m-270m"
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
IMG_TOKENS = 256


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def fixed_len_for(tok, cases, cap: int = 128) -> int:
    from drishyam.tokenize import question_text

    m = 0
    for c in cases:
        for q in c["questions"]:
            m = max(m, len(tok(question_text(q), add_special_tokens=True)["input_ids"]))
    return min(m, cap)


class PageCache:
    def __init__(self):
        self.mmaps = {}

    def get(self, page_key):
        path, idx = page_key.rsplit(":", 1)
        if path not in self.mmaps:
            self.mmaps[path] = np.load(ROOT / path, mmap_mode="r")
        return np.asarray(self.mmaps[path][int(idx)], dtype=np.float32)


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


def load_cases(split, path=None):
    """JSONL rows -> cases with validated Question objects (schema gate at load time)."""
    from drishyam.schemas import Question

    rows = [json.loads(l) for l in
            (ART / (path or f"gate2_labels_{split}.jsonl")).read_text().splitlines() if l.strip()]
    for r in rows:
        r["questions"] = [Question(**q) for q in r["questions"]]
    return rows


def run_epoch(model, tok, cases, device, fixed_len, cache, opt=None, sched=None, seed=0,
              step_log=None, mlflow_active=False):
    import random

    import mlflow

    rng = random.Random(seed)
    train = opt is not None
    total, count = 0.0, 0
    all_p, all_g = [], []
    t_epoch = time.time()
    for start in range(0, len(cases), 4):
        chunk = cases[start:start + 4]
        pages = torch.stack([torch.from_numpy(cache.get(c["page_key"])) for c in chunk])
        page_mask = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)
        pages = pages.to(device=device, dtype=torch.bfloat16)
        qtexts, golds = [], []
        for ci, case in enumerate(chunk):
            for q, g in zip(case["questions"], case["gold"], strict=False):
                qtexts.append((ci, q))
                golds.append(g)
        dec_ids, dec_mask, slot_pos, _ = collate_questions(
            tok, [q for _, q in qtexts], fixed_len, device)
        row_case = torch.tensor([ci for ci, _ in qtexts], device=device)
        if train:
            opt.zero_grad(set_to_none=True)
        out = model.forward_embeds(pages, page_mask, dec_ids, dec_mask, slot_pos,
                                   row_case=row_case)
        per_q = [question_loss(logit.float(), torch.tensor(g, device=logit.device),
                               q.primitive)
                 for logit, (ci, q), g in zip(out, qtexts, golds, strict=False)]
        loss = torch.stack(per_q).mean()
        if train:
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(
                [p for grp in opt.param_groups for p in grp["params"]], 1.0))
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
            if step_log[0] % 25 == 0 and mlflow_active:
                dps = count / max(time.time() - t_epoch, 1e-9)
                mlflow.log_metrics({
                    "train_loss": float(loss.detach()),
                    "grad_norm": gn,
                    "decisions_per_s": dps,
                    "vram_mb": torch.cuda.max_memory_allocated() / 1e6,
                }, step=step_log[0])
    return (total / max(count, 1),
            (np.array(all_p, dtype=object), np.array(all_g, dtype=object)))


def evaluate_split(model, tok, cases, device, fixed_len, cache):
    losses, all_p, all_g = [], [], []
    model.eval()
    for case in cases:
        pages = torch.from_numpy(cache.get(case["page_key"]))[None].to(
            device=device, dtype=torch.bfloat16)
        page_mask = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)
        dec_ids, dec_mask, slot_pos, _ = collate_questions(
            tok, case["questions"], fixed_len, device)
        with torch.no_grad():
            out = model.forward_embeds(pages, page_mask, dec_ids, dec_mask, slot_pos)
        for logit, g, q in zip(out, case["gold"], case["questions"], strict=False):
            gf = torch.tensor(g, device=logit.device)
            losses.append(float(question_loss(logit.float(), gf, q.primitive)))
            all_p.append(torch.softmax(logit.float(), -1).cpu().numpy())
            all_g.append(np.asarray(g, dtype=float))
    model.train()
    p, g = np.array(all_p, dtype=object), np.array(all_g, dtype=object)
    return float(np.mean(losses)), p, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--smoke-pages", type=int, default=0, help="cap train+val cases (smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-cases", type=int, default=4)
    ap.add_argument("--out", default="runs/g2-sft")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda"
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ART.mkdir(exist_ok=True)

    train_cases = load_cases("train")
    val_cases = load_cases("val")
    if a.smoke_pages:
        train_cases = train_cases[: a.smoke_pages]
        val_cases = val_cases[: max(8, a.smoke_pages // 8)]
    cache = PageCache()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    fixed_len = fixed_len_for(tok, train_cases[:2000] + val_cases[:500])
    print(f"fixed_len={fixed_len} train_cases={len(train_cases)} "
          f"val_cases={len(val_cases)}", flush=True)

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
    opt = torch.optim.AdamW(trainable_groups(model, wd=0.01), lr=a.lr)
    steps_per_epoch = (len(train_cases) + a.batch_cases - 1) // a.batch_cases
    total_steps = max(steps_per_epoch * a.epochs, 1)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.10 * total_steps)), total_steps)

    cfg = {"lr": a.lr, "epochs": a.epochs, "seed": a.seed, "batch_cases": a.batch_cases,
           "fixed_len": fixed_len, "smoke_pages": a.smoke_pages}
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

    patience, bad = 3, 0
    t0 = time.time()
    history = []
    step_log = [0]
    with mlflow_run(Path(a.out).name) as run:
        import mlflow

        mlflow.set_tag("gate", "2")
        mlflow.log_params({**cfg, "cfg_hash": cfg_hash, "git_sha": git_sha(),
                           "model_id": MODEL_ID, "resumed_from_epoch": start_epoch})
        for epoch in range(start_epoch, a.epochs):
            model.train()
            tr_loss, _ = run_epoch(model, tok, train_cases, device, fixed_len, cache,
                                   opt=opt, sched=sched, seed=a.seed * 1000 + epoch,
                                   step_log=step_log, mlflow_active=True)
            vl, vp, vg = evaluate_split(model, tok, val_cases, device, fixed_len, cache)
            acc = float(np.mean([float(p.argmax() == g.argmax())
                                 for p, g in zip(vp, vg, strict=False)]))
            bri = float(np.mean([float(((p - g) ** 2).sum())
                                 for p, g in zip(vp, vg, strict=False)]))
            ece = ece15(vp, vg)
            peak = torch.cuda.max_memory_allocated() / 1e6
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl,
                            "val_acc": acc, "val_brier": bri, "val_ece15": ece,
                            "vram_mb": peak, "lr": sched.get_last_lr()[0]})
            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": vl, "val_acc": acc,
                                "val_brier": bri, "val_ece15": ece, "vram_mb": peak},
                               step=epoch)
            print(f"ep{epoch:02d} train={tr_loss:.4f} val={vl:.4f} acc={acc:.4f} "
                  f"brier={bri:.4f} ece={ece:.4f} vram={peak:.0f}MB", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch + 1, "best": best,
                        "fixed_len": fixed_len, "slot_id": slot_id, "cfg_hash": cfg_hash},
                       last)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "fixed_len": fixed_len,
                            "slot_id": slot_id, "epoch": epoch, "val_loss": vl},
                           out / "best.pt")
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"EARLY STOP at epoch {epoch} (patience {patience})")
                    break
        ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        _, vp, vg = evaluate_split(model, tok, val_cases, device, fixed_len, cache)
        np.savez(ART / "gate2_val_logits.npz",
                 probs=np.array(vp, dtype=object), gold=np.array(vg, dtype=object))
        report = {
            "cfg_hash": cfg_hash, "git_sha": git_sha(), "seed": a.seed, **cfg,
            "epochs_run": len(history), "best_val_loss": best,
            "best_epoch": int(np.argmin([h["val_loss"] for h in history])),
            "history": history, "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6,
            "wall_min": (time.time() - t0) / 60,
            "mlflow_run_id": run.info.run_id,
        }
        (ART / "gate2_train_report.json").write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(ART / "gate2_train_report.json"))
    print(f"G2 SFT DONE best_val={best:.4f} wall={report['wall_min']:.1f}min "
          f"vram={report['peak_vram_mb']:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
