"""Gate-2.5 SFT: encoder-row binding trainer (the "actual good way", post-verdict).

Differences from train_gate2.py (Gate-2 v1, decoder path — formally dead 2026-11-25,
2/5 bars, unseen AUC 0.174 < chance), cited:
  (1) forward_row (encoder-row binding): each QUESTION is one encoder row
      [page_embs(256) ; question+option tokens], slots read from ENCODER output
      shifted by the page segment. Textbook (BEV Decider / sF3CNPbWA8o) and
      ktcapraz card both put the question IN pretrained self-attention with the
      state; their decoder-only ablations (ours and theirs) stay at chance.
  (2) Training unit = question ROW (not case); a batch is R rows sharing pages
      via row_case. Row count = decisions (77k), not cases (15k).
  (3) lr default 2e-5 (v1's 1e-4 produced grad-clip 49% + memorize/tear cycle;
      textbook + ktcapraz both train ~1e-5 regime).
  (4) epochs 4, patience 2 (bars gate the run; more epochs bought nothing in v1).
  (5) --init sft loads v1 best.pt weights (diagnostic continuity with the smoke);
      default FRESH (textbook: fresh LoRA on the pretrained backbone).
Everything else (build_backbone/attach_lora r16 all-linear, slot token, vision
offload, grad checkpointing, cosine warmup, per-step MLflow every 25 steps,
best/last ckpt + val_logits.npz, fp32 six-bar eval afterwards) unchanged.
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from drishyam.losses import question_loss
from drishyam.model import DecoupledDecision, attach_lora, build_backbone, trainable_groups
from drishyam.tokenize import add_slot_token, collate_questions
from drishyam.tracking import mlflow_run
from drishyam.train_gate2 import IMG_TOKENS, MODEL_ID, PageCache, ece15, fixed_len_for, load_cases

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def row_forward(model, pages, page_mask, dec_ids, dec_mask, slot_pos):
    """Rows in, logits out; row_case=0 because each batch entry carries its page."""
    rc = torch.zeros(pages.size(0), dtype=torch.long, device=pages.device)
    return model.forward_row(pages, page_mask, dec_ids, dec_mask, slot_pos, rc)


def run_rows(model, tok, cases, device, fixed_len, cache, opt=None, sched=None,
             seed=0, rows_per_batch=16, step_log=None, epoch=0):
    """One pass over all question rows; row = (case_idx, question_idx)."""
    import mlflow

    train = opt is not None
    if train:
        model.train()
        rng = random.Random(seed)
        rows = [(ci, qi) for ci in range(len(cases))
                for qi in range(len(cases[ci]["questions"]))]
        rng.shuffle(rows)
    else:
        model.eval()
        rows = [(ci, qi) for ci in range(len(cases))
                for qi in range(len(cases[ci]["questions"]))]
    total, count = 0.0, 0
    all_p, all_g = [], []
    gn = 0.0
    t_epoch = time.time()
    for start in range(0, len(rows), rows_per_batch):
        batch = rows[start:start + rows_per_batch]
        keys = list({cases[ci]["page_key"] for ci, _ in batch})
        pidx = {k: i for i, k in enumerate(keys)}
        pages = torch.stack([torch.from_numpy(cache.get(k)) for k in keys])
        page_mask = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long,
                               device=device)
        pages = pages.to(device=device, dtype=torch.bfloat16)
        qs = [cases[ci]["questions"][qi] for ci, qi in batch]
        golds = [cases[ci]["gold"][qi] for ci, qi in batch]
        dec_ids, dec_mask, slot_pos, _ = collate_questions(tok, qs, fixed_len, device)
        row_case = torch.tensor([pidx[cases[ci]["page_key"]] for ci, _ in batch],
                                device=device)
        if train:
            opt.zero_grad(set_to_none=True)
        out = model.forward_row(pages, page_mask, dec_ids, dec_mask, slot_pos,
                                row_case)
        per_q = [question_loss(logit.float(), torch.tensor(g, device=logit.device),
                               q.primitive)
                 for logit, q, g in zip(out, qs, golds, strict=False)]
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
            if step_log[0] % 25 == 0:
                dps = count / max(time.time() - t_epoch, 1e-9)
                mlflow.log_metrics({
                    "train_loss": float(loss.detach()),
                    "grad_norm": gn,
                    "rows_per_s": dps,
                    "vram_mb": torch.cuda.max_memory_allocated() / 1e6,
                }, step=step_log[0])
    model.train()
    return (total / max(count, 1),
            (np.array(all_p, dtype=object), np.array(all_g, dtype=object)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--rows-per-batch", type=int, default=16)
    ap.add_argument("--smoke-pages", type=int, default=0, help="cap train cases (smoke)")
    ap.add_argument("--init", choices=["fresh", "sft"], default="fresh",
                    help="sft = load v1 best.pt weights (diagnostic continuity)")
    ap.add_argument("--sft-ckpt", default="runs/g2-sft/best.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/g2-row-sft")
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

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    fixed_len = fixed_len_for(tok, train_cases[:2000] + val_cases[:500])
    n_train_rows = sum(len(c["questions"]) for c in train_cases)
    print(f"fixed_len={fixed_len} train_cases={len(train_cases)} "
          f"train_rows={n_train_rows} val_cases={len(val_cases)}", flush=True)

    bb = build_backbone(MODEL_ID)
    slot_id = add_slot_token(tok, bb)
    bb = attach_lora(bb)
    for path in ("vision_tower", "vision_model", "model.vision_tower",
                 "model.vision_model"):
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
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.10 * total_steps)),
                                            total_steps)

    cfg = {"lr": a.lr, "epochs": a.epochs, "seed": a.seed,
           "rows_per_batch": a.rows_per_batch, "fixed_len": fixed_len,
           "smoke_pages": a.smoke_pages, "init": a.init, "path": "row"}
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

        mlflow.set_tag("gate", "2.5")
        mlflow.log_params({**cfg, "cfg_hash": cfg_hash, "git_sha": git_sha(),
                           "model_id": MODEL_ID, "resumed_from_epoch": start_epoch})
        for epoch in range(start_epoch, a.epochs):
            tr_loss, _ = run_rows(model, tok, train_cases, device, fixed_len, cache,
                                  opt=opt, sched=sched, seed=a.seed * 1000 + epoch,
                                  rows_per_batch=a.rows_per_batch, step_log=step_log,
                                  epoch=epoch)
            vl, (vp, vg) = run_rows(model, tok, val_cases, device, fixed_len, cache)
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
            print(f"row-ep{epoch:02d} train={tr_loss:.4f} val={vl:.4f} acc={acc:.4f} "
                  f"brier={bri:.4f} ece={ece:.4f} vram={peak:.0f}MB "
                  f"({(time.time() - t0) / 60:.1f}min)", flush=True)
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
        _, (vp, vg) = run_rows(model, tok, val_cases, device, fixed_len, cache)
        np.savez(ART / "gate2_row_val_logits.npz",
                 probs=np.array(vp, dtype=object), gold=np.array(vg, dtype=object))
        report = {
            "cfg_hash": cfg_hash, "git_sha": git_sha(), "seed": a.seed, **cfg,
            "epochs_run": len(history), "best_val_loss": best,
            "best_epoch": int(np.argmin([h["val_loss"] for h in history])),
            "history": history, "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6,
            "wall_min": (time.time() - t0) / 60,
            "mlflow_run_id": run.info.run_id,
        }
        (ART / "gate2_row_train_report.json").write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(ART / "gate2_row_train_report.json"))
    print(f"G2.5 ROW SFT DONE best_val={best:.4f} wall={report['wall_min']:.1f}min "
          f"vram={report['peak_vram_mb']:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
