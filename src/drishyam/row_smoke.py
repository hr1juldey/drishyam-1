"""Gate-2.5 K1 SMOKE: encoder-row binding falsification (pre-registered, plan branch).

Question: does putting question+option tokens IN THE ENCODER ROW next to the page
tokens unlock binding (ktcapraz layout) where decoder-only binding stayed at chance?
Slice: thermal-only, 300 train / 60 val pages, 2 epochs, lr 2e-5 (the oscillation
dictated the lower rate). PASS bar (plan): val acc > 0.60. FAIL => no overnight run;
next diagnosis is the projector-to-token-space variant (user's shared-space point).
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from drishyam.evaluate import load_model
from drishyam.losses import question_loss
from drishyam.tokenize import collate_questions
from drishyam.train_gate2 import IMG_TOKENS, PageCache, ece15, load_cases

ROOT = Path(__file__).resolve().parents[2]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pages", type=int, default=300)
    ap.add_argument("--val-pages", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rows-per-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/g2-row-smoke")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda"
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cache = PageCache()

    train_cases = [c for c in load_cases("train") if c["family"] == "thermal"][: a.train_pages]
    val_cases = [c for c in load_cases("val") if c["family"] == "thermal"][: a.val_pages]
    # flatten to question-rows: each row = (case_idx, question_idx)
    rows = [(ci, qi) for ci, c in enumerate(train_cases)
            for qi in range(len(c["questions"]))]
    val_rows = [(ci, qi) for ci, c in enumerate(val_cases)
                for qi in range(len(c["questions"]))]
    print(f"thermal rows: train={len(rows)} val={len(val_rows)}", flush=True)

    model, tok, fixed_len = load_model(ROOT / "runs/g2-sft/best.pt", device)
    model.float()

    import random

    rng = random.Random(a.seed)

    def run_val():
        model.eval()
        ps, gs = [], []
        with torch.no_grad():
            for ci, qi in val_rows:
                case = val_cases[ci]
                pages = torch.from_numpy(cache.get(case["page_key"]))[None].to(
                    device=device, dtype=torch.float32)
                pm = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)
                di, dm, pos, _ = collate_questions(tok, [case["questions"][qi]],
                                                   fixed_len, device)
                out_l = model.forward_row(pages, pm, di, dm, pos,
                                          torch.zeros(1, dtype=torch.long, device=device))
                ps.append(torch.softmax(out_l[0].float(), -1).cpu().numpy())
                gs.append(np.asarray(case["gold"][qi], dtype=float))
        model.train()
        acc = float(np.mean([p.argmax() == g.argmax() for p, g in zip(ps, gs, strict=False)]))
        return acc, ece15(ps, gs)

    acc0, ece0 = run_val()
    vacc, vece = acc0, ece0
    print(f"row-smoke start (SFT-init, ZERO row training): acc={acc0:.4f} ece={ece0:.4f}",
          flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr)

    import mlflow

    from drishyam.tracking import mlflow_run

    t0 = time.time()
    history = []
    with mlflow_run(Path(a.out).name):
        mlflow.set_tag("gate", "2.5")
        mlflow.log_params({**a.__dict__, "git_sha": git_sha(),
                           "cfg_hash": hashlib.sha256(
                               json.dumps(a.__dict__, sort_keys=True).encode()).hexdigest()[:12],
                           "val_acc_zero_shot": acc0})
        mlflow.log_metrics({"val_acc_zero_shot": acc0, "val_ece_zero_shot": ece0}, step=0)
        step = 0
        for epoch in range(a.epochs):
            rng.shuffle(rows)
            t_ep = time.time()
            tot, cnt = 0.0, 0
            for s in range(0, len(rows), a.rows_per_batch):
                batch = rows[s:s + a.rows_per_batch]
                qs, golds = [], []
                for ci, qi in batch:
                    case = train_cases[ci]
                    qs.append(case["questions"][qi])
                    golds.append(case["gold"][qi])
                # unique pages in this batch
                upages = list({train_cases[ci]["page_key"] for ci, _ in batch})
                pidx = {k: i for i, k in enumerate(upages)}
                pages = torch.stack([torch.from_numpy(cache.get(k)) for k in upages])
                pages = pages.to(device=device, dtype=torch.float32)
                pm = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)
                di, dm, pos, _ = collate_questions(tok, qs, fixed_len, device)
                row_case = torch.tensor([pidx[train_cases[ci]["page_key"]]
                                         for ci, _ in batch], device=device)
                opt.zero_grad(set_to_none=True)
                out_l = model.forward_row(pages, pm, di, dm, pos, row_case)
                per = [question_loss(lg.float(), torch.tensor(g, device=lg.device),
                                     q.primitive)
                       for lg, q, g in zip(out_l, qs, golds, strict=False)]
                loss = torch.stack(per).mean()
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                opt.step()
                step += 1
                tot += float(loss.detach()) * len(per)
                cnt += len(per)
                if step % 10 == 0:
                    mlflow.log_metrics({"train_loss": float(loss.detach()), "grad_norm": gn,
                                        "rows_per_s": cnt / max(time.time() - t_ep, 1e-9)},
                                       step=step)
            vacc, vece = run_val()
            mlflow.log_metrics({"val_acc": vacc, "val_ece": vece}, step=epoch)
            history.append({"epoch": epoch, "train_loss": tot / max(cnt, 1), "val_acc": vacc,
                            "val_ece": vece})
            print(f"row-smoke ep{epoch}: train={tot / max(cnt, 1):.4f} val_acc={vacc:.4f} "
                  f"ece={vece:.4f} ({(time.time() - t_ep) / 60:.1f} min)", flush=True)
            torch.save({"model": model.state_dict(), "fixed_len": fixed_len,
                        "epoch": epoch, "history": history}, out / "last.pt")
        verdict = {"pass": vacc > 0.60, "val_acc": vacc, "bar": 0.60, "history": history,
                   "zero_shot_acc": acc0, "wall_min": (time.time() - t0) / 60}
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        mlflow.log_metric("smoke_pass", float(verdict["pass"]))
        mlflow.log_artifact(str(out / "verdict.json"))
        print(f"ROW-SMOKE {'PASS' if verdict['pass'] else 'FAIL'}: "
              f"zero-shot {acc0:.4f} -> {vacc:.4f} (bar 0.60)", flush=True)


if __name__ == "__main__":
    main()
