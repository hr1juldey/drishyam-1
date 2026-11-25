"""Gate-2 RLCD (Reinforcement Learning from Calibrated Decisions) — rung 2 after SFT.

Recipe (Laya decision-model card, banked 2026-11-25):
  policy reports the full distribution; exploration = Gaussian noise on logits;
  reward = strictly proper scoring rules (log-score + spherical; RPS for ordinal score);
  REINFORCE with group baseline; soft-CE anchor prevents drift off the SFT solution.

Abort rails (pre-registered): if windowed in-domain acc drops >1 pt OR ECE rises
>0.02 across a 10-step window, ABORT (exit 2) and keep the SFT checkpoint untouched.
Smoke first: --max-steps 20 (~200 decisions) before any long run.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import mlflow as m
import numpy as np
import torch
import torch.nn.functional as F

from drishyam.evaluate import load_model
from drishyam.tokenize import collate_questions
from drishyam.train_gate2 import IMG_TOKENS, PageCache, ece15, load_cases

ROOT = Path(__file__).resolve().parents[2]


def spherical(p: torch.Tensor, gold_idx: torch.Tensor) -> torch.Tensor:
    # strictly proper on the simplex: sum_i p_i * sqrt(p_i) / ||p||_2 maximized at gold=1
    l2 = p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return (p * p.sqrt() / l2).gather(-1, gold_idx[:, None]).squeeze(-1)


def rps(p: torch.Tensor, gold_idx: torch.Tensor) -> torch.Tensor:
    # ranked probability score for ORDINAL score primitive (options ordered L1..LK)
    K = p.size(-1)
    cdf = torch.cumsum(p, dim=-1)
    onehot = F.one_hot(gold_idx, K).float()
    gold_cdf = torch.cumsum(onehot, dim=-1)
    return -((cdf - gold_cdf) ** 2).sum(dim=-1)  # negative RPS = reward


def reward_of(logp: torch.Tensor, gold_idx: torch.Tensor,
              prim_mask_score: torch.Tensor) -> torch.Tensor:
    """Strictly proper rewards per question: log + spherical for all, + RPS for score rows."""
    p = logp.exp()
    r = logp.gather(-1, gold_idx[:, None]).squeeze(-1)  # log score (proper)
    r = r + 0.5 * spherical(p, gold_idx)                # + spherical (proper)
    if prim_mask_score.any():
        r = r + 0.5 * torch.where(prim_mask_score,
                                  rps(p, gold_idx), torch.zeros_like(r))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/g2-sft/best.pt")
    ap.add_argument("--out", default="runs/g2-rlcd")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full run")
    ap.add_argument("--noise-std", type=float, default=0.3)
    ap.add_argument("--anchor", type=float, default=0.2, help="soft-CE anchor weight")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-cases", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    device = "cuda"
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cache = PageCache()
    train_cases = load_cases("train")
    val_cases = load_cases("val")[:300]  # windowed rails on a fixed val slice

    model, tok, fixed_len = load_model(ROOT / a.ckpt, device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr)
    rng = random.Random(7)

    def measure(cases):
        model.eval()
        ps, gs = [], []
        with torch.no_grad():
            for case in cases:
                pages = torch.from_numpy(cache.get(case["page_key"]))[None].to(
                    device=device, dtype=torch.bfloat16)
                pm = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)
                di, dm, pos, _ = collate_questions(tok, case["questions"], fixed_len, device)
                out_l = model.forward_embeds(pages, pm, di, dm, pos)
                for lg, g in zip(out_l, case["gold"], strict=False):
                    ps.append(torch.softmax(lg.float(), -1).cpu().numpy())
                    gs.append(np.asarray(g, dtype=float))
        model.train()
        acc = float(np.mean([p.argmax() == g.argmax() for p, g in zip(ps, gs, strict=False)]))
        return acc, ece15(ps, gs)

    acc0, ece0 = measure(val_cases)
    print(f"RLCD start: val acc={acc0:.4f} ece={ece0:.4f}", flush=True)

    from drishyam.tracking import mlflow_run

    step, window = 0, {"rewards": [], "acc": [], "ece": []}
    history = []
    with mlflow_run(Path(a.out).name):
        m.set_tag("gate", "2")
        m.log_params(dict(a.__dict__.items()))
        m.log_metrics({"val_acc_start": acc0, "val_ece_start": ece0}, step=0)
        order = list(range(len(train_cases)))
        rng.shuffle(order)
        aborted = None
        for start in range(0, len(order), a.batch_cases):
            chunk = [train_cases[i] for i in order[start:start + a.batch_cases]]
            pages = torch.stack([torch.from_numpy(cache.get(c["page_key"])) for c in chunk])
            pages = pages.to(device=device, dtype=torch.bfloat16)
            pm = torch.ones(pages.size(0), IMG_TOKENS, dtype=torch.long, device=device)
            qtexts, golds = [], []
            for ci, case in enumerate(chunk):
                for q, g in zip(case["questions"], case["gold"], strict=False):
                    qtexts.append((ci, q))
                    golds.append(g)
            di, dm, pos, _ = collate_questions(tok, [q for _, q in qtexts], fixed_len, device)
            row_case = torch.tensor([ci for ci, _ in qtexts], device=device)
            out = model.forward_embeds(pages, pm, di, dm, pos, row_case=row_case)
            gold_idx = torch.stack([torch.tensor(np.asarray(g).argmax(), device=device)
                                    for g in golds])
            logp = F.log_softmax(torch.stack([lg.float() for lg in out]), dim=-1)
            score_mask = torch.stack(
                [torch.tensor(q.primitive.value == "score", device=device)
                 for _, q in qtexts])
            r = reward_of(logp, gold_idx, score_mask)  # group = this decoder batch
            baseline = r.mean()
            reinforce = -((r - baseline).detach() * logp.gather(
                -1, gold_idx[:, None]).squeeze(-1)).mean()
            anchor = F.nll_loss(logp, gold_idx)
            noise = torch.randn_like(logp) * a.noise_std  # Gaussian logit-noise exploration
            explore = reward_of(logp + noise, gold_idx, score_mask).mean()
            loss = (1 - a.anchor) * reinforce + a.anchor * anchor - 0.1 * explore
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            with torch.no_grad():
                batch_acc = float((logp.argmax(-1) == gold_idx).float().mean())
                batch_ece = ece15([torch.softmax(lg.float(), -1).cpu().numpy()
                                   for lg in out], [np.asarray(g) for g in golds])
            window["rewards"].append(float(r.mean()))
            window["acc"].append(batch_acc)
            window["ece"].append(batch_ece)
            if step % 10 == 0:
                wa, we = float(np.mean(window["acc"])), float(np.mean(window["ece"]))
                wr = float(np.mean(window["rewards"]))
                m.log_metrics({"reward": wr, "win_acc": wa, "win_ece": we}, step=step)
                print(f"step {step}: reward={wr:.4f} acc={wa:.4f} ece={we:.4f}", flush=True)
                history.append({"step": step, "reward": wr, "acc": wa, "ece": we})
                # --- abort rails (pre-registered) ---
                if wa < acc0 - 0.01:
                    aborted = f"acc dropped: {wa:.4f} < {acc0 - 0.01:.4f}"
                if we > ece0 + 0.02:
                    aborted = f"ece rose: {we:.4f} > {ece0 + 0.02:.4f}"
                window = {"rewards": [], "acc": [], "ece": []}
                torch.save({"model": model.state_dict(), "fixed_len": fixed_len,
                            "step": step, "history": history}, out / "last.pt")
                if aborted:
                    print(f"RLCD ABORT: {aborted} — SFT checkpoint untouched", flush=True)
                    (out / "abort.json").write_text(json.dumps(
                        {"reason": aborted, "step": step, "history": history}, indent=2))
                    m.log_metric("aborted", 1.0, step=step)
                    sys.exit(2)
            if a.max_steps and step >= a.max_steps:
                break
        acc1, ece1 = measure(val_cases)
        m.log_metrics({"val_acc_end": acc1, "val_ece_end": ece1}, step=step)
        m.log_artifact(str(out / "last.pt"))
        (out / "report.json").write_text(json.dumps({
            "start": {"acc": acc0, "ece": ece0}, "end": {"acc": acc1, "ece": ece1},
            "steps": step, "anchor": a.anchor, "noise_std": a.noise_std,
            "history": history}, indent=2))
        print(f"RLCD DONE: acc {acc0:.4f}->{acc1:.4f} ece {ece0:.4f}->{ece1:.4f}", flush=True)


if __name__ == "__main__":
    main()
