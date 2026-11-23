"""G1-S0: 20 bf16 AdamW steps must be 100% finite with decreasing loss (both arms)."""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM

from drishyam.schemas import SanityReport, SanityStep
from drishyam.tracking import mlflow_run

MODEL_ID = "google/t5gemma-2-270m-270m"
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


class ManualAdamW(torch.optim.Optimizer):
    """Plain-PyTorch AdamW, eps=1e-6 (spec §0 fallback arm)."""

    def __init__(self, params, lr=1e-4, weight_decay=0.01):
        super().__init__(params, {
            "lr": lr, "weight_decay": weight_decay, "eps": 1e-6, "betas": (0.9, 0.999),
        })

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["t"] = 0
                st["t"] += 1
                b1, b2 = g["betas"]
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mh = st["m"] / (1 - b1 ** st["t"])
                vh = st["v"] / (1 - b2 ** st["t"])
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.addcdiv_(mh, vh.sqrt().add_(g["eps"]), value=-g["lr"])


def load_backbone():
    try:
        m = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        return m.to("cuda")
    except (ValueError, KeyError, OSError):
        from transformers import T5Gemma2ForConditionalGeneration

        return T5Gemma2ForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")


def run_arm(name: str, lr: float = 1e-4) -> SanityReport:
    torch.manual_seed(42)
    model = load_backbone()
    vocab = model.get_input_embeddings().num_embeddings
    enc = torch.randint(0, vocab, (4, 24), device="cuda")  # FIXED batch: memorise it in 20 steps
    dec = torch.randint(0, vocab, (4, 16), device="cuda")
    opt = (torch.optim.AdamW if name == "torch.AdamW" else ManualAdamW)(
        model.parameters(), lr=lr, weight_decay=0.01
    )
    steps = []
    for step in range(1, 21):
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=enc, labels=dec).loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        opt.step()
        steps.append(
            SanityStep(
                step=step,
                loss=float(loss),
                grad_norm=float(gn),
                finite=bool(torch.isfinite(loss).item() and torch.isfinite(gn).item()),
            )
        )
    f5 = sum(s.loss for s in steps[:5]) / 5
    l5 = sum(s.loss for s in steps[-5:]) / 5
    del model
    torch.cuda.empty_cache()
    return SanityReport(
        model_id=MODEL_ID,
        optimizer=name,
        device="cuda",
        torch_version=torch.__version__,
        steps=steps,
        loss_decreased=bool(l5 < f5),
        all_finite=all(s.finite for s in steps),
    )


def main() -> None:
    out = {}
    with mlflow_run("g1-s0"):
        for name in ("torch.AdamW", "ManualAdamW"):
            rep = run_arm(name)
            if not rep.passed and name == "torch.AdamW":  # named single retry (DECISIONS §9 style)
                rep = run_arm(name, lr=3e-4)
                out["torch.AdamW_lr3e4_retry"] = rep.model_dump()
            else:
                out[name] = rep.model_dump()
            f5 = sum(s.loss for s in rep.steps[:5]) / 5
            l5 = sum(s.loss for s in rep.steps[-5:]) / 5
            print(f"[{name}] finite={rep.all_finite} decreased={rep.loss_decreased} "
                  f"first5={f5:.4f} last5={l5:.4f}")
    (ART / "sanity_gate.json").write_text(json.dumps(out, indent=2))
    ok = any(r["all_finite"] and r["loss_decreased"] for r in out.values())
    print("S0 PASS" if ok else "S0 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
