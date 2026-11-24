"""2-lakh JIT trainer. Single process: producer thread + optimizer loop.
Faithful to every measured landmine: freeze (SigLIP + text encoder + lm_head)
BEFORE the optimizer, gradient checkpointing on, decoder sees only the short
answer (logits stay small), VRAM guard checkpoints and pauses instead of dying."""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
_gpu = __import__("importlib").import_module("2d_gpu")
_gpu.TD = torch.float32

def load_cfg(path):
    with open(path) as f:
        c = json.load(f)
    c["model_snapshot"] = os.path.expanduser(c["model_snapshot"])
    return c

def build_model(cfg):
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_snapshot"], dtype=torch.bfloat16).to("cuda")
    # freeze BEFORE any optimizer exists (AdamW allocates fp32 state only for
    # params with grads; SigLIP 417M frozen = 3.3GB state avoided)
    for n, p in model.named_parameters():
        if ("vision_tower" in n or "encoder.text_model" in n or "lm_head" in n):
            p.requires_grad = False
    model.gradient_checkpointing_enable()
    model.model.encoder.vision_tower.eval()
    model.train()
    return model

def trainables(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]

def build_opt(model, cfg):
    dec, proj, other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("model.decoder"):
            dec.append(p)
        elif "multi_modal_projector" in n:
            proj.append(p)
        else:
            other.append(p)
    lr_d, lr_p = cfg.get("lr_decoder", 5e-5), cfg.get("lr_projector", 3e-4)
    wd = cfg.get("weight_decay", 0.01)
    groups = [{"params": dec, "lr": lr_d}, {"params": proj, "lr": lr_p}]
    if other:
        groups.append({"params": other, "lr": lr_d})
    return torch.optim.AdamW(groups, weight_decay=wd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="artifacts/paper_rig/train_config.json")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--rows", type=int, default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.max_steps:
        cfg["total_steps"] = args.max_steps

    from jit_pipeline import build_manifest
    rows, counts = build_manifest(int(args.rows or cfg["manifest_rows"]))
    need = cfg["total_steps"] * cfg["grad_accum"] * cfg["micro_batch"] + 64
    from producer import JITProducer
    prod = JITProducer(rows[:min(need, len(rows))], micro=cfg["micro_batch"])
    prod.start()

    model = build_model(cfg)
    opt = build_opt(model, cfg)                     # AFTER freeze
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.0f}M | steps {cfg['total_steps']} | "
          f"micro {cfg['micro_batch']} x accum {cfg['grad_accum']} | "
          f"manifest {counts}", flush=True)

    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "train.log"), "a")

    def save(tag):
        torch.save({"model": model.state_dict(), "step": step},
                   os.path.join(run_dir, f"ckpt_{tag}.pt"))
        ck = sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_"))
        while len(ck) > cfg["keep_checkpoints"]:
            os.remove(os.path.join(run_dir, ck.pop(0)))

    step, t_start, losses, paused = 0, time.time(), [], False
    torch.cuda.reset_peak_memory_stats()
    while step < cfg["total_steps"]:
        free_mb = torch.cuda.mem_get_info()[0] / 1e6
        if free_mb < cfg["vram_pause_free_mb"]:
            if not paused:
                save("pause"); paused = True
                print(f"VRAM guard: {free_mb:.0f} MB free -> checkpointed, waiting", flush=True)
            time.sleep(60)
            continue
        paused = False
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        s_loss = 0.0
        for _ in range(cfg["grad_accum"]):
            b = prod.get(timeout=180)
            t_got = time.time()
            if "ev" in b:
                torch.cuda.current_stream().wait_event(b["ev"])
            t_v = time.time()
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        pixel_values=b["pixel_values"],
                        decoder_input_ids=b["dec_in"], decoder_attention_mask=b["dmask"],
                        labels=b["labels"])
            (out.loss / cfg["grad_accum"]).backward()
            torch.cuda.synchronize()
            if step < 2:
                print(f"  micro: get-wait {(t_got):.1f}s-> {(t_v - t_got)*1000:.0f}ms ev-wait, "
                      f"fwd+bwd+sync {(time.time() - t_v)*1000:.0f}ms", flush=True)
            s_loss += float(out.loss)
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg["max_grad_norm"])
        opt.step()
        step += 1
        losses.append(s_loss / cfg["grad_accum"])
        if step % cfg["log_every_steps"] == 0 or step == 1:
            rows_s = step * cfg["grad_accum"] * cfg["micro_batch"] / (time.time() - t_start)
            msg = (f"step {step}/{cfg['total_steps']} loss {losses[-1]:.4f} "
                   f"gn {float(gn):.2f} rows/s {rows_s:.1f} "
                   f"vram {torch.cuda.max_memory_allocated()/1e9:.2f}GB "
                   f"step_ms {(time.time()-t0)*1000:.0f}")
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % cfg["checkpoint_every_steps"] == 0:
            save(f"{step}")
    save("final")
    print(f"DONE {step} steps in {(time.time()-t_start)/3600:.2f} h | "
          f"loss first-10 {np.mean(losses[:10]):.4f} last-10 {np.mean(losses[-10:]):.4f}",
          flush=True)

if __name__ == "__main__":
    main()
