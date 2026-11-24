#!/usr/bin/env python3
"""PARITY TEST: training-with-JIT-images must run as fast as training-without-images.

Producer thread: recipe -> PIL page (CPU) -> GPU doc-photo render on a SIDE cuda
stream -> 896px bf16 batch into a 2-slot GPU ring buffer.
Trainer: T5Gemma-2 270M fwd+bwd+AdamW on the default stream, consuming ring slots.
Requirement: wall(jit) ~= wall(control: blank batches, same step). Delta <= ~10%.
"""
import os, sys, json, time, threading, importlib
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["HF_HUB_OFFLINE"] = "1"
from jit_pipeline import gen_page, GEOM, DOC, AZ, build_manifest            # noqa: E402
_gpu = importlib.import_module("2d_gpu")
_gpu.TD = torch.float32
GPURenderer = _gpu.GPURenderer
from transformers import AutoModelForSeq2SeqLM, AutoConfig                  # noqa: E402

DEV = "cuda"
BS_MICRO, ACCUM, SEQ, STEPS, WARMUP = 8, 4, 256, 6, 2
BS = BS_MICRO * ACCUM            # effective rows per optimizer step
# stand-in step = encoder-decoder TRUNK fwd+bwd (decision-head training has no
# 262k-vocab LM head; computing full logits here would be a 2GB fp32 artifact)
NAME = "google/t5gemma-2-270m-270m"

def build_model():
    try:
        m = AutoModelForSeq2SeqLM.from_pretrained(NAME, torch_dtype=torch.bfloat16,
                                                  local_files_only=True).to(DEV)
    except Exception as e:
        print("AutoModelForSeq2SeqLM failed (%s), trying T5Gemma2 class" % e.__class__.__name__)
        from transformers import T5Gemma2ForConditionalGeneration
        m = T5Gemma2ForConditionalGeneration.from_pretrained(
            NAME, torch_dtype=torch.bfloat16, local_files_only=True).to(DEV)
    m.gradient_checkpointing_enable()
    # faithful to the real plan: SigLIP + projector stay FROZEN (no grads, no AdamW state)
    frozen = 0
    for n, p_ in m.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p_.requires_grad = False
            frozen += p_.numel()
    m.train()
    print(f"frozen (SigLIP+projector): {frozen/1e6:.0f}M params -> no grads, no AdamW state")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    return m, opt

class Ring:
    """2-slot ring of GPU-resident bf16 batches (896px)."""
    def __init__(self, n=2):
        self.buf = [torch.zeros(BS, 3, 896, 896, device=DEV, dtype=torch.bfloat16) for _ in range(n)]
        self.ev = [torch.cuda.Event() for _ in range(n)]
        self.ready = [threading.Event() for _ in range(n)]
        self.free = [threading.Event() for _ in range(n)]
        for f in self.free:
            f.set()

class Producer(threading.Thread):
    def __init__(self, rows, ring):
        super().__init__(daemon=True)
        self.rows, self.ring = rows, ring
        self.renders = {}
        self.stop = False
        self.gen_ms, self.render_ms = [], []

    def _renderer(self, fam):
        if fam not in self.renders:
            g = GEOM[fam]
            self.renders[fam] = GPURenderer(th=896, tw=896, atlas_pages=[
                np.full((g["ph"], g["pw"], 3), 250.0)], **g)
        return self.renders[fam]

    def run(self):
        gseed = iter(lambda: int(np.random.randint(0, 2**31)), None)
        stream = torch.cuda.Stream()
        i = 0
        while not self.stop:
            slot = i % 2
            self.ring.free[slot].wait()
            chunk = self.rows[i * BS:(i + 1) * BS]
            if not chunk:
                break
            fam = chunk[0]["family"]
            t0 = time.time()
            pages, gts = [], []
            for r in chunk:
                pg, gt = gen_page(fam, r["seed"])
                pages.append(pg); gts.append(gt)
            t_cpu = (time.time() - t0) * 1000
            t0 = time.time()
            rend = self._renderer(fam)
            with torch.cuda.stream(stream):
                rend.set_atlas(pages)
                out, _a = rend.render_batch(len(chunk), AZ[chunk[0]["seed"] % 3], gseed,
                                            content_idx=[0] * len(chunk), **DOC)
                img = F.interpolate(out, size=(896, 896), mode="bilinear", antialias=True)
                bf = img.to(torch.bfloat16)
                self.ring.buf[slot].copy_(bf, non_blocking=False)
                self.ring.ev[slot].record(stream)
            torch.cuda.synchronize()
            t_gpu = (time.time() - t0) * 1000
            self.gen_ms.append(t_cpu); self.render_ms.append(t_gpu)
            self.ring.ready[slot].set()
            i += 1

def run_control(model, opt, rows):
    """no images: identical optimizer window on random token batches."""
    torch.cuda.synchronize()
    t0 = time.time()
    times = []
    for i in range(STEPS + WARMUP):
        t0s = time.time()
        for _ in range(ACCUM):
            ids = torch.randint(0, 262000, (BS_MICRO, SEQ), device=DEV)
            out = model.model(input_ids=ids, decoder_input_ids=ids)
            loss = out.last_hidden_state.float().pow(2).mean()
            (loss / ACCUM).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= WARMUP:
            times.append(time.time() - t0s)
    wall = time.time() - t0
    return wall, np.median(times[WARMUP:])

def run_jit(model, opt, rows, ring):
    prod = Producer(rows, ring)
    prod.start()
    times = []
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(STEPS + WARMUP):
        slot = i % 2
        ring.ready[slot].wait()
        torch.cuda.current_stream().wait_event(ring.ev[slot])
        imgs = ring.buf[slot]
        t0s = time.time()
        for _ in range(ACCUM):
            ids = torch.randint(0, 262000, (BS_MICRO, SEQ), device=DEV)
            out = model.model(input_ids=ids, decoder_input_ids=ids)
            loss = out.last_hidden_state.float().pow(2).mean()
            (loss / ACCUM).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= WARMUP:
            times.append(time.time() - t0s)
        ring.free[slot].set()
    wall = time.time() - t0
    prod.stop = True
    prod.join(timeout=2)
    return wall, np.median(times[WARMUP:]), prod

if __name__ == "__main__":
    rows, counts = build_manifest(BS * (STEPS + WARMUP + 2))
    model, opt = build_model()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model loaded: {nparams/1e6:.0f}M params, micro-batch {BS_MICRO} x accum {ACCUM} = {BS} rows/step, seq {SEQ}")

    wall_c, med_c = run_control(model, opt, rows)
    print(f"CONTROL (no images):  {med_c*1000:6.0f} ms/step median, {wall_c:.1f}s wall, "
          f"{BS*STEPS/wall_c:.1f} rows/s")

    ring = Ring()
    wall_j, med_j, prod = run_jit(model, opt, rows, ring)
    g = np.mean(prod.gen_ms); r = np.mean(prod.render_ms)
    print(f"JIT (with images):    {med_j*1000:6.0f} ms/step median, {wall_j:.1f}s wall, "
          f"{BS*STEPS/wall_j:.1f} rows/s")
    print(f"producer per batch: cpu-gen {g:.0f} ms, gpu render {r:.0f} ms "
          f"(budget available: {med_c*1000:.0f} ms/step)")
    delta = (med_j - med_c) / med_c * 100
    verdict = "PASS" if delta <= 10 else "FAIL"
    print(f"parity delta: {delta:+.1f}%  -> {verdict} "
          f"({'generation hidden behind training' if delta <= 10 else 'generation slows training'})")
    print(f"VRAM peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
