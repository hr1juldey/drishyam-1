"""Decoder-only training on CACHED embeddings (precompute_embed.py output).
Skips SigLIP entirely: splices cached [256,640] projected embeddings into the
encoder's inputs_embeds at the image positions, then the top-level forward runs
encoder + decoder + loss as usual. Pixel path never executed.

FREEZE LAW: everything frozen except the decoder stack (NOT embed_tokens, NOT
lm_head, NOT projector). The projector must NEVER train after precompute —
cached embeddings are its frozen output; changing the projector would silently
invalidate every shard on disk.

Packing parity with pack_batch.py/producer.py:
  encoder order = [256 image tokens][question]  -> combined embeds same order
  labels = answer_ids + eos; HF shifts decoder_input_ids internally.

--smoke: 10 steps on a tiny slice, asserts finite loss, exit 3 on failure.
"""
import os, sys, json, time, math, glob, random, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--t5gemma-2-270m-270m/"
    "snapshots/7c38f16641f455ef0685b18431faf1b17722d5a1")
IMG_TOKENS = 256


def load_entries(dirs, exclude_holdout=True):
    """[(emb_path, {row_id,family,seed,q,a})] for every row across shard dirs."""
    ents = []
    for d in dirs:
        for emb_path in sorted(glob.glob(os.path.join(d, "emb_*.npy"))):
            lab_path = os.path.join(
                os.path.dirname(emb_path),
                os.path.basename(emb_path).replace("emb_", "lab_", 1).replace(".npy", ".jsonl"))
            if not os.path.exists(lab_path):
                continue
            rows = []
            with open(lab_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            n_emb = np.load(emb_path, mmap_mode="r").shape[0]  # filename = start row, NOT count
            if len(rows) != n_emb:
                print(f"SKIP {emb_path}: {len(rows)} labels vs {n_emb} embeds", flush=True)
                continue
            for i, r in enumerate(rows):
                if exclude_holdout and (r["seed"] % 10 == 7):
                    continue
                ents.append((emb_path, i, r))
    return ents


def build_model_and_freeze(cpu=False, init_from=None):
    from transformers import T5Gemma2ForConditionalGeneration, AutoTokenizer
    src = init_from if init_from else SNAP
    tok = AutoTokenizer.from_pretrained(src)
    model = T5Gemma2ForConditionalGeneration.from_pretrained(
        src, dtype=torch.bfloat16)
    if not cpu:
        model = model.cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    # unfreeze the decoder stack, keep its embedding + LM head frozen (shared/tied)
    n_unfrozen = 0
    for name, p in model.model.decoder.named_parameters():
        if name.startswith("embed_tokens"):
            continue
        p.requires_grad_(True)
        n_unfrozen += p.numel()
    if n_unfrozen == 0:
        raise RuntimeError("freeze law broke: decoder has no trainable params")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if any("projector" in n or "vision" in n or "encoder" in n or "lm_head" in n
           for n in trainable):
        raise RuntimeError("freeze law broke: non-decoder param is trainable")
    print(f"trainable {n_unfrozen/1e6:.1f}M (decoder only; projector+SigLIP+embed frozen)",
          flush=True)
    return model, tok


def collate(batch, tok, device):
    """batch: list of (emb_row np[256,640] fp16, row dict) -> tensors."""
    embs = torch.from_numpy(np.stack([b[0] for b in batch]))            # [B,256,640] fp16
    embs = embs.to(device=device, dtype=torch.bfloat16)
    qs = [b[1]["q"] for b in batch]
    ans = [b[1]["a"] for b in batch]
    qe = tok(qs, return_tensors="pt", padding=True, add_special_tokens=False)
    q_ids = qe["input_ids"].to(device)                                  # [B,Lq]
    q_mask = qe["attention_mask"].to(device)                            # [B,Lq]
    text_embeds = model_embeds_lookup(model, q_ids)                     # [B,Lq,D]
    # pad question positions -> zero embeds, masked
    pad_mask = q_mask.unsqueeze(-1).to(text_embeds.dtype)
    text_embeds = text_embeds * pad_mask
    enc_embeds = torch.cat([embs, text_embeds], dim=1)                  # [B,256+Lq,D]
    img_ones = torch.ones(q_ids.shape[0], IMG_TOKENS, dtype=torch.long,
                          device=q_ids.device)
    attn = torch.cat([img_ones, q_mask], dim=1)                         # [B,256+Lq]
    ans_ids = tok(ans, return_tensors="pt", padding=True,
                  add_special_tokens=False)["input_ids"].to(device)     # [B,La]
    ans_mask = tok(ans, return_tensors="pt", padding=True,
                   add_special_tokens=False)["attention_mask"].to(device)
    eos = torch.full((ans_ids.shape[0], 1),
                     tok.eos_token_id if tok.eos_token_id is not None else 1,
                     dtype=torch.long, device=device)
    ans_ids = torch.cat([ans_ids, eos], dim=1)
    ans_mask = torch.cat([ans_mask, torch.ones_like(ans_mask[:, :1])], dim=1)
    labels = ans_ids.masked_fill(ans_mask == 0, -100)
    return enc_embeds, attn, labels


_M_EMB = {}


def model_embeds_lookup(model, ids):
    return model.get_input_embeddings()(ids)


def main():
    global model
    # --- MLflow optional live tracking (never fatal: training runs without it)
    _mf = {"m": None}

    def ml_init(total_steps):
        if not a.mlflow:
            return
        try:
            import mlflow
            mlflow.set_tracking_uri("sqlite:///" + os.path.join(os.getcwd(), "runs", "mlflow.db"))
            mlflow.set_experiment("drishyam-doc")
            mlflow.start_run(run_name=os.path.basename(a.out.rstrip("/")))
            mlflow.log_params({k: getattr(a, k) for k in
                               ["epochs", "micro", "accum", "lr", "warmup", "seed"]})
            mlflow.log_param("total_steps", total_steps)
            _mf["m"] = mlflow
        except Exception as e:
            print(f"mlflow disabled: {e}", flush=True)

    def ml_log(step, **metrics):
        if _mf["m"] is None:
            return
        try:
            _mf["m"].log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=str, required=True, help="comma list of embedding dirs")
    ap.add_argument("--out", type=str, required=True, help="checkpoint dir (runs/...)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1500)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--vram-pause-free-mb", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init-from", type=str, default=None,
                    help="warm-start decoder from this checkpoint instead of SNAP")
    ap.add_argument("--cpu", action="store_true", help="wiring smoke on CPU")
    ap.add_argument("--smoke", action="store_true", help="10-step wiring smoke on 200 rows")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--mlflow", action="store_true",
                    help="log step metrics to MLflow file-store under runs/mlruns")
    a = ap.parse_args()
    dirs = [d.strip() for d in a.dirs.split(",") if d.strip()]
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    ents = load_entries(dirs)
    print(f"rows {len(ents)} from {dirs} (holdout excluded)", flush=True)
    if a.smoke:
        ents = ents[:200]
    if len(ents) < 32:
        raise SystemExit(f"too few trainable rows: {len(ents)}")

    model, tok = build_model_and_freeze(cpu=a.cpu, init_from=a.init_from)
    dev = "cpu" if a.cpu else "cuda"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.01)

    emb_cache = {}
    def shard_np(path):
        if path not in emb_cache:
            if len(emb_cache) > 6:
                emb_cache.clear()
            emb_cache[path] = np.load(path, mmap_mode="r")
        return emb_cache[path]

    steps_per_epoch = math.ceil(len(ents) / (a.micro * a.accum))
    total_steps = steps_per_epoch * a.epochs if not a.smoke else 10
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    print(f"steps {total_steps} | micro {a.micro} x accum {a.accum}", flush=True)

    step = 0
    t0 = time.time()
    ml_init(total_steps)
    rows_seen = 0
    loss0 = None
    s_loss = 0.0
    ckpts = []
    for epoch in range(a.epochs):
        order = list(range(len(ents)))
        random.Random(a.seed * 1000 + epoch).shuffle(order)
        micro_idx = 0
        for bi in range(0, len(order), a.micro):
            if step >= total_steps:
                break
            idxs = order[bi:bi + a.micro]
            if len(idxs) < 4:
                continue
            batch = [(np.asarray(shard_np(ents[i][0])[ents[i][1]]), ents[i][2])
                     for i in idxs]
            enc_embeds, attn, labels = collate(batch, tok, dev)
            # explicit encoder call: the top-level inputs_embeds path mis-routes
            # the decoder cross-attention mask (262 vs decoder-len mismatch);
            # with encoder_outputs given, the wrapper maps attention_mask to
            # encoder_attention_mask correctly (verified in modeling_t5gemma2.py)
            enc = model.model.encoder.text_model(inputs_embeds=enc_embeds,
                                                 attention_mask=attn)
            out = model(encoder_outputs=enc, attention_mask=attn, labels=labels)
            loss = out.loss / a.accum
            loss.backward()
            lv = float(loss.detach())
            if not math.isfinite(lv):
                print(f"NaN/inf loss at step {step} — halting, last ckpt kept",
                      flush=True)
                save_ckpt(model, tok, a.out, step, ckpts, a.keep, final=True)
                sys.exit(3)
            s_loss += lv
            rows_seen += len(idxs)
            micro_idx += 1
            if micro_idx % a.accum == 0:
                if step < a.warmup:
                    lr = a.lr * (step + 1) / a.warmup
                    for g in opt.param_groups:
                        g["lr"] = lr
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if loss0 is None:
                    loss0 = lv * a.accum
                if step % a.log_every == 0:
                    rps = rows_seen / (time.time() - t0)
                    if not a.cpu:
                        vram = torch.cuda.memory_allocated() / 2**20
                    else:
                        vram = 0
                    print(f"step {step}/{total_steps} loss {s_loss/(a.accum*a.log_every):.4f} "
                          f"gn {float(gn):.2f} rows/s {rps:.1f} vram {vram:.0f}MB", flush=True)
                    loss_avg = s_loss / (a.accum * a.log_every)
                    s_loss = 0.0
                    ml_log(step, loss=loss_avg, grad_norm=float(gn),
                           rows_per_s=rps, vram_mb=vram)
                    # divergence guard: >3x the first logged avg after warmup -> halt
                    if step > a.warmup and loss_avg > 3.0 * loss0:
                        print(f"divergence: loss {loss_avg:.4f} > 3x initial {loss0:.4f} "
                              f"at step {step} — halting, last ckpt kept", flush=True)
                        save_ckpt(model, tok, a.out, step, ckpts, a.keep, final=True)
                        sys.exit(3)
                    if not a.cpu:
                        free = torch.cuda.mem_get_info()[0] / 2**20
                        if free < a.vram_pause_free_mb:
                            torch.cuda.empty_cache()
                            time.sleep(5)
                if a.ckpt_every and step % a.ckpt_every == 0:
                    save_ckpt(model, tok, a.out, step, ckpts, a.keep)
        if step >= total_steps:
            break

    save_ckpt(model, tok, a.out, step, ckpts, a.keep, final=True)
    dt = time.time() - t0
    print(f"DONE steps {step} rows {rows_seen} in {dt/60:.0f} min "
          f"({rows_seen/dt:.1f} rows/s)", flush=True)
    ml_log(step, train_minutes=dt / 60, final_rows_per_s=rows_seen / dt)
    if _mf["m"] is not None:
        try:
            _mf["m"].end_run(status="FINISHED")
        except Exception:
            pass
    if a.smoke:
        assert step >= 5, f"smoke ran only {step} steps"
        print("SMOKE OK", flush=True)


def save_ckpt(model, tok, out, step, ckpts, keep, final=False):
    d = os.path.join(out, "final" if final else f"ckpt_{step:06d}")
    if os.path.exists(d):
        return d
    model.save_pretrained(d, safe_serialization=True)
    tok.save_pretrained(d)
    try:  # full processor (image preprocessor config) so ckpts are self-contained
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(SNAP).save_pretrained(d)
    except Exception as e:
        print(f"processor save skipped: {e}", flush=True)
    ckpts.append(d)
    while len(ckpts) > keep:
        old = ckpts.pop(0)
        import shutil
        if os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
    print(f"checkpoint saved: {d}", flush=True)
    return d


if __name__ == "__main__":
    main()
