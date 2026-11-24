"""Overnight SigLIP embedding precompute for the 2-lakh manifest.
JIT pages (CPU) -> vision tower + projector (GPU, inference_mode, batch 32)
-> fp16 embeddings [N,256,640] in 10k-row shard files + QA-labels jsonl per shard.
Resumable: skips shards already complete. ~3.4 img/s measured -> 200k ~ 16h.
Pixels are still never stored; embeddings are 500x smaller than pixels."""
import os, sys, json, time, importlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
_gpu = importlib.import_module("2d_gpu")
_gpu.TD = torch.float32
from jit_pipeline import gen_page, GEOM, DOC, AZ                  # noqa: E402
from row_labels import build_qa                                   # noqa: E402
from pack_batch import get_processor, _SNAP                       # noqa: E402
import importlib as _il                                           # noqa: E402
GPURenderer = _il.import_module("2d_gpu").GPURenderer

OUT = os.path.join(HERE, "embeddings")
SHARD = 10000
BATCH = 32

def main(limit_rows=None, balanced=None):
    """balanced=N -> per-family quota slice (40/30/30), NOT a prefix — a prefix
    of the family-blocked manifest would be single-family."""
    from jit_pipeline import build_manifest
    from transformers import AutoModelForImageTextToText
    os.makedirs(OUT, exist_ok=True)
    rows, _ = build_manifest(200000)
    if balanced:
        q = {"thermal": int(balanced * 0.4), "savings": int(balanced * 0.3),
             "card": balanced - int(balanced * 0.4) - int(balanced * 0.3)}
        byfam = {}
        for r in rows:
            byfam.setdefault(r["family"], []).append(r)
        rows = []
        for fam in ("thermal", "savings", "card"):
            rows += byfam[fam][:q[fam]]
        print(f"balanced slice: { {f: len([r for r in rows if r['family']==f]) for f in q} }",
              flush=True)
    elif limit_rows:
        rows = rows[:limit_rows]
    proc = get_processor()
    model = AutoModelForImageTextToText.from_pretrained(_SNAP, dtype=torch.bfloat16).to("cuda").eval()
    rends = {}
    def rend(fam):
        if fam not in rends:
            g = GEOM[fam]
            rends[fam] = GPURenderer(th=896, tw=896, atlas_pages=[
                np.full((g["ph"], g["pw"], 3), 250.0)], **g)
        return rends[fam]
    class G:
        def seed_drawn(self, a, b): return 1

    # resumable: count completed shard rows
    done = 0
    shard_files = sorted(f for f in os.listdir(OUT) if f.startswith("emb_") and f.endswith(".npy"))
    if shard_files:
        last = int(shard_files[-1].split("_")[1].split(".")[0])
        done = last + SHARD
    print(f"manifest {len(rows)} rows, resuming at {done}", flush=True)

    t_start = time.time()
    n_done = 0
    while done < len(rows):
        chunk = rows[done:done + SHARD]
        emb_path = os.path.join(OUT, f"emb_{done:06d}.npy")
        lab_path = os.path.join(OUT, f"lab_{done:06d}.jsonl")
        emb_mm = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16,
                                           shape=(len(chunk), 256, 640))
        lf = open(lab_path, "w")
        n_shard = 0
        t_shard = time.time()
        i = 0
        while i < len(chunk):
            sub = chunk[i:i + BATCH]
            fam = sub[0]["family"]
            pages, gts = [], []
            for r in sub:
                pg, gt = gen_page(fam, r["seed"])
                pages.append(pg); gts.append(gt)
            rend(fam).set_atlas(pages)
            with torch.inference_mode():
                rend(fam)
                class GG:
                    def seed_drawn(self, a, b): return 1
                out, _a = rend(fam).render_batch(len(sub), AZ[sub[0]["seed"] % 3], GG(),
                                                 content_idx=[0] * len(sub), **DOC)
                px = torch.nn.functional.interpolate(out, size=(896, 896), mode="bilinear",
                                                     antialias=True).to(torch.bfloat16)
                hid = model.model.encoder.vision_tower(px).last_hidden_state
                emb = model.model.encoder.multi_modal_projector(hid)
            emb_mm[i:i + len(sub)] = emb.to(torch.float16).cpu().numpy()
            for r, gt in zip(sub, gts):
                qa = build_qa(fam, gt, r["seed"])[0]
                lf.write(json.dumps({"row_id": r["row_id"], "family": fam, "seed": r["seed"],
                                     "field": qa["field"], "q": qa["q"], "a": qa["a"]}) + "\n")
            i += len(sub); n_shard += len(sub); n_done += len(sub)
            if n_done % 640 < BATCH:
                rate = n_done / (time.time() - t_start)
                print(f"{n_done}/{len(rows)} rows | {rate:.1f} rows/s | "
                      f"eta {((len(rows)-n_done)/rate)/3600:.1f} h", flush=True)
        emb_mm.flush(); lf.close()
        print(f"shard {done:06d} done ({n_shard} rows) in {(time.time()-t_shard)/60:.0f} min",
              flush=True)
        done += len(chunk)
    print("PRECOMPUTE COMPLETE", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None, help="prefix limit (unbalanced)")
    ap.add_argument("--balanced", type=int, default=None, help="per-family quota slice")
    a = ap.parse_args()
    main(a.rows, a.balanced)
