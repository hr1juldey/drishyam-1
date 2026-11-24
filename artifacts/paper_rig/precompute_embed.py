"""Overnight SigLIP embedding precompute for the 2-lakh manifest.
JIT pages (CPU) -> vision tower + projector (GPU, inference_mode, batch 32)
-> fp16 embeddings [N,256,640] in 10k-row shard files + QA-labels jsonl per shard.
Resumable: skips shards already complete. ~3.4 img/s measured -> 200k ~ 16h.
Pixels are still never stored; embeddings are 500x smaller than pixels."""
import os, sys, json, time, importlib, threading, queue
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

def _page_worker(rows_q, pages_q):
    """Background CPU thread: gen+degrade pages for the NEXT batch while the GPU
    renders/encodes the current one. gen_page(family, seed) is fully seeded per
    row, so threaded page production is byte-identical to the inline path."""
    while True:
        item = rows_q.get()
        try:
            if item is None:
                return
            batch_rows = item
            fam = batch_rows[0]["family"]
            pages, gts = [], []
            for r in batch_rows:
                pg, gt = gen_page(fam, r["seed"])
                pages.append(pg); gts.append(gt)
            pages_q.put((fam, pages, gts, batch_rows))
        except Exception as e:
            pages_q.put(e)
            return
        finally:
            rows_q.task_done()


def main(limit_rows=None, balanced=None, families=None, out_dir=None):
    """balanced=N -> per-family quota slice (40/30/30), NOT a prefix — a prefix
    of the family-blocked manifest would be single-family.
    families='a:8000,b:6000' -> explicit per-family quotas (slice-2 new families).
    out_dir -> separate embedding dir so a partial slice-2 can never touch slice-1."""
    global OUT
    if out_dir:
        OUT = os.path.abspath(os.path.expanduser(out_dir))
    from jit_pipeline import build_manifest
    from transformers import AutoModelForImageTextToText
    os.makedirs(OUT, exist_ok=True)
    rows, _ = build_manifest(200000)
    if families:
        want = {}
        for part in families.split(","):
            name, cnt = part.split(":")
            want[name.strip()] = int(cnt)
        byfam = {}
        for r in rows:
            byfam.setdefault(r["family"], []).append(r)
        missing = [f for f in want if f not in byfam]
        if missing:
            raise SystemExit(f"families not in manifest: {missing}; known: {sorted(byfam)}")
        rows = []
        for fam, cnt in want.items():
            rows += byfam[fam][:cnt]
        print(f"families slice: { {f: len([r for r in rows if r['family']==f]) for f in want} }",
              flush=True)
    elif balanced:
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
    rows_q = queue.Queue(maxsize=3)
    pages_q = queue.Queue(maxsize=3)
    worker = threading.Thread(target=_page_worker, args=(rows_q, pages_q), daemon=True)
    worker.start()

    def feed(chunk):
        """Enqueue same-family batches (bounded: worker stays <=3 batches ahead)."""
        i = 0
        while i < len(chunk):
            fam = chunk[i]["family"]
            j = i
            while j < len(chunk) and chunk[j]["family"] == fam and j - i < BATCH:
                j += 1
            rows_q.put(chunk[i:j])
            i = j

    while done < len(rows):
        chunk = rows[done:done + SHARD]
        emb_path = os.path.join(OUT, f"emb_{done:06d}.npy")
        lab_path = os.path.join(OUT, f"lab_{done:06d}.jsonl")
        emb_mm = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16,
                                           shape=(len(chunk), 256, 640))
        lf = open(lab_path, "w")
        feed(chunk)
        n_shard = 0
        t_shard = time.time()
        got = 0
        while got < len(chunk):
            item = pages_q.get()
            if isinstance(item, Exception):
                raise item
            fam, pages, gts, sub = item
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
            emb_mm[got:got + len(sub)] = emb.to(torch.float16).cpu().numpy()
            for r, gt in zip(sub, gts):
                qa = build_qa(fam, gt, r["seed"])[0]
                lf.write(json.dumps({"row_id": r["row_id"], "family": fam, "seed": r["seed"],
                                     "field": qa["field"], "q": qa["q"], "a": qa["a"]}) + "\n")
            got += len(sub); n_shard += len(sub); n_done += len(sub)
            if n_done % 640 < BATCH:
                rate = n_done / (time.time() - t_start)
                print(f"{n_done}/{len(rows)} rows | {rate:.1f} rows/s | "
                      f"eta {((len(rows)-n_done)/rate)/3600:.1f} h", flush=True)
        emb_mm.flush(); lf.close()
        print(f"shard {done:06d} done ({n_shard} rows) in {(time.time()-t_shard)/60:.0f} min",
              flush=True)
        done += len(chunk)
    rows_q.put(None)
    worker.join(timeout=30)
    print("PRECOMPUTE COMPLETE", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None, help="prefix limit (unbalanced)")
    ap.add_argument("--balanced", type=int, default=None, help="per-family quota slice")
    ap.add_argument("--families", type=str, default=None,
                    help="explicit quotas: 'electricity_bill:8000,telecom_invoice:6000,...'")
    ap.add_argument("--out-dir", type=str, default=None, help="embedding output dir")
    a = ap.parse_args()
    main(a.rows, a.balanced, a.families, a.out_dir)
