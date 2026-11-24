#!/usr/bin/env python3
"""Just-in-time data pipeline for 2-lakh image rows.

DISK: manifest.jsonl only — {row_id, family, seed} per row (~90 B/row, ~20 MB
for 2 lakh). NO PIXELS EVER HIT DISK. Labels are supervision derived from the
deterministic ground truth + the recorded legibility plan (thermal), with the
teacher harness calibrating a golden subset — not 2-lakh teacher calls.

TRAIN TIME: __iteration__ = recipe -> PIL page (CPU) -> degradation (numpy)
-> GPU doc-photo render (fp32) -> resize 896 -> yield batch. Pixels are used
by the training step and disposed; only the bounded CUDA working set exists.

Determinism: same seed -> same pixels + same plan (proven property of the
generators), so any row can be regenerated exactly, any time, on any run.
"""
import os, sys, json, time, importlib
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from thermal_generator import gen_thermal, apply_degradation, choose_plan    # noqa: E402
from content_generator import gen_statement, gen_card                        # noqa: E402
_gpu = importlib.import_module("2d_gpu")
_gpu.TD = torch.float32                    # text-carrying textures: fp32 only
GPURenderer = _gpu.GPURenderer

MANIFEST = os.path.join(HERE, "manifest_200k.jsonl")
BASE_SEED = 1_000_000
MIX = [("thermal", 0.40), ("savings", 0.30), ("card", 0.30)]   # kyc slots in later
AZ = ["light_left", "light_right", "light_top"]
GEOM = {"thermal": dict(ph=1000, pw=334), "savings": dict(ph=1000, pw=707),
        "card": dict(ph=1000, pw=707)}
DOC = dict(n_lo=2, n_hi=3, amp_lo=0.3, amp_hi=0.8, drift_sigma=0.6,
           cdepth_lo=1, cdepth_hi=2, field_strength=0.07, br_lo=0.94, br_hi=0.97,
           noise_sigma=1.0)
THERMAL_DECISION = ["total_amt", "date", "bill_no", "gstin",
                    "item_0_name", "item_1_name", "item_0_amt", "subtotal_amt"]

def gen_page(family, seed):
    """deterministic page + gt for one row. Returns (degraded_np, gt)."""
    rng = np.random.RandomState(seed)
    if family == "thermal":
        page, gt = gen_thermal(rng, seed)
        drng = np.random.RandomState(seed + 1)
        k = drng.randint(0, 4)                       # 0..3 degraded fields
        if k:
            fids = list(drng.choice(THERMAL_DECISION, size=k, replace=False))
            gt["degradation_plan"] = choose_plan(drng, n_ops=k, candidate_fids=list(fids))
        deg, applied = apply_degradation(page, gt, np.random.RandomState(seed + 2))
        gt["legibility_applied"] = applied
        return deg, gt
    page, gt = (gen_statement if family == "savings" else gen_card)(rng)
    return page, gt            # statements: field-registry + degradation = next work item

def build_manifest(n_rows=200_000, path=MANIFEST):
    counts = {f: int(round(n_rows * w)) for f, w in MIX}
    counts[list(counts)[-1]] = n_rows - sum(list(counts.values())[:-1])
    rows = []
    for fam, c in counts.items():
        rows += [{"row_id": len(rows) + i, "family": fam, "seed": BASE_SEED + i * 7919}
                 for i in range(c)]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows, counts

class JITTloader:
    """single-process GPU loader: yields {image (B,3,896,896) cuda, family, rows}.
    Grouped by family (manifest is family-blocked) so each render_batch is uniform."""

    def __init__(self, rows, batch_size=32):
        self.rows = rows
        self.bs = batch_size
        self.renders = {}

    def _renderer(self, fam):
        if fam not in self.renders:
            g = GEOM[fam]
            self.renders[fam] = GPURenderer(th=1080, tw=1080, atlas_pages=[
                np.full((g["ph"], g["pw"], 3), 250.0)], **g)
        return self.renders[fam]

    def __iter__(self):
        class G:
            def seed_drawn(self, a, b): return int(np.random.randint(a, b))
        g = G()
        for i in range(0, len(self.rows), self.bs):
            chunk = self.rows[i:i + self.bs]
            fam = chunk[0]["family"]
            t0 = time.time()
            pages, gts = [], []
            for r in chunk:
                pg, gt = gen_page(fam, r["seed"])
                gt["row_id"] = r["row_id"]
                pages.append(pg); gts.append(gt)
            t_cpu = time.time() - t0
            t0 = time.time()
            rend = self._renderer(fam)
            rend.set_atlas(pages)
            az = AZ[chunk[0]["seed"] % 3]
            out, _a = rend.render_batch(len(chunk), az, g, content_idx=[0] * len(chunk), **DOC)
            img896 = F.interpolate(out, size=(896, 896), mode="bilinear", antialias=True)
            torch.cuda.synchronize()
            t_gpu = time.time() - t0
            yield {"image": img896, "family": fam, "rows": gts,
                   "t_cpu": t_cpu, "t_gpu": t_gpu}

if __name__ == "__main__":
    N = int(os.environ.get("N_ROWS", "200000"))
    rows, counts = build_manifest(N)
    size_mb = os.path.getsize(MANIFEST) / 1e6
    print(f"manifest: {len(rows)} rows, {size_mb:.1f} MB on disk, mix {counts}")
    print(f"pixels on disk: 0 bytes (JIT)")
    loader = JITTloader(rows[:64 * 12], batch_size=32)
    n_done, t_all, t_cpu_all, t_gpu_all = 0, 0.0, 0.0, 0.0
    sheet = []
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for bi, batch in enumerate(loader):
        n_done += batch["image"].shape[0]
        t_all += batch["t_cpu"] + batch["t_gpu"]
        t_cpu_all += batch["t_cpu"]; t_gpu_all += batch["t_gpu"]
        if len(sheet) < 8:
            sheet.append(batch["image"][0].detach().cpu())
        if bi == 11:
            break
    wall = time.time() - t_start
    rps = n_done / wall
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"streamed {n_done} rows in {wall:.1f}s wall -> {rps:.1f} rows/s "
          f"(cpu gen {t_cpu_all/n_done*1000:.1f} ms/row, gpu render {t_gpu_all/n_done*1000:.1f} ms/row)")
    print(f"epoch wall for {N} rows @ this rate: {N/rps/3600:.2f} h; "
          f"VRAM peak {peak:.2f} GB; disk: manifest only")
    grid = torch.cat([torch.cat(sheet[:4], dim=2), torch.cat(sheet[4:8], dim=2)], dim=1)
    from PIL import Image
    arr = np.clip(grid.permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(f"{HERE}/jit_contact_sheet.png")
    print("contact sheet saved: jit_contact_sheet.png (8 rows @ 896x896)")
