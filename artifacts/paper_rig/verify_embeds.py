"""Verify embedding shard dirs are complete and finite. Exit 1 on any failure."""
import os, sys, json, glob
import numpy as np

def check_dir(d, expect_rows=None):
    embs = sorted(glob.glob(os.path.join(d, "emb_*.npy")))
    labs = sorted(glob.glob(os.path.join(d, "lab_*.jsonl")))
    if not embs:
        return {"dir": d, "ok": False, "why": "no shards"}
    if len(embs) != len(labs):
        return {"dir": d, "ok": False, "why": f"{len(embs)} emb vs {len(labs)} lab files"}
    total = 0
    for e in embs:
        i = int(e.split("_")[-1].split(".")[0])
        l = os.path.join(os.path.dirname(e),
                         os.path.basename(e).replace("emb_", "lab_", 1).replace(".npy", ".jsonl"))
        n = sum(1 for line in open(l) if line.strip())
        shape = np.load(e, mmap_mode="r").shape
        if shape[0] != n:
            return {"dir": d, "ok": False, "why": f"shard {i}: {shape[0]} embeds vs {n} labels"}
        mm = np.load(e, mmap_mode="r")
        for probe in (0, shape[0] // 2, shape[0] - 1):
            row = np.asarray(mm[probe], dtype=np.float32)
            if not np.isfinite(row).all():
                return {"dir": d, "ok": False, "why": f"shard {i} row {probe}: NaN/inf"}
        total += n
    if expect_rows is not None and total != expect_rows:
        return {"dir": d, "ok": False, "why": f"{total} rows, expected {expect_rows}"}
    return {"dir": d, "ok": True, "rows": total, "shards": len(embs)}

if __name__ == "__main__":
    out = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--expect":
            out.append(check_dir(args[i + 1], int(args[i + 2])))
            i += 3
        else:
            out.append(check_dir(args[i]))
            i += 1
    ok = all(r["ok"] for r in out)
    print(json.dumps(out, indent=2))
    sys.exit(0 if ok else 1)
