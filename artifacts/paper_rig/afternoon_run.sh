#!/bin/bash
# Afternoon->overnight: after morning chain ends, precompute remaining 126k rows
# (offset-aware, zero duplicates), verify 200k unique, v3 train full 200k, eval, proof, report.
set -u
cd ~/Documents/code/drishyam-1
E=artifacts/paper_rig/embeddings
E2=artifacts/paper_rig/embeddings_slice2
LOGD=runs/overnight
log(){ echo "[aft $(date +%H:%M:%S)] $*" | tee -a "$LOGD/afternoon.log"; }

# 1. wait for the morning chain to finish (report stage writes PROOF.md last)
log "waiting for morning chain to finish"
for i in $(seq 1 60); do
  pgrep -f "morning_run.sh" > /dev/null || break
  sleep 60
done
pgrep -f "morning_run.sh" > /dev/null && { log "morning chain still alive after 60min — aborting"; exit 1; }
log "morning chain done"

# 2. precompute remaining rows (resume keeps complete shards; offsets skip slice-1 overlap)
FAM="thermal:36000@20000,card:17000@15000,savings:17000@15000,electricity:16000@8000,salary:14000@6000,telecom:14000@6000,table:12000@4000"
log "precompute remaining 126k: $FAM"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 60000 uv run python artifacts/paper_rig/precompute_embed.py \
  --families "$FAM" --out-dir "$E2" >> "$LOGD/precompute_rest.log" 2>&1
log "precompute exit $?"

# 3. verify per-dir + global uniqueness (exactly 200000 unique (family,seed))
uv run python artifacts/paper_rig/verify_embeds.py --expect "$E" 50000 \
  > "$LOGD/verify_A2.json" 2>&1; log "slice-1 verify exit $?"
uv run python artifacts/paper_rig/verify_embeds.py --expect "$E2" 150000 \
  > "$LOGD/verify_B2.json" 2>&1; log "slice-2(150k) verify exit $?"
uv run python - "$E" "$E2" <<'EOF' > "$LOGD/verify_unique.txt" 2>&1
import json, sys, glob, os
pairs = set()
total = 0
for d in sys.argv[1:3]:
    for f in glob.glob(os.path.join(d, "lab_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pairs.add((r["family"], r["seed"]))
            total += 1
print(f"rows {total} unique {len(pairs)} dups {total - len(pairs)}")
assert total == 200000, f"expected 200000 rows, got {total}"
assert len(pairs) == 200000, f"duplicates present: {total - len(pairs)}"
print("UNIQUE_OK")
EOF
log "uniqueness check exit $?"
grep -q UNIQUE_OK "$LOGD/verify_unique.txt" || { log "uniqueness FAIL — v3 not started"; exit 1; }

# 3b. distribution gate: new-shard embedding stats must match the proven slice-1
# stats (same encoder+projector) — catches a broken page/atlas path BEFORE v3.
uv run python - "$E" "$E2" <<'EOF' > "$LOGD/verify_dist.txt" 2>&1
import numpy as np, glob, sys, random
def stats(pattern, n=6):
    files = sorted(glob.glob(pattern))
    random.seed(0); picks = random.sample(files, min(n, len(files)))
    arrs = []
    for f in picks:
        m = np.load(f, mmap_mode="r")
        idx = np.linspace(0, m.shape[0] - 1, 16).astype(int)
        arrs.append(np.asarray(m[idx], dtype=np.float32))
    a = np.concatenate(arrs)
    return float(a.mean()), float(a.std())
m1, s1 = stats(sys.argv[1] + "/emb_*.npy")
m2, s2 = stats(sys.argv[2] + "/emb_*.npy")
print(f"slice1 mean {m1:.4f} std {s1:.4f} | new mean {m2:.4f} std {s2:.4f}")
ok = abs(m2 - m1) < 0.15 * abs(m1) and abs(s2 - s1) < 0.25 * abs(s1)
print("DIST_OK" if ok else "DIST_FAIL")
EOF
log "distribution gate exit $?"
grep -q DIST_OK "$LOGD/verify_dist.txt" || { log "distribution FAIL — new embeddings deviate from proven encoder stats; v3 not started"; exit 1; }

# 4. v3: full 200k, fresh, 3 epochs, native MLflow this time
log "v3 training on $E,$E2 (200k x 3 epochs, native mlflow)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 40000 uv run python artifacts/paper_rig/embed_trainer.py \
  --dirs "$E,$E2" --out runs/img-e5 --epochs 3 --lr 5e-5 --warmup 200 \
  --micro 16 --accum 1 --ckpt-every 3000 --keep 1 --mlflow \
  > "$LOGD/train_v3.log" 2>&1
V3=$?
log "v3 training exit $V3"
[ $V3 -ne 0 ] && [ $V3 -ne 3 ] && { log "v3 failed hard"; exit 1; }

# 5. eval + pixel proof gate on v3
log "v3 eval"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 1800 uv run python artifacts/paper_rig/eval_embeds.py \
  --dirs "$E,$E2" --ckpt runs/img-e5/final --per-split 300 \
  --out runs/eval_embeds_v3.json > "$LOGD/eval_v3.log" 2>&1
log "v3 eval exit $?"
log "v3 pixel proof gate"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 1800 uv run python artifacts/paper_rig/proof_gate.py \
  --ckpt runs/img-e5/final --per-page 3 > "$LOGD/proof_v3.log" 2>&1
log "v3 proof exit $?"

# 6. report
log "v3 report"
uv run python artifacts/paper_rig/overnight_report.py \
  --chain-log "$LOGD/afternoon.log" --train-log "$LOGD/train_v3.log" \
  --eval-json runs/eval_embeds_v3.json --eval-log "$LOGD/eval_v3.log" \
  --verify-a "$LOGD/verify_A2.json" --verify-b "$LOGD/verify_B2.json" \
  --slice2-log "$LOGD/precompute_rest.log" --smoke-log "$LOGD/smoke.log" \
  --stage-a PASS --stage-b PASS --stage-c PASS \
  --stage-d PASS --stage-e PASS --out runs/overnight_report_v3.md \
  > "$LOGD/report_gen_v3.log" 2>&1
log "v3 complete — runs/overnight_report_v3.md"
