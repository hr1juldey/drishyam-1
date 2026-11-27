#!/usr/bin/env bash
# GATE-2.5 V4 OVERNIGHT CHAIN — armed 2026-11-26 ~15:30, executes AFTER the GPU frees.
# Order: wait GPU free -> gate checks -> build v4 twin registry/manifest -> embed ->
# labels v4 (balanced) -> v4 train -> v4 eval. Every step logs; a failed gate ABORTS
# before any GPU work. Contract: ~/.hermes/plans/2026-11-25_144900-gate2-vision-decision.md
set -uo pipefail
cd ~/Documents/code/drishyam-1
LOG=runs/overnight/v4_chain.log
echo "[v4 $(date '+%H:%M:%S')] chain armed; waiting for GPU to free (text-train + jevbench eval)" | tee -a "$LOG"

# 1) wait until no train_text_row / eval_jevbench processes remain
while pgrep -f "drishyam\.(train_text_row|eval_jevbench)" > /dev/null; do sleep 120; done
echo "[v4 $(date '+%H:%M:%S')] GPU free" | tee -a "$LOG"

# 2) gate: subagent's K5-v4 commit must exist (factory omit guards for 4 families)
# NOTE: no `grep -q` in a pipefail pipeline — early-exit grep SIGPIPEs the producer
# and fakes a failure (cost: one false ABORT at 06:15 on 2026-11-27).
GIT_RECENT="$(git log --oneline -40)"
case "$GIT_RECENT" in *K5-v4*) ;; *) echo "[v4 ABORT] K5-v4 commit not found — subagent work not landed" | tee -a "$LOG"; exit 1;; esac
uv run python -c "
import sys; sys.path.insert(0, 'artifacts/paper_rig')
import fin_docs, content_generator
fams = {'electricity': fin_docs, 'telecom': fin_docs, 'savings': content_generator, 'card': content_generator}
for fam, mod in fams.items():
    sets = getattr(mod, 'OMITTABLE_FIELDS', {})
    assert fam in sets and len(sets[fam]) >= 1, f'{fam}: omittable set missing'
print('gate: omittable sets OK for all 4 families')
" || { echo "[v4 ABORT] omittable-field sets missing" | tee -a "$LOG"; exit 1; }

# 3) build v4 registry (500 twins/family, field rotating) + append manifest
python3 - <<'EOF'
import sys, json, random
sys.path.insert(0, "artifacts/paper_rig")
import fin_docs, content_generator
SETS = {**fin_docs.OMITTABLE_FIELDS, **getattr(content_generator, "OMITTABLE_FIELDS", {})}
OFF = {"electricity": 501_000, "telecom": 502_000, "savings": 503_000, "card": 504_000}
reg = json.load(open("artifacts/paper_rig/k5_omissions.json"))
# idempotent: never append manifest rows that already exist (rerun-safe)
try:
    have = {json.loads(l)["row_id"] for l in open("artifacts/paper_rig/k5_twin_manifest.jsonl")}
except FileNotFoundError:
    have = set()
new_rows = []
for fam, off in OFF.items():
    fields = sorted(SETS[fam])
    for i in range(500):
        s = off + 1 + i
        reg.setdefault(fam, {})[str(s)] = [fields[i % len(fields)]]
        rid = f"k5twin_{fam}_{s}"
        if rid not in have:
            new_rows.append({"row_id": rid, "family": fam, "seed": s})
json.dump(reg, open("artifacts/paper_rig/k5_omissions.json", "w"))
with open("artifacts/paper_rig/k5_twin_manifest.jsonl", "a") as mf:
    for r in new_rows:
        mf.write(json.dumps(r) + "\n")
print(f"registry: +{len(new_rows)} twins across 4 train families")
EOF
[ $? -ne 0 ] && { echo "[v4 ABORT] registry build failed" | tee -a "$LOG"; exit 1; }

# 4) embed all twins (~2.6k pages, ~15 min)
echo "[v4 $(date '+%H:%M:%S')] embedding twins" | tee -a "$LOG"
# thread caps + --sync + expandable segments: async pipeline wedged at 600s;
# sync path then OOM'd on 3GB allocator fragmentation without it
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 uv run python artifacts/paper_rig/precompute_embed.py \
  --manifest artifacts/paper_rig/k5_twin_manifest.jsonl \
  --out-dir artifacts/paper_rig/embeddings_k5 --sync 2>&1 | tail -4 | tee -a "$LOG"
test "${PIPESTATUS[0]}" -eq 0 || { echo "[v4 ABORT] embed failed (PIPESTATUS ${PIPESTATUS[0]})" | tee -a "$LOG"; exit 1; }

# 5) labels v4 (balanced presence)
echo "[v4 $(date '+%H:%M:%S')] deriving labels v4 (balanced)" | tee -a "$LOG"
uv run python -m drishyam.vision_data --unseen-pages 1200 --balance-presence --mlflow \
  2>&1 | grep -E "twins|dropped|WROTE" | tee -a "$LOG"

# 6) v4 train (fresh, proven hyperparams)
echo "[v4 $(date '+%H:%M:%S')] v4 retrain starting" | tee -a "$LOG"
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m drishyam.train_gate2_row --lr 2e-5 --epochs 4 --rows-per-batch 16 \
  --out runs/g2-row-sft-v4 2>&1 | tee runs/overnight/g2_row_v4_sft.log

# 7) v4 eval
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m drishyam.eval_gate2 --ckpt runs/g2-row-sft-v4/best.pt --path row \
  --out artifacts/gate2_row_table_v4.json 2>&1 | tee runs/overnight/g2_row_v4_eval.log
echo "[v4 $(date '+%H:%M:%S')] CHAIN COMPLETE — verdict in artifacts/gate2_row_table_v4.json" | tee -a "$LOG"
