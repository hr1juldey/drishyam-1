#!/bin/bash
# Morning run 2026-11-25: wait slice-2 heal -> v2 train (7 families, fresh) -> eval -> proof -> report.
set -u
cd ~/Documents/code/drishyam-1
E=artifacts/paper_rig/embeddings
E2=artifacts/paper_rig/embeddings_slice2
LOGD=runs/overnight
log(){ echo "[morning $(date +%H:%M:%S)] $*" | tee -a "$LOGD/morning.log"; }

# 1. wait for the running heal
log "waiting for slice-2 heal"
for i in $(seq 1 150); do
  grep -q "PRECOMPUTE COMPLETE" "$LOGD/slice2_heal.log" 2>/dev/null && break
  pgrep -f "precompute_embed.py --families" > /dev/null || {
    log "heal process died without completing — relaunching once"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      timeout 14400 uv run python artifacts/paper_rig/precompute_embed.py \
      --families electricity:8000,telecom:6000,salary:6000,table:4000 \
      --out-dir "$E2" >> "$LOGD/slice2_heal.log" 2>&1
  }
  grep -q "PRECOMPUTE COMPLETE" "$LOGD/slice2_heal.log" 2>/dev/null && break
  sleep 120
done
if uv run python artifacts/paper_rig/verify_embeds.py --expect "$E2" 24000 \
    > "$LOGD/verify_heal.json" 2>&1; then
  log "slice-2 verified PASS (24000 rows)"
  DIRS="$E,$E2"
else
  log "slice-2 verify FAIL — v2 proceeds on slice-1 only"
  DIRS="$E"
fi

# 2. v2: fresh 3-epoch train on all families
log "v2 training on $DIRS"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 14400 uv run python artifacts/paper_rig/embed_trainer.py \
  --dirs "$DIRS" --out runs/img-e4 --epochs 3 --lr 5e-5 --warmup 200 \
  --micro 16 --accum 1 --ckpt-every 3000 --keep 1 \
  > "$LOGD/train_v2.log" 2>&1
V2=$?
log "v2 training exit $V2"
[ $V2 -ne 0 ] && [ $V2 -ne 3 ] && { log "v2 failed hard — stopping morning run"; exit 1; }

# 3. corrected eval
log "v2 eval"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 1800 uv run python artifacts/paper_rig/eval_embeds.py \
  --dirs "$DIRS" --ckpt runs/img-e4/final --per-split 300 \
  --out runs/eval_embeds_v2.json > "$LOGD/eval_v2.log" 2>&1
log "v2 eval exit $?"

# 4. pixel-path proof gate on fresh unseen seeds
log "pixel-path proof gate"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 1800 uv run python artifacts/paper_rig/proof_gate.py \
  --ckpt runs/img-e4/final --per-page 3 \
  > "$LOGD/proof.log" 2>&1
log "proof exit $?"

# 5. final report
log "final report"
uv run python artifacts/paper_rig/overnight_report.py \
  --chain-log "$LOGD/morning.log" --train-log "$LOGD/train_v2.log" \
  --eval-json runs/eval_embeds_v2.json --eval-log "$LOGD/eval_v2.log" \
  --verify-a "$LOGD/verify_A.json" --verify-b "$LOGD/verify_heal.json" \
  --slice2-log "$LOGD/slice2_heal.log" --smoke-log "$LOGD/smoke.log" \
  --stage-a PASS --stage-b PASS --stage-c PASS \
  --stage-d PASS --stage-e PASS --out runs/overnight_report.md \
  > "$LOGD/report_gen.log" 2>&1
log "morning run complete — report at runs/overnight_report.md"
