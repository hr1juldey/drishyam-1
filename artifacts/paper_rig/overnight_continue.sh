#!/bin/bash
# Continuation: 3 more epochs warm-started from tonight's v1 (loss was still declining),
# then corrected eval, then the final morning report.
set -u
cd ~/Documents/code/drishyam-1
LOGD=runs/overnight
log(){ echo "[cont $(date +%H:%M:%S)] $*" | tee -a "$LOGD/chain.log"; }

log "continuation: 3 epochs warm-start from runs/img-e2/final"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 10800 uv run python artifacts/paper_rig/embed_trainer.py \
  --dirs artifacts/paper_rig/embeddings --out runs/img-e3 \
  --init-from runs/img-e2/final --epochs 3 --lr 2e-5 --warmup 100 \
  --micro 16 --accum 1 --ckpt-every 2000 --keep 1 \
  > "$LOGD/train_e3.log" 2>&1
D3=$?
log "continuation training exit $D3"

log "corrected eval on runs/img-e3/final"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 1800 uv run python artifacts/paper_rig/eval_embeds.py \
  --dirs artifacts/paper_rig/embeddings --ckpt runs/img-e3/final \
  --per-split 300 --out runs/eval_embeds.json \
  > "$LOGD/eval_e3.log" 2>&1
E3=$?
log "corrected eval exit $E3"

log "writing final report"
uv run python artifacts/paper_rig/overnight_report.py \
  --chain-log "$LOGD/chain.log" --train-log "$LOGD/train_e3.log" \
  --eval-json runs/eval_embeds.json --eval-log "$LOGD/eval_e3.log" \
  --verify-a "$LOGD/verify_A.json" --verify-b "$LOGD/verify_B.json" \
  --slice2-log "$LOGD/slice2.log" --smoke-log "$LOGD/smoke.log" \
  --stage-a PASS --stage-b FAIL --stage-c EVAL_WIRING_FAIL_FIXED \
  --stage-d PASS --stage-e "PASS_CORRECTED" --out runs/overnight_report.md \
  > "$LOGD/report_gen.log" 2>&1
log "continuation complete"
