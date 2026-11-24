#!/bin/bash
# Overnight chain 2026-11-24: A wait slice-1 -> B slice-2 -> C trainer smoke -> D train -> E eval -> F report.
# Every stage: verify, resume-once on failure, never stop the chain (statuses tracked).
# Report runs/overnight_report.md is written at the END no matter what happened.
set -u
cd ~/Documents/code/drishyam-1
E=artifacts/paper_rig/embeddings
E2=artifacts/paper_rig/embeddings_slice2
LOGD=runs/overnight
mkdir -p "$LOGD"
declare -A STAGE

log(){ echo "[chain $(date +%H:%M:%S)] $*" | tee -a "$LOGD/chain.log"; }

families="electricity:8000,telecom:6000,salary:6000,table:4000"

# ---------- STAGE A: wait for the running slice-1 precompute (pid 3212065) ----------
log "stage A: waiting for slice-1 precompute (50k)"
A_OK=0
for i in $(seq 1 240); do   # poll up to 8h
  if grep -q "PRECOMPUTE COMPLETE" "$E/precompute.log" 2>/dev/null; then A_OK=1; break; fi
  if ! ps -p 3212065 > /dev/null 2>&1; then
    log "stage A: precompute process died before completion — resuming once"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      uv run python artifacts/paper_rig/precompute_embed.py --balanced 50000 \
      >> "$E/precompute.log" 2>&1
    grep -q "PRECOMPUTE COMPLETE" "$E/precompute.log" && A_OK=1
    break
  fi
  sleep 120
done
if [ "$A_OK" = 1 ]; then
  if uv run python artifacts/paper_rig/verify_embeds.py --expect "$E" 50000 \
      > "$LOGD/verify_A.json" 2>&1; then
    STAGE[A]=PASS; log "stage A PASS"
  else
    STAGE[A]=VERIFY_FAIL; log "stage A: rows complete but verification FAILED — resuming once"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      uv run python artifacts/paper_rig/precompute_embed.py --balanced 50000 \
      >> "$E/precompute.log" 2>&1
    if uv run python artifacts/paper_rig/verify_embeds.py --expect "$E" 50000 \
        > "$LOGD/verify_A.json" 2>&1; then
      STAGE[A]=PASS_RETRY; log "stage A PASS after resume"
    else
      STAGE[A]=FAIL; log "stage A FAIL — training will proceed on whatever shards verify"
    fi
  fi
else
  STAGE[A]=FAIL_TIMEOUT; log "stage A: never completed — skipping to slice-2 anyway"
fi

# ---------- STAGE B: slice-2, four new families, own directory ----------
log "stage B: slice-2 precompute ($families)"
B_OK=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout 14400 uv run python artifacts/paper_rig/precompute_embed.py --families "$families" \
  --out-dir "$E2" > "$LOGD/slice2.log" 2>&1
grep -q "PRECOMPUTE COMPLETE" "$LOGD/slice2.log" && B_OK=1
if [ "$B_OK" = 0 ]; then
  log "stage B: died or timed out — resuming once (shard-skip)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    timeout 14400 uv run python artifacts/paper_rig/precompute_embed.py --families "$families" \
    --out-dir "$E2" >> "$LOGD/slice2.log" 2>&1
  grep -q "PRECOMPUTE COMPLETE" "$LOGD/slice2.log" && B_OK=1
fi
if [ "$B_OK" = 1 ] && uv run python artifacts/paper_rig/verify_embeds.py "$E2" \
    > "$LOGD/verify_B.json" 2>&1; then
  STAGE[B]=PASS; log "stage B PASS"
  DIRS="$E,$E2"
else
  STAGE[B]=FAIL; log "stage B FAIL — training proceeds slice-1 only"
  DIRS="$E"
fi

# ---------- STAGE C: GPU trainer smoke + eval wiring smoke ----------
log "stage C: trainer GPU smoke (10 steps)"
if uv run python artifacts/paper_rig/embed_trainer.py --dirs "$DIRS" \
    --out "$LOGD/smoke_ckpt" --smoke --micro 16 --accum 1 --warmup 2 \
    --log-every 5 --ckpt-every 0 \
    > "$LOGD/smoke.log" 2>&1 && grep -q "SMOKE OK" "$LOGD/smoke.log"; then
  STAGE[C]=PASS; log "stage C PASS (trainer)"
  # eval wiring smoke: 2 rows per split through the greedy loop
  if uv run python artifacts/paper_rig/eval_embeds.py --dirs "$DIRS" \
      --ckpt "$LOGD/smoke_ckpt/final" --per-split 2 --max-new 6 \
      > "$LOGD/eval_smoke.log" 2>&1; then
    log "stage C eval wiring PASS"
  else
    STAGE[C]=EVAL_WIRING_FAIL; log "stage C: trainer OK but eval wiring FAILED — training will run, eval will be rerun manually in the morning"
  fi
else
  STAGE[C]=FAIL; log "stage C FAIL — training aborted, embeddings are the night's deliverable"
fi

# ---------- STAGE D: full training ----------
if [ "${STAGE[C]}" = "PASS" ] || [ "${STAGE[C]}" = "EVAL_WIRING_FAIL" ]; then
  log "stage D: training on $DIRS"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python artifacts/paper_rig/embed_trainer.py --dirs "$DIRS" \
    --out runs/img-e2 --micro 16 --accum 1 --epochs 1 --ckpt-every 1500 --keep 2 \
    > "$LOGD/train.log" 2>&1
  case $? in
    0) STAGE[D]=PASS; log "stage D PASS" ;;
    3) STAGE[D]=HALTED_DIVERGENCE; log "stage D halted (divergence guard) — final ckpt kept" ;;
    *) STAGE[D]=FAIL; log "stage D failed — last checkpoint kept if any" ;;
  esac
else
  STAGE[D]=SKIPPED
fi

# ---------- STAGE E: eval ----------
if [ "${STAGE[D]}" = "PASS" ] || [ "${STAGE[D]}" = "HALTED_DIVERGENCE" ]; then
  log "stage E: embedding-space eval (LOIO holdout)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python artifacts/paper_rig/eval_embeds.py --dirs "$DIRS" \
    --ckpt runs/img-e2/final --per-split 300 --out runs/eval_embeds.json \
    > "$LOGD/eval.log" 2>&1 && STAGE[E]=PASS || STAGE[E]=FAIL
  log "stage E ${STAGE[E]}"
else
  STAGE[E]=SKIPPED
fi

# ---------- STAGE F: report ----------
log "stage F: writing report"
uv run python artifacts/paper_rig/overnight_report.py \
  --chain-log "$LOGD/chain.log" --train-log "$LOGD/train.log" \
  --eval-json runs/eval_embeds.json --eval-log "$LOGD/eval.log" \
  --verify-a "$LOGD/verify_A.json" --verify-b "$LOGD/verify_B.json" \
  --slice2-log "$LOGD/slice2.log" --smoke-log "$LOGD/smoke.log" \
  --stage-a "${STAGE[A]:-NOT_RUN}" --stage-b "${STAGE[B]:-NOT_RUN}" \
  --stage-c "${STAGE[C]:-NOT_RUN}" --stage-d "${STAGE[D]:-NOT_RUN}" \
  --stage-e "${STAGE[E]:-NOT_RUN}" --out runs/overnight_report.md \
  > "$LOGD/report_gen.log" 2>&1
log "chain finished: A=${STAGE[A]:-?} B=${STAGE[B]:-?} C=${STAGE[C]:-?} D=${STAGE[D]:-?} E=${STAGE[E]:-?}"
