#!/usr/bin/env bash
# v5-A: SROIE-trainval domain-adaptation chain (user option A, 2026-11-27).
# FIRE ONLY ON EXPLICIT USER WORD (GPU-HOLD law). Sequence:
#   rows (CPU) -> embed 625 trainval JPEGs (GPU ~18m) -> bank/label gate ->
#   fine-tune v4 @ lr 1e-5 x 2 epochs on real rows only (~40m) ->
#   six-bar eval + SROIE re-score (~45m)
# Contract bands (pre-registered in the plan file):
#   PASS    SROIE AUC >= 0.65 AND rejection >= 0.40 AND internal bars hold
#   PARTIAL SROIE AUC 0.55-0.65 -> scale real data, then consider B on top
#   DEAD    SROIE AUC <= 0.55 OR any internal bar breaks -> B (scanify) is next
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=runs/overnight/v5a_chain.log
mkdir -p runs/overnight

log() { echo "[v5a $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "chain armed — waiting for GPU to be free (no expected-GPU job may be running)"
while pgrep -f 'precompute_embed|train_gate2_row|eval_gate2|eval_sroie|g1b_codec_train' >/dev/null 2>&1; do
  sleep 60
done
log "GPU free"

log "gate 1: build trainval rows"
uv run python -m drishyam.sroie_trainval >> "$LOG" 2>&1 \
  || { echo "[v5a ABORT] row build failed" | tee -a "$LOG"; exit 1; }
grep -c '"family": "sroie"' artifacts/gate2_labels_sroie_train.jsonl >> "$LOG" 2>&1

log "gate 2: embed 625 trainval receipts"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  uv run python -m drishyam.embed_sroie_trainval --ckpt runs/g2-row-sft-v4/best.pt \
  2>&1 | tail -3 | tee -a "$LOG"
test "${PIPESTATUS[0]}" -eq 0 || { echo "[v5a ABORT] trainval embed failed (PIPESTATUS ${PIPESTATUS[0]})" | tee -a "$LOG"; exit 1; }

log "gate 3: bank/label consistency"
uv run python scripts/check_sroie_bank.py >> "$LOG" 2>&1 \
  || { echo "[v5a ABORT] bank/label mismatch" | tee -a "$LOG"; exit 1; }

log "v5a fine-tune starting (v4 init, real rows only, lr 1e-5 x 2ep)"
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
  uv run python -m drishyam.train_gate2_row --lr 1e-5 --epochs 2 --rows-per-batch 16 \
  --init sft --sft-ckpt runs/g2-row-sft-v4/best.pt \
  --train-labels gate2_labels_sroie_train.jsonl \
  --val-labels gate2_labels_sroie_val.jsonl \
  --out runs/g2-row-sft-v5a 2>&1 | tee -a "$LOG"
test "${PIPESTATUS[0]}" -eq 0 || { echo "[v5a ABORT] fine-tune failed (PIPESTATUS ${PIPESTATUS[0]})" | tee -a "$LOG"; exit 1; }

log "six-bar eval (v5a best.pt)"
HF_HUB_OFFLINE=1 uv run python -m drishyam.eval_gate2 \
  --ckpt runs/g2-row-sft-v5a/best.pt --path row \
  --out artifacts/gate2_row_table_v5a.json 2>&1 | tail -30 | tee -a "$LOG"
test "${PIPESTATUS[0]}" -eq 0 || { echo "[v5a ABORT] six-bar eval failed (PIPESTATUS ${PIPESTATUS[0]})" | tee -a "$LOG"; exit 1; }

log "SROIE re-score (v5a best.pt) — THE DECIDING NUMBER"
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m drishyam.eval_sroie \
  --ckpt runs/g2-row-sft-v5a/best.pt \
  --out artifacts/sroie_table_v5a.json \
  --dump-probs artifacts/sroie_v5a_probs.json 2>&1 | tail -6 | tee -a "$LOG"
test "${PIPESTATUS[0]}" -eq 0 || { echo "[v5a ABORT] sroie eval failed (PIPESTATUS ${PIPESTATUS[0]})" | tee -a "$LOG"; exit 1; }

log "CHAIN COMPLETE — verdict: artifacts/sroie_table_v5a.json vs the pre-registered bands"
