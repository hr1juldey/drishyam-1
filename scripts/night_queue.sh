#!/usr/bin/env bash
# Night GPU queue — FULLY RESUMABLE: on any relaunch it skips every step whose
# artifact already exists and fast-forwards to the next undone step.
# One GPU job at a time, sequential, retry-once-then-PARK. New training stops after 05:00.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=docs/NIGHT_LOG.md
mkdir -p artifacts runs
touch "$LOG"
log() { echo "- $(date '+%H:%M:%S') $1" >> "$LOG"; }
now() { date +%s; }
budget_ok() { [ "$(now)" -lt "$(date -d '05:00' +%s)" ]; }

step() { # step NAME -- cmd...
  local name="$1" tries=0 rc
  shift
  log "STEP $name start: $*"
  while [ "$tries" -lt 2 ]; do
    "$@" >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then log "STEP $name PASS (rc=0)"; return 0; fi
    tries=$((tries + 1))
    log "STEP $name fail#$tries rc=$rc — retrying"
    sleep 20
  done
  log "STEP $name PARKED (3-strikes)"
  return 1
}

log "=== NIGHT GPU QUEUE start (pid $$) ==="

# -- 0. main run evidence chain (idempotent)
if [ -f artifacts/verdict_g1t1.json ]; then
  log "verdict chain already done — fast-forward to lr screens"
else
  grep -q 'EXIT=' runs/g1-t1.log || {
    log "waiting for runs/g1-t1.log EXIT marker"
    for _ in $(seq 1 240); do
      grep -q 'EXIT=' runs/g1-t1.log && break
      sleep 60
    done
  }
  [ -f runs/g1-t1/best.pt ] || { log "FATAL no best.pt — GPU queue end"; exit 1; }
  uv run python scripts/next_step.py --done G1-T1 --note "night queue: training EXIT marker seen" >> "$LOG" 2>&1

  [ -f artifacts/temperatures.json ] || step C1 uv run python -m drishyam.calibrate
  [ -f artifacts/gate_table.json ] || step V1 uv run python -m drishyam.evaluate \
    --ckpt runs/g1-t1/best.pt --out artifacts/gate_table.json
  [ -f artifacts/verdict.json ] || step VERDICT uv run python scripts/gate_verdict.py \
    --gate artifacts/gate_table.json --out artifacts/verdict.json

  if [ -f artifacts/verdict.json ]; then
    cp -n artifacts/val_logits.npz artifacts/val_logits_g1t1.npz
    cp -n artifacts/train_report.json artifacts/train_report_g1t1.json
    cp -n artifacts/gate_table.json artifacts/gate_table_g1t1.json
    cp -n artifacts/verdict.json artifacts/verdict_g1t1.json
    cp -n artifacts/temperatures.json artifacts/temperatures_g1t1.json
    log "main-run evidence snapshotted"
    step PUSH1 git push -q origin main
  else
    log "verdict missing — lr screens withheld, queue end (watchdog may retry)"
    exit 1
  fi
fi

# -- 4. lr screen 3e-4 (idempotent: artifact check first, then resume-by-best.pt)
if [ -f artifacts/gate_table_lr3e4.json ]; then
  log "lr3e4 already evaluated — skip"
elif budget_ok; then
  if [ ! -f runs/g1-t1-lr3e4/best.pt ]; then
    log "lr screen 3e-4 training (budget ok)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      step T1LR3 uv run python -m drishyam.train --lr 3e-4 --epochs 40 --seed 42 \
        --out runs/g1-t1-lr3e4
  else
    log "lr3e4 best.pt exists (killed mid-eval?) — skip to its eval"
  fi
  if [ -f runs/g1-t1-lr3e4/best.pt ]; then
    # val_logits.npz at this point is lr3e4's (just written by its run)
    uv run python -c "
from pathlib import Path
import json
from drishyam.calibrate import fit_all
t = json.dumps(fit_all(), indent=2)
Path('artifacts/temperatures_lr3e4.json').write_text(t)
Path('artifacts/temperatures.json').write_text(t)
" >> "$LOG" 2>&1
    step V1LR3 uv run python -m drishyam.evaluate --ckpt runs/g1-t1-lr3e4/best.pt \
      --out artifacts/gate_table_lr3e4.json
    cp -n artifacts/val_logits.npz artifacts/val_logits_lr3e4.npz
  fi
else
  log "lr screen 3e-4 SKIPPED (budget)"
fi

# -- 5. lr screen 5e-5 only if plenty of budget
if [ -f artifacts/gate_table_lr5e5.json ]; then
  log "lr5e5 already evaluated — skip"
elif budget_ok && [ "$(now)" -lt "$(date -d '03:30' +%s)" ]; then
  if [ ! -f runs/g1-t1-lr5e5/best.pt ]; then
    log "lr screen 5e-5 training"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      step T1LR5 uv run python -m drishyam.train --lr 5e-5 --epochs 40 --seed 42 \
        --out runs/g1-t1-lr5e5
  fi
  if [ -f runs/g1-t1-lr5e5/best.pt ]; then
    uv run python -c "
from pathlib import Path
import json
from drishyam.calibrate import fit_all
t = json.dumps(fit_all(), indent=2)
Path('artifacts/temperatures_lr5e5.json').write_text(t)
Path('artifacts/temperatures.json').write_text(t)
" >> "$LOG" 2>&1
    step V1LR5 uv run python -m drishyam.evaluate --ckpt runs/g1-t1-lr5e5/best.pt \
      --out artifacts/gate_table_lr5e5.json
    cp -n artifacts/val_logits.npz artifacts/val_logits_lr5e5.npz
  fi
else
  log "lr screen 5e-5 SKIPPED (budget)"
fi

git add -A >> "$LOG" 2>&1
git commit -q -m 'chore(night): verdict + lr screen evidence' >> "$LOG" 2>&1
step PUSH2 git push -q origin main
log "=== NIGHT GPU QUEUE end ==="
