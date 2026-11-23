#!/usr/bin/env bash
# Night GPU queue: wait for T1 -> C1 -> V1 -> verdict -> push -> lr screen.
# One GPU job at a time, sequential, retry-once-then-PARK. Budget stops new training at 05:00.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=docs/NIGHT_LOG.md
mkdir -p artifacts runs
touch "$LOG"
log() { echo "- $(date '+%H:%M:%S') $1" >> "$LOG"; }
now() { date +%s; }
budget_ok() { [ "$(now)" -lt "$(date -d '05:00' +%s)" ]; }

step() { # step NAME SECONDS -- cmd...
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

# -- 0. wait for the running T1 to finish (EXIT marker appended by its launcher)
log "waiting for runs/g1-t1.log EXIT marker"
for _ in $(seq 1 240); do
  grep -q 'EXIT=' runs/g1-t1.log && break
  sleep 60
done
grep -q 'EXIT=0' runs/g1-t1.log || { log "T1 EXIT nonzero or timeout — check log, queue continues to C1/V1 only if best.pt exists"; }
[ -f runs/g1-t1/best.pt ] || { log "FATAL no best.pt — GPU queue end"; exit 1; }
uv run python scripts/next_step.py --done G1-T1 --note "night queue: training EXIT marker seen" >> "$LOG" 2>&1

# -- 1..3 verdict chain
step C1 uv run python -m drishyam.calibrate
step V1 uv run python -m drishyam.evaluate --ckpt runs/g1-t1/best.pt --out artifacts/gate_table.json
step VERDICT uv run python scripts/gate_verdict.py --gate artifacts/gate_table.json --out artifacts/verdict.json

# -- preserve main-run evidence before any second run overwrites shared artifacts
cp artifacts/val_logits.npz artifacts/val_logits_g1t1.npz
cp artifacts/train_report.json artifacts/train_report_g1t1.json
cp artifacts/gate_table.json artifacts/gate_table_g1t1.json
cp artifacts/verdict.json artifacts/verdict_g1t1.json
[ -f artifacts/temperatures.json ] && cp artifacts/temperatures.json artifacts/temperatures_g1t1.json
step PUSH1 git push -q origin main

# -- 4. lr screen 3e-4
if budget_ok; then
  log "lr screen 3e-4 launching (budget ok)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    step T1LR3 uv run python -m drishyam.train --lr 3e-4 --epochs 40 --seed 42 \
      --out runs/g1-t1-lr3e4
  if [ -f runs/g1-t1-lr3e4/best.pt ]; then
    # val_logits.npz at this point is lr3e4's (just overwritten by its run)
    uv run python -c "
from pathlib import Path
import json
from drishyam.calibrate import fit_all
Path('artifacts/temperatures.json').write_text(json.dumps(fit_all(), indent=2))
" >> "$LOG" 2>&1
    step V1LR3 uv run python -m drishyam.evaluate --ckpt runs/g1-t1-lr3e4/best.pt \
      --out artifacts/gate_table_lr3e4.json
    cp artifacts/val_logits.npz artifacts/val_logits_lr3e4.npz
  fi
else
  log "lr screen 3e-4 SKIPPED (budget)"
fi

# -- 5. lr screen 5e-5 only if plenty of budget
if budget_ok && [ "$(now)" -lt "$(date -d '03:30' +%s)" ]; then
  log "lr screen 5e-5 launching"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    step T1LR5 uv run python -m drishyam.train --lr 5e-5 --epochs 40 --seed 42 \
      --out runs/g1-t1-lr5e5
  if [ -f runs/g1-t1-lr5e5/best.pt ]; then
    step V1LR5 uv run python -m drishyam.evaluate --ckpt runs/g1-t1-lr5e5/best.pt \
      --out artifacts/gate_table_lr5e5.json
    cp artifacts/val_logits.npz artifacts/val_logits_lr5e5.npz
  fi
else
  log "lr screen 5e-5 SKIPPED (budget)"
fi

step PUSH2 git add -A && true
git commit -q -m 'chore(night): verdict + lr screen evidence' >> "$LOG" 2>&1
git push -q origin main >> "$LOG" 2>&1
log "=== NIGHT GPU QUEUE end ==="
