#!/usr/bin/env bash
# Idempotent night watchdog: relaunch any dead lane whose work is not yet complete.
# Runs from cron every ~20 min. Lock prevents double-launch races.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=docs/NIGHT_LOG.md
touch "$LOG"
exec 9>"$HOME/.hermes/cache/scratch/night_watchdog.lock"
flock -n 9 || exit 0
now() { date +%s; }
log() { echo "- $(date '+%H:%M:%S') [watchdog] $1" >> "$LOG"; }

# -- GPU lane: needed until verdict + lr screens exist; new training stops after 05:00
GPU_ALIVE=$(pgrep -fc 'scripts/night_queue\.sh' || true)
GPU_JOB=$(pgrep -fc 'drishyam\.(train|evaluate)' || true)
GPU_WORK_LEFT=0
[ -f artifacts/verdict_g1t1.json ] || GPU_WORK_LEFT=1
[ -f artifacts/gate_table_lr3e4.json ] || GPU_WORK_LEFT=1
if [ "$GPU_ALIVE" -eq 0 ] && [ "$GPU_JOB" -eq 0 ] && [ "$GPU_WORK_LEFT" -eq 1 ] \
   && [ "$(now)" -lt "$(date -d '05:00' +%s)" ]; then
  log "GPU lane dead with work left — relaunching queue"
  setsid bash scripts/night_queue.sh >> "$LOG" 2>&1 < /dev/null &
fi

# -- CPU lane: needed until bakeoff.json exists and drafter hits 400/06:30
CPU_ALIVE=$(pgrep -fc 'scripts/night_cpu_chain\.sh' || true)
CPU_JOB=$(pgrep -fc 'scripts/bakeoff\.py' || true)
CPU_WORK_LEFT=0
[ -f artifacts/bakeoff.json ] || CPU_WORK_LEFT=1
NDRAFTS=$(wc -l < drafts/seen3_drafts.jsonl 2>/dev/null || echo 0)
if [ "$NDRAFTS" -lt 400 ] && [ "$(now)" -lt "$(date -d '06:30' +%s)" ]; then
  CPU_WORK_LEFT=1
fi
if [ "$CPU_ALIVE" -eq 0 ] && [ "$CPU_JOB" -eq 0 ] && [ "$CPU_WORK_LEFT" -eq 1 ]; then
  log "CPU lane dead with work left — relaunching chain"
  setsid bash scripts/night_cpu_chain.sh >> "$LOG" 2>&1 < /dev/null &
fi
exit 0
