#!/usr/bin/env bash
# Night CPU lane: wait for training EXIT -> teacher bake-off -> night drafter.
# Teacher never runs while training holds the GPU (co-residency law).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=docs/NIGHT_LOG.md
touch "$LOG"
log() { echo "- $(date '+%H:%M:%S') [cpu] $1" >> "$LOG"; }

log "=== NIGHT CPU LANE start (pid $$) — waiting for training EXIT ==="
for _ in $(seq 1 240); do
  grep -q 'EXIT=' runs/g1-t1.log && break
  sleep 60
done
log "training EXIT seen — CPU lane starting (num_thread=8 capped so GPU lane keeps headroom)"

log "bake-off start (lfm3-omni vs gemma4:31b, 50 cases x 3 samples)"
if timeout 9000 uv run python scripts/bakeoff.py --mode bakeoff \
     --model "lfm3-omni:latest+gemma4:31b" --cases 50 --samples 3 \
     --out artifacts/bakeoff.json >> "$LOG" 2>&1; then
  log "bake-off PASS"
else
  log "bake-off PARKED (rc=$?) — drafter still runs with gemma4:31b default"
fi

BUDGET=$(date -d '06:30' +%s)
LEFT=$(( BUDGET - $(date +%s) ))
if [ "$LEFT" -gt 1800 ]; then
  log "night drafter start (gemma4:31b, budget $((${LEFT} / 3600))h, max 400 cases, drafts/ quarantined)"
  timeout "$LEFT" uv run python scripts/bakeoff.py --mode draft \
    --model gemma4:31b --budget-hours "$((${LEFT} / 3600))" --max-cases 400 \
    >> "$LOG" 2>&1
  log "night drafter end — drafts at drafts/seen3_drafts.jsonl ($(wc -l < drafts/seen3_drafts.jsonl 2>/dev/null || echo 0) cases)"
else
  log "drafter SKIPPED (insufficient budget)"
fi
log "=== NIGHT CPU LANE end ==="
