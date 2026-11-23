#!/usr/bin/env bash
# Night CPU lane — RESUMABLE chain: bake-off (skip if done) -> drafter until 06:30/400.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=docs/NIGHT_LOG.md
touch "$LOG"
log() { echo "- $(date '+%H:%M:%S') [cpu] $1" >> "$LOG"; }

log "=== NIGHT CPU LANE start (pid $$) ==="
if [ ! -f artifacts/bakeoff.json ]; then
  log "bake-off start (lfm3-omni vs gemma4:31b, 50 cases x 3 samples)"
  if timeout 9000 uv run python scripts/bakeoff.py --mode bakeoff \
       --model "lfm3-omni:latest+gemma4:31b" --cases 50 --samples 3 \
       --out artifacts/bakeoff.json >> "$LOG" 2>&1; then
    log "bake-off PASS"
  else
    log "bake-off PARKED (rc=$?) — drafter still runs with gemma4:31b default"
  fi
else
  log "bake-off already done — skip"
fi

NDRAFTS=$(wc -l < drafts/seen3_drafts.jsonl 2>/dev/null || echo 0)
BUDGET=$(date -d '06:30' +%s)
LEFT=$((BUDGET - $(date +%s)))
if [ "$LEFT" -gt 1800 ] && [ "$NDRAFTS" -lt 400 ]; then
  log "night drafter start (gemma4:31b, ${LEFT}s budget, max 400 total, drafts/ quarantined)"
  timeout "$LEFT" uv run python scripts/bakeoff.py --mode draft \
    --model gemma4:31b --budget-hours "$((LEFT / 3600))" --max-cases 400 \
    >> "$LOG" 2>&1
  log "night drafter end — drafts at drafts/seen3_drafts.jsonl ($(wc -l < drafts/seen3_drafts.jsonl 2>/dev/null || echo 0) cases)"
else
  log "drafter SKIPPED (budget=${LEFT}s, drafts=${NDRAFTS})"
fi
log "=== NIGHT CPU LANE end ==="
