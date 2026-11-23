#!/usr/bin/env bash
# Night CPU lane — drafter FIRST (feeds the Gate-1.5 data lever), bake-off after.
# Resumable: skips whichever phase already produced its artifact.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=docs/NIGHT_LOG.md
touch "$LOG"
log() { echo "- $(date '+%H:%M:%S') [cpu] $1" >> "$LOG"; }

log "=== NIGHT CPU LANE start (pid $$) ==="

NDRAFTS=$(wc -l < drafts/seen3_drafts.jsonl 2>/dev/null || echo 0)
BUDGET=$(date -d '06:00' +%s)
LEFT=$((BUDGET - $(date +%s)))
if [ "$LEFT" -gt 600 ] && [ "$NDRAFTS" -lt 400 ]; then
  log "night drafter start (gemma4:31b, ${LEFT}s to 06:00, max 400 total, num_predict=900)"
  timeout "$LEFT" uv run python scripts/bakeoff.py --mode draft \
    --model gemma4:31b --budget-hours "$((LEFT / 3600))" --max-cases 400 \
    >> "$LOG" 2>&1
  log "night drafter end — drafts at drafts/seen3_drafts.jsonl ($(wc -l < drafts/seen3_drafts.jsonl 2>/dev/null || echo 0) cases)"
else
  log "drafter SKIPPED (budget=${LEFT}s, drafts=${NDRAFTS})"
fi

# bake-off only if drafter finished early and time remains
if [ ! -f artifacts/bakeoff.json ]; then
  B2=$(date -d '06:30' +%s)
  LEFT2=$((B2 - $(date +%s)))
  if [ "$LEFT2" -gt 2700 ]; then
    log "bake-off start (lfm3-omni vs gemma4:31b, 20 cases x 3 samples, ${LEFT2}s window)"
    if timeout "$LEFT2" uv run python scripts/bakeoff.py --mode bakeoff \
         --model "lfm3-omni:latest+gemma4:31b" --cases 20 --samples 3 \
         --out artifacts/bakeoff.json >> "$LOG" 2>&1; then
      log "bake-off PASS"
    else
      log "bake-off PARKED (rc=$?)"
    fi
  else
    log "bake-off SKIPPED (budget=${LEFT2}s < 45min)"
  fi
else
  log "bake-off already done — skip"
fi
log "=== NIGHT CPU LANE end ==="
