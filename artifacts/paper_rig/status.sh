#!/bin/bash
# Self-serve run status — run any time:
#   bash artifacts/paper_rig/status.sh          (once)
#   watch -n 30 bash artifacts/paper_rig/status.sh   (live dashboard, Ctrl-C to stop)
cd ~/Documents/code/drishyam-1 || exit 1
L=runs/overnight/train_v2.log

echo "== stage =="
grep "^\[morning" runs/overnight/morning.log 2>/dev/null | tail -2

echo; echo "== v2 training =="
T=$(grep "^step" "$L" 2>/dev/null | tail -1)
if [ -n "$T" ]; then
  STEP=$(echo  "$T" | awk '{print $2}' | cut -d/ -f1)
  TOTAL=$(echo "$T" | awk '{print $2}' | cut -d/ -f2)
  LOSS=$(echo  "$T" | awk '{print $4}')
  RPS=$(echo   "$T" | awk '{print $8}')
  REM=$(( (TOTAL - STEP) * 16 ))
  ETA_MIN=$(( REM / 34 / 60 ))
  echo "$T"
  echo "progress: $STEP/$TOTAL ($(( STEP * 100 / TOTAL ))%)   loss $LOSS   eta ~${ETA_MIN} min"
else
  echo "no training lines yet"
fi

echo; echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

echo; echo "== artifacts (verdict pieces) =="
for f in runs/img-e4/final runs/eval_embeds_v2.json runs/proof/PROOF.md runs/overnight_report.md; do
  [ -e "$f" ] && echo "READY   $f" || echo "pending $f"
done

echo; echo "== eval (when ready) =="
[ -f runs/eval_embeds_v2.json ] && cat runs/eval_embeds_v2.json | head -30
echo; echo "== proof (when ready) =="
[ -f runs/proof/PROOF.md ] && grep -E "exact match|Per-family" runs/proof/PROOF.md
