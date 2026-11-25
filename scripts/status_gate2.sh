#!/usr/bin/env bash
# gate2 stage lights — reads ONLY the runs' own logs. stale report marked OLD (0319363 law).
cd "$(dirname "$0")/.." || exit 1
L=runs/overnight
light() { printf "%-8s %s\n" "$1" "$2"; }
[ -f docs/REPO_MAP.md ] && light MAP "OK ($(date -r docs/REPO_MAP.md +%H:%M))" || light MAP "-"
[ -f artifacts/gate2_labels.jsonl ] && light LABELS "$(wc -l < artifacts/gate2_labels.jsonl) rows" || light LABELS "-"
grep -q "2 passed" runs/overnight/g2_glue.log 2>/dev/null && light GLUE "PASS" || light GLUE "-"
[ -f runs/g2-smoke/best.pt ] && light SMOKE "DONE" || light SMOKE "-"
[ -f runs/g2-sft/last.pt ] && light SFT "$(grep -oE 'ep[0-9]+ .*acc=[0-9.]+' $L/g2_sft.log 2>/dev/null | tail -1 || echo running)" || light SFT "-"
[ -f artifacts/gate2_table.json ] && light EVAL "table written $(date -r artifacts/gate2_table.json +%H:%M)" || light EVAL "-"
[ -f artifacts/gate2_verdict.json ] && light VERDICT "$(python3 -c "import json;d=json.load(open('artifacts/gate2_verdict.json'));print(d.get('verdict','?'))" 2>/dev/null)" || light VERDICT "-"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
