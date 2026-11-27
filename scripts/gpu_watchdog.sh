#!/usr/bin/env bash
# GPU-util watchdog (user law 2026-11-27): fire ONLY when an expected-GPU job is
# alive but GPU utilization is hugely low (>=5 consecutive seconds < 10%).
# Output is deliberately deterministic (no timestamps) so the cron monitor
# compares equal across healthy ticks and wakes the agent only on a state change.
# Healthy states: IDLE (no expected GPU job) / OK (job alive, util healthy).
JOB=$(pgrep -f 'precompute_embed|train_gate2_row|eval_gate2|eval_sroie|embed_sroie_trainval' | tail -1)
if [ -z "$JOB" ]; then
  echo "IDLE"
  exit 0
fi
low=0; last=""
for i in 1 2 3 4 5 6; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  case "$u" in ''|*[!0-9]*) u=0;; esac
  if [ "$u" -lt 10 ]; then low=$((low+1)); else low=0; fi
  last=$u
  [ "$i" -lt 6 ] && sleep 1
done
if [ "$low" -ge 5 ]; then
  echo "TRIGGERED: GPU util <10% for >=5s while expected-GPU job pid $JOB is alive (etime $(ps -o etime= -p $JOB 2>/dev/null | tr -d ' '))"
  ps -o pid,stat,time,wchan:30 -p "$JOB"
  tail -2 /home/riju279/Documents/code/drishyam-1/runs/overnight/v4_chain.log 2>/dev/null
  ls -la --time-style=+%H:%M:%S /home/riju279/Documents/code/drishyam-1/artifacts/paper_rig/embeddings_k5/* 2>/dev/null | tail -2
else
  echo "OK job=$JOB util=${last}% low_streak=$low"
fi
