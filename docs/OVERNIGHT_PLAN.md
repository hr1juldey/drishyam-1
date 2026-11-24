# Overnight Plan — 2026-11-24 night

Goal: user sleeps; morning = a finished, usable model (checkpoint + inference command + honest eval numbers + report).

## Chain stages (overnight_chain.sh)
A. WAIT slice-1 precompute (50k: thermal 20k/savings 15k/card 15k) → verify 5 shards + NaN spot-checks + label counts.
   FAIL: resume once (script is shard-resumable); still fails → skip to B, report.
B. Slice-2 precompute (NEW families → embeddings_slice2/): electricity 8000, telecom 6000, salary 6000, table 4000 (~2h).
   FAIL: resume once; fail → report, proceed with slice-1-only training.
C. Trainer smoke (10 steps on 200 embeddings): finite loss required.
   FAIL: chain stops at C; embeddings are the night's deliverable; report.
D. Train decoder-only on ALL cached embeddings (~74k, bf16, grad ckpt, VRAM guard).
   Projector FROZEN (protects cached embeddings' validity). NaN or loss >3x initial → auto-halt, keep last ckpt.
E. Eval: greedy-decode exact-match on train-institution rows vs holdout bucket (seed%10==7) = LOIO claim.
   FAIL: report says eval failed; checkpoint still delivered.
F. runs/overnight_report.md — always written, pass or fail, with next actions.

## Morning usage
python artifacts/paper_rig/infer_docs.py --image <png> --q "What is the total amount?" -> answer

## Rules
- Paper/report states only measured facts.
- Every stage logged to runs/overnight/.
- Chain never deletes slices; failures leave resumable state.
