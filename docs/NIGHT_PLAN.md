# Night Plan — 2026-11-23 → 2026-11-24 morning

Goal: user sleeps, machine works. Morning = verdicts, not promises.

## Lanes (co-residency law respected)

- **GPU lane** (`scripts/night_queue.sh`): waits for the running G1-T1 training →
  C1 temperatures → V1 gate suites → **verdict assembly (criteria 1–5)** →
  commit+push → lr screen 3e-4 (train+eval) → lr screen 5e-5 **only if ≥2 h
  budget remains before 07:00**. One GPU job at a time, sequential, 3-strikes.
- **CPU lane** (`scripts/night_cpu.sh`): waits for training EXIT (box law: teacher
  never runs while training holds the GPU) → **G2-TF1 bake-off** (lfm3-omni vs
  gemma4:31b, 50 gold train cases × 3 samples, agreement-per-hour) → night
  drafter (gemma4:31b drafts new seen-3 states into `drafts/`, quarantined,
  NEVER mixed into Gate-1 benchmark training) until 06:30 or 400 cases.
  Drafter runs with num_thread=8 so it cannot starve the GPU lane.

## SSD guard

- Checkpoints: best.pt/last.pt only (~120 MB/run × ≤3 runs).
- Drafts/bakeoff: jsonl text, ≤ ~20 MB.
- No new model downloads except banking77 (~5 MB). No dataset duplication.
- Total projected night write < 1 GB on 620 GB free.

## Morning deliverables

1. `artifacts/verdict.json` — criteria 1–5 PASS/FAIL each, with the numbers.
2. `artifacts/gate_table.json` — our model vs `baseline_report.json` (B1) vs
   `b2_report.json` (B2) on the same splits.
3. lr screen: `runs/g1-t1-lr3e4/` + its gate table (and 5e-5 if budget allowed).
4. `artifacts/bakeoff.json` — drafter decision data (agreement-per-hour).
5. `drafts/seen3_drafts.jsonl` — teacher-drafted cases for the OPTIONAL Gate-1.5
   augmentation experiment (user decides; never auto-mixed).
6. `docs/NIGHT_LOG.md` — minute-by-minute trail; everything committed + pushed.

## Failure containment

Every step: retry once, then PARK and continue the queue (3-strikes law).
State lives in `docs/NIGHT_LOG.md`; wake-up notifications steer the live lanes.
