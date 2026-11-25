# GATE-2.5 MORNING PLAYBOOK (pre-committed 2026-11-25 22:05, before the verdict)

Rule of the whole document: **every branch names a mechanism and a measurement —
"tune lr and rerun" is forbidden unless the mechanism is understood and stated.**
The eval JSON lands at `artifacts/gate2_table.json` (row path: `--out` may differ —
the run prints its path). Training health lives in MLflow run `g2-row-sft`
(newest), metrics: val_acc/val_loss per epoch, train_loss/grad_norm per 25 steps.

## Stage T — training failures (detect BEFORE eval)

**T1. Process crashed overnight (any exit != 0 / no `G2.5 ROW SFT DONE` line).**
- Checkpoint exists (`runs/g2-row-sft/last.pt`)? → `touch runs/g2-row-sft/resume_flag`
  and relaunch the same command. Resume re-enters at the saved epoch.
- Crashed AGAIN at val with OOM (inference_mode was supposed to prevent it):
  → mechanism = batch-shape memory, measured fix: `--rows-per-batch 8` + resume.
  Do NOT touch lr or epochs — they are not the mechanism.

**T2. Val acc never leaves ≤0.50 (chance) across all epochs.**
Contradicts the smoke (0.6079) → suspect the FRESH-init path, not the pathway.
- Step 1 (30 min): rerun smoke variant `row_smoke` unchanged but `--init sft` —
  if SFT-init clicks again, mechanism = fresh LoRA needs warmup/longer schedule.
- Step 2: resume full run 2 more epochs (mechanism: schedule too short for fresh
  init; smoke had only 250 steps but SFT-init head start).
- Step 3 (only if both dead): K4 mid-layer slot read — read slots at encoder
  layer ~2/3 depth (textbook cut at 20/28 layers; ktcapraz found layer 14).
  This is the LAST architecture knob; nothing else gets touched first.

**T3. Memorization signature (train_loss → <0.15 while val_loss rises).**
- best.pt is protected (selected by val loss) → the run is still usable.
- If best.pt is epoch 0–1: mechanism = under-trained + too-hot late schedule.
  Action: resume 2 epochs at `--lr 1e-5`. NOT a blind lr knob-turn — the
  mechanism (late-cosine reheating) is named and measured by the val curve.

## Stage E — per-bar failures (the six pre-registered bars)

**E1. g2_1 ablation flip < 0.30 (doesn't read the page) — but val acc is decent.**
Mechanism: model answers from question priors, page is decoration.
- Action: K5 counterfactual pairs (PRE-REGISTERED in plan): same page, one
  gt-derived quantity flipped so gold flips — forces page-reading. Label
  factory change only, ~20-min re-derivation, then retrain. NOT lr, NOT epochs.

**E2. g2_4 unseen AUC < 0.75 (salary family transfer fails).**
Mechanism: family-style variety, not quantity.
- Action: re-derive labels with unseen = telecom (2nd most-distinct family,
  6,000 pages banked) as a ROBUSTNESS CHECK first — if salary failed but
  telecom passes, the bar is fragile-style, not capacity; if both fail,
  add per-family question-text diversity in the factory (v3) + retrain.

**E3. g2_3 ECE ≥ 0.08 (calibration).**
Mechanism: confidence misallocation, NOT accuracy.
- Action: RLCD stage (already built + TDD'd, `train_gate2_rlcd.py`,
  proper-scoring rewards, pre-registered abort rails). SFT ckpt untouched.
- If TS temperatures again pin at grid ceiling (T=5.0): val distribution is
  inverted → that is an E1 symptom wearing a calibration mask; fix E1 first.

**E4. g2_5 jitter flip ≥ 0.05 (question-text fragility).**
Mechanism: text-perturbation robustness of the token side.
- Action: add jitter augmentation rows in the label factory (v3): same case,
  jittered question text, same gold. One re-derivation + retrain.

**E5. g2_2 multi-q delta ≠ 0 at 4dp (determinism broken).**
Mechanism: EVAL machinery, not the model (this bar passed even in dead v1).
- Action: fp32 law breach hunt — check `--path row` kept `model.float()` and
  fp32 pages. Fix the eval, re-run eval. NEVER retrain for this bar.

**E6. Val acc high but EVERYTHING page-related fails (E1+E2 together).**
Mechanism: train-family overfit via question priors.
- Action: K5 counterfactuals + LoFO unseen rotation (telecom) in the same
  re-derivation. One retrain, both levers, pre-registered together here.

## Stage W — the win path (all six pass)

1. Second holdout: re-eval with unseen=telecom (robustness, not a new bar).
2. RLCD calibration polish (g2_3 typically tightens here).
3. Then: bank the negative result v1 + positive row-path into the skill
   ledger; Anmol pitch demo uses this model.

## What is NEVER done in any branch
- Editing the six bars after seeing the verdict.
- Retraining for g2_2 (determinism = code bug).
- Touching lr without naming the mechanism that lr fixes.
- Generating more bulk data without naming the bar that needs it.
