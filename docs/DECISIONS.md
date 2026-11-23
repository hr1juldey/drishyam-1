# DECISIONS — spec deltas, improvements, open questions

Original spec: "Gate 1 — Decoupled T5Gemma-2-270M System-1 Decision Engine"
(user message 2026-11-23). This file records every deviation and every proposal
made during the planning run. Items marked **[OPEN]** need the user's word; the
recorded default applies until then.

## 1. ROCm → CUDA conversion (mandated by user)

| Spec (ROCm wording) | This repo (RTX 3060, CUDA) |
|---|---|
| "If torch.optim.AdamW NaNs on ROCm, fall back to ManualAdamW" | Gate KEPT as hard precondition but expectation inverted: AdamW+bf16 on CUDA/SM86 is battle-tested. Both arms (torch AdamW, ManualAdamW eps=1e-6) run in S0 and are recorded; either passing satisfies the gate. ManualAdamW stays a code path for the ablation regardless. |
| (implicit ROCm env) | bf16 native (SM 8.6); no fp16 anywhere (no GradScaler); TF32 on for fp32 matmuls; `attn_implementation="sdpa"`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for 12 GB fragmentation headroom. |
| — | CUDA runtime comes bundled in the PyPI torch wheel; no system toolkit needed. No HIP/HSA env vars anywhere. |
| — | GPU timing = CUDA events + `torch.cuda.synchronize()`; invariance/ablation evals under deterministic algorithms; training nondeterministic by choice. |

## 2. Environment facts verified (2026-11-23)

RTX 3060 12 GB · driver 580.178.04 · CC 8.6 · uv 0.13.7 · ast-grep 0.51.3 ·
gh authed as `hr1juldey` · 635 GB free. All three HF assets live (dataset
Apache-2.0, not gated; base model multimodal → vision freeze required exactly as
spec §0 says; reference baseline exists with its own engine code).

## 3. Slot token strategy — default: dedicated added token

The decoder input needs a `<slot>` marker after each option. Options: (a) add a
new special token and resize the tied embeddings; (b) reuse an existing rare
token. **Default: (a)** — Gemma's tokenizer has no T5-style sentinels to reuse
and hunting a "rare" token is fragile; resizing adds one row to a 262k vocab
(negligible) and the tied LM head resizes with it. Resize happens BEFORE LoRA
attach; the new row trains. Sentinel-reuse is kept as a named ablation.
**[OPEN]** — confirm, or name a token you want reused.

## 4. Cross-question invariance — metric made exact (improvement)

Mathematically, Q1's slot logit is independent of Q2..Qn's presence (causal
masking) — but NOT necessarily bitwise: different decoder sequence lengths
change kernel shapes, and bf16 reduction-order changes can shift probabilities
by ~1e-4. Spec's "== 0.0000" is therefore at risk from float physics, not
architecture. Protocol: (i) report max|Δp| raw (fp64 softmax over bf16 logits),
at 4dp, and the bitwise-identical fraction; (ii) an eval mode that pads every
multi-question batch to a FIXED token shape (masked) so "Q1 alone" and "Q1 in
batch" launch identical kernels — exact 0.0000 by construction; (iii) GO bar
applies to the 4dp reading of the deployment (unpadded) path: **0.0000 @4dp
required; raw Δ > 5e-5 is a structural bug to fix, not a tolerance to grant.**
**[OPEN]** — accept the padded-shape eval mode as the shipped inference mode?

## 5. Temperature scaling on the unseen workflow (clarification)

Spec says per-primitive T fitted on val. Made explicit: fitted on the val split
of the THREE SEEN workflows only, applied UNCHANGED to the held-out workflow.
Refitting T on unseen data would leak the "zero-shot" claim.

## 6. Multi-question suite sizing (deviation, recorded)

Dataset provides exactly 5 questions per case. Spec said 4–6. Suite = 100 states
× their native 5 (Q1-alone vs Q1..Q5 batched); deviation recorded rather than
synthesizing fake 4/6-question variants that no deployment would produce.

## 7. Val carve sizes (data reality)

Each workflow has only 300 train cases. Split = 240 train / 60 val per seen
workflow (stratified), official test 100 stays test. Total LoRA training data:
720 cases × 5 questions = 3,600 decisions/epoch — hours-cap ~1 h on the 3060.

## 8. Baseline 2 fallback (risk management)

`ktcapraz/t5gemma2-decision-270m-v2` exists but ships a custom engine
(trust_remote_code, own prompt/pointer format). If it cannot be driven cleanly
within 2 attempts, the documented fallback trains OUR encoder-stuffed T5Gemma
twin (questions+options stuffed into the encoder — the exact anti-pattern the
benchmark's write-up documents) to reproduce the 17–27% flip bug class as the
contrast model. The external model remains preferred (it's the spec's reference).

## 9. Underfit contingency (pre-named refinement)

If LoRA stalls at the majority floor on val after full budget (720 cases is
small), the pre-named single-variable move is opening the full-fine-tune ablation
slot (spec already reserves it). VRAM budget for full FT ≈ 5.5 GB — fits 12 GB.

## 10. Dangling criterion "5." **[OPEN — user input needed]**

The spec says "ALL FOUR hold" but lists an empty item "5." after criterion 4.
Treated as a truncation artifact; the gate runs on criteria 1–4. Proposed
candidate if a 5th was intended: *unseen-workflow post-TS ECE < 0.12*
(calibration must survive the domain shift, not just in-domain). Confirm or supply.

## 11. "identic" reading **[OPEN — confirm]**

Read as **pydantic** (config + results-row validation). If a different library
was meant, correct this and the roadmap updates in one patch.

## 12. License: MIT (applied)

MIT chosen over GPLv3: HF/ML research norm, maximal adoption for weights-adjacent
tooling, no copyleft friction for people building decision stacks on top.
**[OPEN]** — say the word and it becomes GPLv3 in one commit.

## 13. Metrics matched to the benchmark's official scorer

Accuracy / KL-from-gold / Brier / ECE computed exactly as
`bench/score_with_jobs.py` in the dataset repo defines them (extracted into
`data_card.json` at G1-D1) so our numbers are leaderboard-comparable. RPS,
score-MAE, AUC, jitter-flip, invariance, ablation and speedup are our additions.

## 14. Option-position bias diagnostic (report-only improvement)

Train randomizes choice-option order; eval reports mean logit per option index
to prove no residual position bias survived. Report-only, no gate.

## 15. Score-MAE definition

Expected score vs gold expected score (gold distributions are soft): MAE over
`score`-type questions of |Σ_k k·p_k − Σ_k k·y_k|.

## 16. Tracking & artifacts (box law)

MLflow (sqlite) is the PRIMARY tracker; `runs/` JSON secondary; every results
number rendered from artifacts by script. Best-checkpoint-per-epoch + resume;
`num_workers=0` loaders (this box's DataLoader worker IPC dies ENOMEM).
