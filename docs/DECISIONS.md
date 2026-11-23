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
attach; the new row trains. Sentinel-reuse is kept as a named ablation. CONFIRMED by user (2026-11-23).

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
**CONFIRMED by user (2026-11-23): the padded fixed-shape mode IS the shipped
inference mode; the gate reads its exact 0.0000 and the raw unpadded Δ is
published alongside.**

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

## 10. Criterion 5 — RESOLVED (user, 2026-11-23)

The spec's dangling empty "5." is now a real gate: **unseen-workflow post-TS
ECE < 0.12** (temperatures fitted on the seen workflows only, applied
unchanged) — calibration must survive the domain shift, not just in-domain.
Gate 1 runs on criteria 1–5; ROADMAP G1-G1 carries the full list.

## 11. "identic" reading — RESOLVED (user, 2026-11-23)

**pydantic** confirmed: validates every config and every results row.

## 12. License: MIT — CONFIRMED (user, 2026-11-23)

MIT stays. (Rationale on record: HF/ML research norm, maximal adoption for
weights-adjacent tooling, no copyleft friction for downstream decision stacks.)

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

## 17. Rename: decoupled-t5g2 → drishyam-1 (user, 2026-11-23)

User named the repo **drishyam-1** (Sanskrit दृश्यम्, "that which is seen"):
the project becomes a vision decision model in time (Gate 2), and real-time
performance matters beyond vision too. Spec-internal artifact names stay as the
spec wrote them (`src/t5g2_decision/`, `results/t5g2_results.jsonl`, sub-task
IDs `G1-*`) — the spec's own vocabulary now lives inside repo drishyam-1.
No scope change: Gate 1 remains text-only. RESOLVED (2026-11-23, user): the
Python package DID take the project name — `src/drishyam/`, project `drishyam`
in pyproject, `results/drishyam_results.jsonl`; sub-task IDs `G1-*` unchanged.

## 18. Real-time constraint (user, 2026-11-23)

Beyond vision, the engine targets real-time use broadly. This reinforces the
decoupled design (encode once → cache → answer many questions in parallel) and
rules OUT any resident teacher LLM in the serving path. Training-time teacher
signals, if ever needed, are offline batches only (§19).

## 19. Resident teacher LLM in the loop? (user question, 2026-11-23)

**Answer: No for Gate 1 — and never a resident one.**
1. The teacher is already in the data: typed-decisions gold = mean of three
   ~4B-teacher samples; training on those soft distributions with KL/Brier/RPS
   IS distillation from that teacher, with zero hosting cost.
2. The base model is the second teacher: Gemma-3-270M pretrained
   representations, protected by LoRA (optionally reinforced by the frozen
   LM-head prior residual) supply the reasoning prior — that is LoRA's job.
3. A resident Llama-class teacher in the training loop on this box: 4–9B bf16
   resident (8–18 GB) blows the 12 GB budget next to training, slows every
   step, and injects sampling nondeterminism that breaks the reproducibility
   gates (sanity gate, invariance suite, byte-reproducible verdicts).
4. Gate 2 revisit, OFFLINE only: for workflows beyond the benchmark, labels
   are batch-generated by a bigger model → verified → filtered → trained
   (exactly how this dataset itself was built; teacher now identified — user's
   local ollama `gemma4:31b`, see §20). The serving path stays
   teacher-free: real-time System-1 means the deployed artifact is the 270M
   decoupled engine, nothing co-resident.

## 20. Overnight teacher data factory (user, 2026-11-23) — GATE 2 PREVIEW

Teacher CORRECTED by user: **gemma4:31b in local ollama** — my "431 billion"
reading was wrong. VERIFIED 2026-11-23 via `ollama list`: `gemma4:31b` (20 GB)
and `gemma4:31b-16k` (20 GB, 16k context) both present, server live on :11434,
nothing loaded. Default teacher tag: **`gemma4:31b-16k`** (context headroom for
long option lists / Banking77); `gemma4:31b` fallback. Fully offline, zero API
cost.

- OFFLINE batch generation only — never resident in training or serving (§19
  stands unchanged).
- **CO-RESIDENCY LAW (hard):** 31 B @ 20 GB cannot fit the 3060's 12 GB —
  ollama offloads most layers to the 62 GB RAM, so the factory owns the whole
  box. It NEVER runs while drishyam Gate-1 training occupies the GPU; nights
  alternate, or the factory runs after the training day. Default to sequencing,
  never to sharing.
- **Minimized reasoning per call** (user directive): no chain-of-thought; ONE
  structured pass via ollama structured output (`format: json`) returning the
  typed schema directly (noul probability / choice distribution / score
  distribution + expected score), ~300 output tokens per case. 3 samples at
  temp ≈ 0.7, averaged — mirroring the benchmark's own gold recipe.
- Throughput (MEASURED night 1, never promised): mostly-CPU offload ⇒ ~3–8
  tok/s ⇒ ~2–4 min per case incl. 3 samples ⇒ order 120–160 cases per 8 h
  night, single stream; seen-workflow pools (300/workflow) double in ~2–3
  nights. `qwen3:14b` (9.3 GB, fits the card, much faster) is recorded as the
  cross-check/fast teacher candidate — user's call, not planned.
- **Verify/filter gate before any generated case enters training**: pydantic
  schema validation, 3-sample self-consistency threshold, dedupe of normalized
  state text (sha256) against existing states. EVAL GOLD IS NEVER REGENERATED —
  the benchmark's gold stays the only eval truth.
- **LEAKAGE GUARDRAIL (hard), user-confirmed:** teacher data is restricted to
  the THREE SEEN workflows; the held-out workflow and every eval split stay
  pure — the zero-shot claim dies otherwise.
- Domains, user-confirmed: (a) expand the binding constraint — seen-workflow
  train pools of 300 cases/workflow; (b) **Banking77 first** for Gate 2
  (text-only, already in the spec, zero interference with Gate 1); browser UI
  vision states come later, paired with §21.

## 21. Browser image pipeline (user, 2026-11-23) — GATE 2 PREVIEW

User: "attach one browser to get some image pipeline — most of the work is
done." The box already drives a real Firefox with capture machinery. Vision
states for drishyam-1 = captured UI states (dashboard / invoice / console
screenshots at the encoder's native 896×896), teacher-labeled with the same
typed schema → (image state → decisions) pairs for Gate 2's native vision
decisions. Recorded now; built only after the Gate 1 verdict.
