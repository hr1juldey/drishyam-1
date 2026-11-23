# ROADMAP — Gate 1: Decoupled T5Gemma-2-270M System-1 Decision Engine (CUDA)

Machine: RTX 3060 12 GB (compute cap 8.6, driver 580.178.04, native bf16).
Original spec was ROCm-worded; all deltas are recorded in `docs/DECISIONS.md`.

**Completeness index — sub-task IDs:**
`G1-E0 G1-S0 G1-D1 G1-D2 G1-D3 G1-M1 G1-T1 G1-C1 G1-V1 G1-B1 G1-B2 G1-G1`
(a missing ID here = plan corruption; stop and restore before executing).

Workspace mapping (original spec names → this repo):
`decoupled_t5g2` → `src/drishyam/` · `t5g2_results.jsonl` (spec literal) →
`results/drishyam_results.jsonl` · `t5g2_progress.md` → `docs/PROGRESS.md` +
`docs/EXEC_STATE.json` (machine state, per `docs/EXEC_ENGINE.md`).

## Standing conventions (bind every sub-task)

1. Python ONLY through uv: `uv sync` once, then `uv run python ...`. Never bare
   `python3`/system Python. Enforced by `ast-grep scan` (`rules/`).
2. ruff must pass before any commit; ast-grep structural pre/post-check around
   every code edit; diff-based edits (`patch`), never blind whole-file rewrites.
3. Every config and every results row is a pydantic model (`drishyam.schemas`).
4. Training: `num_workers=0` loaders (this box's worker IPC dies ENOMEM), MLflow
   tracking at `sqlite:///mlflow.db` (PRIMARY; runs/ JSON secondary), best
   checkpoint per epoch + resume-from-checkpoint, smoke run before any full run,
   exit codes captured honestly (`EXIT=${PIPESTATUS[0]}`).
5. Docs that report numbers are rendered from artifacts by script; a missing
   artifact renders `MISSING`, never a placeholder or 0.
6. GPU timing always wraps `torch.cuda.synchronize()`; invariance/ablation evals
   run with deterministic algorithms enabled; training stays nondeterministic.
7. `bf16` everywhere (SM86 native; no fp16, no GradScaler);
   `attn_implementation="sdpa"`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
   TF32 on for any fp32 matmuls.
8. Never modify archived JLE/ELECTRA files. Dataset is downloaded at runtime
   (Apache-2.0) and never vendored.

---

G1-E0 Bootstrap exec engine + environment
Goal: lockfile + venv, device probe, amnesia-proof loop engine.
Files: `uv.lock`, `.venv/`, `src/drishyam/__init__.py`,
`src/drishyam/probe.py`, `src/drishyam/schemas.py` (pydantic models:
RunConfig, DecisionRow, GateRow), `scripts/next_step.py` (modes: default/
`--done/--fail/--verify/--init` per `docs/EXEC_ENGINE.md`), `tests/test_engine.py`
(adversarial engine self-tests: stale-state, orphan IDs, parked-flap).
Commands: `uv sync`; `ast-grep scan` (green + fires on fixture);
`uv run python -m drishyam.probe` (prints device name, SM 8.6, bf16 OK,
torch/CUDA versions); `uv run pytest tests/test_engine.py -q`.
PASS: probe prints SM≥8.0 + bf16 supported; engine `--verify` green (V1–V5);
ruff green; rules fire on `tests/fixtures/bad_shell.sh`.
ON-FAIL: fix import/lock issues (mechanical); 3 fails → PARK.

G1-S0 ROCm→CUDA sanity gate (spec §0, CUDA edition)
Goal: prove 20 training steps of `google/t5gemma-2-270m-270m` (text path, bf16,
`torch.optim.AdamW`) are 100% finite with decreasing loss.
Files: `src/drishyam/sanity_gate.py`, `artifacts/sanity_gate.json`,
MLflow run `g1-s0`.
Protocol: load model bf16 on cuda, synthetic batch (24 tokens, B=4), 20 AdamW
steps (lr 1e-4, wd 0.01); assert per-step `torch.isfinite` on loss + every grad
norm; PASS requires mean loss steps 16–20 < mean loss steps 1–5 AND zero NaN/Inf.
Write JSON (step, loss, grad_norm, finite flags, device, torch version) + MLflow.
Second arm with `ManualAdamW` (eps=1e-6) runs in the same invocation and must
match (both arms recorded; either passing satisfies the gate).
PASS: `artifacts/sanity_gate.json` shows 20/20 finite steps + decreasing loss for
the primary arm.
ON-FAIL: if primary NaNs but ManualAdamW passes → primary=ManualAdamW recorded,
continue; if both NaN → PARK (hardware/toolchain issue, human decides).

G1-D1 Dataset inspection (spec §1.1)
Goal: verify schema + counts against the dataset card; extract official scorer's
metric definitions for comparability.
Files: `src/drishyam/data_inspect.py`, `artifacts/data_card.json`.
Protocol: `load_dataset("LocalLLaMA/typed-decisions", <config>)` per workflow +
`all`; print + record: per-workflow train/test counts (card says 300/100, 5
questions/case), per-primitive counts (`noul`/`choice`/`score`) per split, 3 full
sample rows per workflow (state excerpt, questions, gold), column inventory
(`state`/`questions`/`gold` JSON strings + flat `__label`/`__probabilities`/
`__score`/`__probability_true`; `factors`/`label_agreement` = NOT model input),
sha256 of each parquet. Fetch `bench/score_with_jobs.py` from the dataset repo
and record its exact accuracy/KL/Brier/ECE formulas in `data_card.json`.
PASS: counts match card exactly; JSON schema-valid; sample rows printed.
ON-FAIL: count mismatch → STOP, dataset changed upstream → surface to user.

G1-D2 Zero-leakage splits (spec §1.2)
Goal: in-domain train/val/test + fully held-out unseen workflow.
Files: `src/drishyam/splits.py`, `splits/split_v1.json`.
Protocol: seed 42. In-domain = invoice_processing, security_incidents,
customer_service: per workflow, stratified (by question primitive mix + gold
argmax) carve train 240 / val 60 from the official `train` 300; test = official
`test` 100. Unseen = `agent_trace_observability`: its `test` 100 cases only; its
`train` parquet is NEVER loaded (loader asserts the path is never opened).
Save indices + seed + parquet sha256s; assert: unseen indices ∩ (train∪val∪test)
= ∅; per-primitive label distributions reported for every split.
PASS: JSON valid; disjointness + never-loaded asserts green; counts
{3×(240/60/100)} + {100 unseen} recorded.
ON-FAIL: stratification impossible (too few of a class) → relax to plain
stratified-by-workflow, record deviation in JSON.

G1-D3 Stress suites (spec §1.3)
Goal: deterministic case/whitespace jitter + multi-question invariance suite.
Files: `src/drishyam/stress.py`, `artifacts/jitter_suite.jsonl` (n=300),
`artifacts/multiq_suite.jsonl` (n=100 states).
Protocol: jitter — seed 1337, from in-domain test: capitalize 30% of words +
append trailing newlines; case/whitespace-only transforms (meaning preserved by
construction); record (case_id, question, clean_text, jittered_text).
multi-question — 100 unseen-workflow-independent states drawn from in-domain
test, each with its native 5 questions (dataset gives exactly 5/case; spec said
4–6 — deviation recorded in DECISIONS.md); record Q1 vs Q1..Q5 groupings.
Determinism proof: two regenerations byte-identical (sha256 compare).
PASS: n counts exact; determinism check green.
ON-FAIL: n/a (pure construction).

G1-M1 Decoupled model (spec §2)
Goal: `DecoupledT5GemmaDecision` + slot token + head + LoRA attach.
Files: `src/drishyam/model.py`, `src/drishyam/tokenize.py`,
`tests/test_model_shapes.py`.
Protocol:
- Encoder: state text ONLY, prefix `[STATE]`; output cached `H_enc`
  `[B, L_state, 640]` (`encoder_outputs=BaseModelOutput` reuse; vision tower of
  the multimodal checkpoint never instantiated).
- Slot token: add dedicated `<slot>` special token, `resize_token_embeddings`
  BEFORE LoRA attach (tied embeddings resize together); config records the
  choice; sentinel-reuse kept as an ablation slot (DECISIONS.md §3).
- Decoder input per question: `Question: {q} | Type: {primitive} | Options:
  {opt_1} <slot> {opt_2} <slot> ...` — `<slot>` AFTER its option text so causal
  attention reads the description before scoring. `noul` → `No <slot> Yes <slot>`;
  `score` → rubric levels in order, never shuffled.
- Head: final decoder hidden states at slot positions → LayerNorm → MLP
  640→320→1 → one logit per option. Optional hybrid residual (frozen LM-head
  prior) behind a flag, default OFF.
- LoRA r=16 α=32 dropout 0.05 on encoder+decoder q/k/v/o + gate/up/down +
  decoder cross-attention (`enc_dec_attn`) projections; embed/lm_head/vision
  excluded; slot-head trained full.
Commands: `uv run pytest tests/test_model_shapes.py -q` (one fwd/bwd on synthetic
batch; shapes; all-finite; trainable-param count; peak VRAM logged).
PASS: shapes correct (slots per question = n_options), finite, VRAM < 4 GB in
LoRA mode; parameter report written to `artifacts/model_profile.json`.
ON-FAIL: sdpa + added-token interaction bug → retry with `attn_implementation=
"eager"` recorded as deviation; 3 fails → PARK.

G1-T1 Training (spec §3)
Goal: LoRA training with proper-scoring losses + anti-shortcut augs.
Files: `src/drishyam/train.py`, `src/drishyam/losses.py`
(CE+0.5·Brier; CE+0.5·RPS per spec formulas), `src/drishyam/augs.py`,
`runs/` (MLflow sqlite + checkpoints).
Protocol: smoke first (subset 64 cases, 2 epochs — GPU-box law), then full:
batch 8 cases (questions of a batch packed as decoder-batch sharing H_enc), bf16
autocast, AdamW wd 0.01, 10% warmup + cosine, lr screen {1e-4, 3e-4} seed 42
(both logged; better val picked), early stop patience 5 on val proper-scoring
loss (NLL + Brier). Augs train-only: choice option order randomized every step
(never for score), case jitter 20% of steps (15–30% of state words uppercased),
whitespace jitter 20%. MLflow: params + step-indexed metrics + checkpoint
artifacts; best-checkpoint-per-epoch + resume (crash = resume, not re-run).
PASS: smoke artifact shows finite decreasing loss + VRAM print; full run
artifact has config hash, git SHA, seed, per-epoch val NLL/Brier, peak VRAM.
ON-FAIL: OOM → halve batch, record; underfit (val ≈ majority floor after full
budget) → full-finetune ablation slot opens (DECISIONS.md §9); 3 fails → PARK.

G1-C1 Calibration (spec §2.3)
Goal: per-primitive temperature scaling fitted on val, frozen for unseen.
Files: `src/drishyam/calibrate.py`, `artifacts/temps.json`.
Protocol: fit `T_noul`, `T_choice`, `T_score` minimizing val NLL
(L-BFGS on log T, grid+refine fallback); report val NLL before/after + post-TS
val ECE. Temperatures are fitted ONLY on the 3 seen workflows' val and applied
UNCHANGED to the unseen workflow (honest zero-shot calibration; DECISIONS.md §5).
PASS: temps.json schema-valid; val NLL improves; post-TS val ECE recorded.
ON-FAIL: n/a.

G1-V1 Evaluation suites on OUR model (spec §4 diagnostics + §5 table inputs)
Goal: every diagnostic, every number the gate table needs.
Files: `src/drishyam/evaluate.py`, `results/drishyam_results.jsonl`
(one pydantic-validated row per decision), `artifacts/gate_inputs.json`.
Suites (exact definitions in DECISIONS.md §4): in-domain test (300 cases × 5);
unseen workflow (100 × 5; TS applied unchanged); case-jitter flip (n=300: argmax
change clean vs jittered); multi-question invariance (n=100 states: Q1 alone vs
batched Q1..Q5 sharing cached H_enc; report max|Δp| raw and @4dp and bitwise-
identical fraction; padded-shape eval mode per DECISIONS.md §6); state ablation
(zero `H_enc` at inference: flip rate + accuracy drop); speedup (CUDA events,
10 warmup + 50 reps, p50/p95: per-question full enc+dec ×5 vs 1 enc + batched
5 dec on cache). Metrics mirror the official scorer (accuracy, KL, Brier, ECE
15-bin) + reliability curve artifact + automated top-bin inversion check +
option-position bias report (report-only).
PASS: all suites present in gate_inputs.json; row count = decisions evaluated;
schema validation green; every metric reproducible by re-run.
ON-FAIL: invariance raw Δ > 5e-5 → structural bug (fix, not tolerance);
ablation flip < 30% pre-gate → investigate decoder wiring BEFORE G1-G1 verdict.

G1-B1 Baseline 1 — trivial (spec §4.1)
Goal: per-question majority class + TF-IDF + calibrated LogReg on identical
splits.
Files: `src/drishyam/baseline_tfidf.py`.
Protocol: word(1-2)+char(2-5) TF-IDF on state+question text, `LogisticRegression`
+ CalibratedClassifierCV (isotonic on val), majority-class per question-id;
same splits, same metrics, results appended to `results/drishyam_results.jsonl`.
PASS: rows present for in-domain + unseen; skill vs majority computed.
ON-FAIL: n/a.

G1-B2 Baseline 2 — external reference (spec §4.2)
Goal: zero-shot eval of `ktcapraz/t5gemma2-decision-270m-v2` on the exact same
test, unseen, jitter and invariance suites.
Protocol: confirmed to exist (downloads=61; custom `decisions/` engine +
`decision_config.json`, v4-joint-voice layout, per-primitive temps pre-fitted).
Load per ITS repo instructions (trust_remote_code in an isolated step; never
vendored into our src/). If its interface cannot be driven in ≤1 parked attempt
→ documented fallback: train our own encoder-stuffed T5Gemma twin (same data,
same budget: questions+options concatenated into the ENCODER input) to
reproduce the bug class, used as the contrast model (DECISIONS.md §8).
PASS: reference rows on all four suites OR fallback artifact + logged reason.
ON-FAIL: PARK after 2 attempts; fallback activates.

G1-G1 Gate verdict (spec §5)
Goal: render the gate table + GO/NO-GO, from artifacts only.
Files: `src/drishyam/render_gate.py`, `docs/RESULTS.md`.
Protocol: render table rows: In-Domain Acc/Brier/ECE/Score-MAE; Unseen Acc/AUC/
ECE; Case-Jitter Flip %; Cross-Question Max Δ; State-Ablation Flip %; Cached
Multi-Q ms/query; then verdict against criteria 1–5 (criterion 5 added by user 2026-11-23 —
DECISIONS.md §10): (1) state_ablation_flip_rate > 30% AND real−blank acc >
+0.15; (2) max Δp == 0.0000 @4dp (padded fixed-shape inference mode) AND
speedup > 1.5× at 5 questions; (3) jitter flip < 5% AND monotonic reliability
(no top-bin inversion) AND post-TS in-domain ECE < 0.08; (4) unseen AUC > 0.75
AND positive skill over majority; (5) unseen-workflow post-TS ECE < 0.12
(temperatures fitted on seen workflows only, applied unchanged).
Any MISSING artifact ⇒ verdict = NO-GO-INCOMPLETE (fail closed).
PASS: RESULTS.md byte-reproducible from artifacts; verdict printed.
ON-FAIL: n/a (verdict is computed, not argued).

---

## Execution order & dependencies

E0 → S0 → D1 → D2 → D3 → M1 → T1(smoke→full) → C1 → V1 → {B1, B2} → G1.
B1/B2 may run in parallel with V1 after C1. Estimated wall-clock on the 3060:
S0 < 2 min · D* < 10 min · M1 < 10 min · T1 full ≤ 1 h (small data: 720 train
cases × 5 questions) · V1+B* < 30 min · G1 instant.

## Gate 2 unlock (recorded, not scheduled)

Full multi-dataset training · native 896×896 multimodal vision decisions ·
Banking77 >70-option scaling. Requires G1-G1 verdict = GO.

## Gate 2 preview sub-tasks (recorded 2026-11-23, NOT scheduled — outside the G1 engine)

G2-TF1 Overnight teacher data factory: Gemma-431B offline batch generation,
minimized-reasoning structured output, verify/filter gate, leakage guardrails
per DECISIONS §20. Spec exists; no PASS criterion yet — specified at Gate 2 open.
G2-V0 Browser image pipeline: Firefox capture → 896×896 vision states →
teacher-labeled pairs, per DECISIONS §21. Spec exists; no PASS criterion yet.

These IDs are NOT in the Gate-1 completeness index above and are never
returned by `next_step.py` during Gate 1.
