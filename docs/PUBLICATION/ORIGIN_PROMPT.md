# ORIGIN — The prompt that started this thread (verbatim, user-provided)

> Preserved 2026-11-27 per the user's no-dilution directive: "We have to actually
> make it publication worthy and good communication — for that we have to know what
> we did at every step and report honestly." This document is the paper's Section-2
> ground truth. Nothing below is paraphrased.

Date of the kickoff: 2026-11-23 (planning run, repo hr1juldey/drishyam-1, public, MIT).

# TASK: Gate 1 — Decoupled T5Gemma-2-270M System-1 Decision Engine (`decoupled_t5g2`)

## 0. Scope & Hygiene
- Create a clean workspace (`t5g2_decision/`, `t5g2_results.jsonl`, `t5g2_progress.md`). Do NOT modify archived JLE or ELECTRA files.
- **Base model:** `google/t5gemma-2-270m-270m` (18 enc / 18 dec layers, d=640, tied embeddings, merged decoder attention).
- **Gate 1 focus:** Text-only architectural proof first (freeze/bypass vision encoder in Gate 1; multimodal unlocks in Gate 2 once text decoupling passes).
- **ROCm sanity gate first:** Before any full run, verify 20 training steps of `google/t5gemma-2-270m-270m` in `bf16` with `torch.optim.AdamW` are 100% finite (grad norm finite, loss decreasing). If `torch.optim.AdamW` NaNs on ROCm, fall back to our plain-PyTorch `ManualAdamW` (`eps=1e-6`).

## 1. Datasets & Zero-Leakage Unseen-Task Split
Use `LocalLLaMA/typed-decisions` (4 workflows: invoice processing, security incidents, customer service, agent-trace observability; primitives: `noul`, `choice`, `score`) as the primary Gate 1 benchmark:
1. **Inspect & verify schema first:** Print workflow counts, primitive counts (`noul`/`choice`/`score`), and 3 sample rows.
2. **Two evaluation regimes (strictly separated):**
   - **In-Domain Split:** Stratified train / val / test across 3 workflows (e.g., invoice, security, customer service).
   - **Zero-Shot Unseen-Workflow Split (The Real Test):** Hold out the **4th workflow completely** (e.g., `agent-trace observability`) from training and calibration. Never train on it. This directly measures unseen-task generalization where encoder-stuffed baselines collapse.
3. **Stress Suites (built deterministically on test):**
   - **Case/Whitespace Jitter (`n=300`):** Randomly capitalize 30% of words + add trailing newlines without changing meaning (tests the 17–27% flip bug).
   - **Multi-Question Latency & Invariance (`n=100` states with 4–6 questions each):** Run Q1 alone vs. Q1 batched with Q2..Q6 sharing cached `H_enc`. Verify max probability delta == `0.0000` (exact invariance) and measure wall-clock speedup over re-encoding per question.

## 2. Decoupled Architecture (`DecoupledT5GemmaDecision`)
1. **State-Only Encoder:**
   - Input: Raw state text ONLY (`[STATE]`). Zero questions, zero options.
   - Output: Cached encoder hidden states `H_enc` (`[B, L_state, 640]`).
2. **Parallel Non-Autoregressive Decoder (1 forward pass per question):**
   - Input sequence to decoder:
     `Question: {q_text} | Type: {primitive} | Options: {opt_1_text} <slot> {opt_2_text} <slot> ...`
   - (For `noul`, options are `No <slot> Yes <slot>`; for `score`, options are the ordered rubric levels `L_1 <slot> ... L_K <slot>`).
   - **Critical detail:** Place each `<slot>` token **after** its option's full description so causal decoder attention reads the option text before scoring.
   - **Merged-attention cross-attention:** Pass `encoder_hidden_states=H_enc` into the T5Gemma 2 decoder so every question/option token cross-attends directly to the cached state.
3. **Unified Calibrated Head:**
   - Extract final decoder hidden states at every `<slot>` position -> LayerNorm -> 2-layer MLP (`640 -> 320 -> 1`) producing one scalar logit per option.
   - **Optional hybrid residual (test in ablation):** Compare pure slot-MLP logit vs. slot-MLP + frozen LM-head prior.
   - Post-hoc **per-primitive temperature scaling** (`T_noul`, `T_choice`, `T_score`) fitted strictly on the **val** split (minimizing NLL).

## 3. Training & Anti-Shortcut Recipe
- **Strictly Proper Scoring Loss:**
  - `choice` / `noul`: `CE(logits, y) + 0.5 * Brier(softmax(logits), one_hot(y))`
  - `score` (ordinal): `CE(logits, y) + 0.5 * RPS(softmax(logits), y)` (Ranked Probability Score over cumulative distributions).
- **Anti-Shortcut Augmentations (applied on train only):**
  - Randomize option order on every step for `choice` (never for ordinal `score`).
  - Random case jitter (20% of steps: randomly uppercase 15–30% of state words) + random whitespace/newline jitter (20% of steps).
- **Optimization:**
  - Start with **LoRA (`r=16, alpha=32`) on encoder + decoder attention/MLP projections + full training on the slot head** (fast, prevents catastrophic forgetting of Gemma 3's pretrained representations on small data). Keep an ablation slot for full fine-tune if LoRA underfits.
  - `bf16`, AdamW (`wd=0.01`), 10% warmup + cosine decay, `lr in [1e-4, 3e-4]` screen on seed 42, early stopping on val **proper-scoring loss (NLL + Brier)**.

## 4. Mandatory Diagnostic & Reference Baselines
Run on the exact same test + unseen-workflow splits:
1. **Baseline 1 (Trivial):** Per-question majority class & TF-IDF + calibrated LogReg.
2. **Baseline 2 (External Reference, if HF download works cleanly):** Zero-shot eval of `ktcapraz/t5gemma2-decision-270m-v2` on the exact same test, unseen-workflow, case-jitter, and multi-question invariance suites.
3. **Anti-Dead-Weight Ablation (Mandatory):** On our trained `DecoupledT5GemmaDecision`, replace `H_enc` with a zero/blank-state `H_enc` at inference and log `state_ablation_flip_rate` and accuracy drop. This proves the decoder is genuinely reading the encoder state.

## 5. Gate 1 Go/No-Go Bar
Print a clean comparison table (`In-Domain Acc / Brier / ECE / Score MAE`, `Unseen-Workflow Acc / AUC / ECE`, `Case-Jitter Flip %`, `Cross-Question Max Delta`, `State-Ablation Flip %`, and `Cached Multi-Q ms/query`).

Declare Gate 1 a **GO** (unlocking Gate 2: full multi-dataset training + native 896x896 multimodal vision decisions + Banking77 >70-option scaling) only if ALL FOUR hold:
1. **Real Decoder Cross-Attention:** `state_ablation_flip_rate > 30%` and real-state accuracy beats blank-state by `> +0.15`.
2. **Exact Structural Invariance:** Cross-question max probability delta == `0.0000` with measurable multi-question cache speedup (`> 1.5x` at 5 questions/state).
3. **Robustness & Calibration:** Case-jitter flip rate `< 5%` (vs. `17–27%` in encoder-stuffed T5Gemma) AND monotonic reliability curve (no top-bin accuracy inversion like ELECTRA) with post-TS `ECE < 0.08` in-domain.
4. **Unseen-Workflow Generalization:** Beats trivial baselines substantially on the held-out unseen workflow (`AUC > 0.75` and positive skill over majority).
