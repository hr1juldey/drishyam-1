# DRISHYAM GATE-2.5 — STANDALONE CHECKPOINT REPORT
Written 26 Nov 2026, 00:30 IST. Assumes the reader knows NOTHING about this project.

## 1. What this model is
A ~270M-parameter **document-page decision model**. Input: one synthetic document
page (banked as 256 embedding tokens, 640-dim, from a frozen vision tower) plus a
typed question with candidate options. Output: a **probability distribution over
the options** — it answers questions ABOUT the page with calibrated confidence,
it does not generate text. Questions come in three primitives:
- **noul** (2 options): yes/no legibility questions ("Is the account number readable?")
- **choice** (2–5 options): pick-the-right-candidate questions ("Which institution issued this?")
- **score** (5 ordinal options): quality grades ("Rate the print legibility L1..L5")

The purpose (path-B doctrine): a cheap, calibrated *selector* that routes/verifies
documents — NOT a generator. Generative transcription was measured and closed
earlier (a 270M generative model transcribes digits at ~1–1.7% accuracy).

## 2. Data (all synthetic, deterministic factory, zero leakage)
- 90,000 document pages banked as cached embeddings (89,976 verified:
  thermal 36,000 / card 18,976 / savings 15,024 / electricity 8,000 /
  salary 6,000 / telecom 6,000; zero `table` pages — generation closed at 90k).
- Decision labels derived from the factory's ground truth (same gt chain that
  rendered the pages — labels are exact by construction, no human labeling):
  **train 77,022 / val 10,282 / test 15,393 / unseen-family 3,000 decisions**
  (15,000 / 2,000 / 3,000 / 600 pages).
- Leave-one-family-out: **salary** is the unseen family (never in train/val/TS).
  Train sees 5 families; unseen tests style transfer.
- Per-primitive train mix: noul 47,211 / choice 26,811 / score 3,000.
- Files: `artifacts/gate2_labels_{train,val,test,unseen}.jsonl`,
  stats+hash in `artifacts/gate2_label_stats.json` (sha 3c6be4b3e037).

## 3. Architecture (the load-bearing story)
**Base:** `google/t5gemma-2-270m-270m` (T5-style stack: bidirectional encoder +
cross-attending decoder). Snapshot cached, HF_HUB_OFFLINE=1.
**Vision:** frozen SigLIP-style tower, offloaded to CPU during training; pages
enter as **cached embedding tokens** (never re-encoded — this is why per-question
rows are affordable).

**Pathway v1 (DEAD, formally):** page embeddings entered the DECODER via
cross-attention; question+options were decoder tokens. After 4 epochs (lr 1e-4,
LoRA r16 all-linear): val acc 0.4157–0.4612 ≈ chance, val CE 3.46–4.0
(below the ln(k) guessing baseline = confidently wrong), ECE 0.36–0.41,
grad-clip hit on 49% of steps, memorize/tear oscillation (train 0.08↔0.5).
Root cause: binding never happened — the decoder ignored page content
(independently corroborated by the ktcapraz/t5gemma2-decision-270m-v2 model
card: "the encoder makes the answer; the decoder does not read the text", 0%
flip).

**Pathway v2 (CURRENT):** encoder-row binding (`forward_row`). Each QUESTION is
one encoder row: `inputs_embeds = [page_embs(256) ; question+option tokens(≤65)]`,
slots read from the ENCODER output at slot positions shifted by the page
segment. The question lives INSIDE pretrained self-attention with the page —
the textbook-correct joint-encoding design (independently confirmed by the
BEV-Decider-from-scratch video and the ktcapraz card).
**Head:** same small slot decision head as v1 (reads slot positions → per-option
logits), loss = per-primitive cross-entropy (`question_loss`).
**PEFT:** LoRA r16 all-linear, FRESH init (not warm-started from v1).

## 4. Training (current run, live)
- `train_gate2_row.py`, commit 07aa5f6+2427990. RTX 3060 12 GB, bf16, grad
  checkpointing, expandable_segments. Peak VRAM 2,219 MB stable.
- lr **2e-5** (v1's 1e-4 was measured too hot: 49% clip + oscillation),
  AdamW wd 0.01, cosine schedule, 10% warmup, batch = 16 question-rows.
- 4,814 steps/epoch × 4 epochs max, **patience 2**, best.pt by val loss.
- Per-question rows: 77,022 rows/epoch, ~71 min/epoch on the 3060 (~19 rows/s).
- MLflow: experiment `drishyam-doc`, run `g2-row-sft` (live uuid 8f27b2b5…);
  per-step train_loss/grad_norm/rows_per_s/vram_mb, per-epoch val metrics.

### Measured so far (val = 10,282 held-out decisions, 2,000 pages)
| epoch | train CE | val CE | val acc | brier | ECE(15) |
|---|---|---|---|---|---|
| ep0 | 0.684 | 0.743 | **0.7987** | 0.306 | 0.047 |
| ep1 | 0.419 | 0.485 | **0.8890** | 0.172 | 0.046 |
| ep2 | in progress (train CE hit 0.017 = memorization floor; guards armed) |

For contrast, pathway v1's BEST epoch ever: acc 0.4612, CE 3.46, ECE 0.36.
Chance floors: noul ln2=0.69 CE, choice ln3=1.10, score ln5=1.61.

### Checkpoint forensics
- `runs/g2-row-sft/best.pt` (ep1, 1.60 GB) and `last.pt` on disk.
- Zero-shot diagnostic (v1 weights through the row pathway, no row training):
  acc 0.4839 — the pathway binds even untrained; training unlocks it.
- 300-page falsification smoke: 0.4839 → 0.6079 in 2 epochs (bar >0.60, PASS,
  pre-registered before running). Full-scale v2 then confirmed at 0.80+.

## 5. Formal evaluation (pre-registered, NOT edited after results)
Six bars in `eval_gate2.py` (run: `eval_gate2 --ckpt runs/g2-row-sft/best.pt --path row`):
1. **g2_1 ablation flip > 0.30** — blank the page, answers must change (reads the page)
2. **g2_2 multi-q delta == 0.0000 @4dp** — same state, any batch shape, identical probs
3. **g2_3 post-TS ECE < 0.08** — temperature scaling fitted on val, applied at eval
4. **g2_4 unseen-family AUC > 0.75** — salary noul, never seen anywhere in training
5. **g2_5 jitter flip < 0.05** — question case/whitespace perturbation
6. **g2_6 decisions/s** — throughput (informational)
fp32 eval law: eval runs `model.float()` + fp32 pages (bf16 GEMM order flips near-ties).
v1 results (formal, 25 Nov): g2_1 0.150 FAIL / g2_2 0.0 PASS / g2_3 0.197 FAIL
(T pinned at ceiling 5.0) / g2_4 0.174 FAIL (below coin flip) / g2_5 0.047 PASS.
**v2 verdict: PENDING — eval auto-runs after training exits (~03:30 IST).**

## 6. Infrastructure & reproduction
- Repo `~/Documents/code/drishyam-1`, branch main. Key commits: 44ada8f (K1
  forward_row + smoke), 07aa5f6 (row pipeline), 2427990 (OOM fix), 6fc08f7
  (playbook), plus v1 line: RLCD stage, LOFO fix, tracking fix.
- Reproduce training: `HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  uv run python -m drishyam.train_gate2_row --lr 2e-5 --epochs 4 --rows-per-batch 16 --out runs/g2-row-sft`
  (resume: `touch runs/g2-row-sft/resume_flag` first).
- Reproduce eval: `... uv run python -m drishyam.eval_gate2 --ckpt runs/g2-row-sft/best.pt --path row`
- MLflow UI: `uv run mlflow ui --backend-store-uri sqlite:///$PWD/runs/mlflow.db
  --host 127.0.0.1 --port 5000` (running; experiment drishyam-doc).
- Logs: `runs/overnight/g2_row_sft.log`, `g2_eval.log`, `g2_row_smoke.log`.
- Failure playbook (pre-committed, per-branch actions with named mechanisms):
  `docs/GATE25_MORNING_PLAYBOOK.md`. Pre-registration contract:
  `~/.hermes/plans/2026-11-25_144900-gate2-vision-decision.md`.

## 7. Known warts (measured, not hidden)
- K3 label wart: light-degradation `score` pages carry "L5 clean" gold
  (consistent train/val — cannot explain tonight's numbers, but a v3-label fix).
- `score` questions are only 3,000/77,022 train rows (3.9%) — per-primitive
  scores will be lopsided; check per-primitive in the morning eval.
- External GPU neighbor: `histogate-serve` (232 MiB, the user's own API server —
  never kill it). One earlier OOM was caused by a val pass missing `no_grad`
  (fixed: inference_mode + pre-val checkpoint save, commit 2427990).
- Fresh-init run at ep2 has train CE at the memorization floor (0.017); val
  guards (best.pt-by-val-loss, patience-2) are armed; late-epoch checkpoints
  can only lose to best.pt, never poison it.

## 8. Open decisions (owner: user, on waking)
1. Read the six-bar verdict; follow the playbook branch it selects.
2. If g2_1/g2_4 lag: K5 counterfactual pairs (pre-registered; same page, one
   gt quantity flipped so gold flips; factory-only, ~20-min re-derivation).
   Rule: every data addition must name the failing bar it targets.
3. If val was still climbing at the last epoch: resume 2 epochs (mechanism:
   schedule too short), not more data.
4. Calibration polish (if needed): RLCD stage is built + TDD'd
   (`train_gate2_rlcd.py`, proper-scoring rewards, abort rails), SFT ckpt untouched.
5. Robustness confirmation on pass: second holdout unseen=telecom.
