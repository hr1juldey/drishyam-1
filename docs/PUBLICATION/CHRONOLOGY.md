# CHRONOLOGY — drishyam-1, the complete measured arc (2026-11-23 → 2026-11-27)

Reconstruction contract: every row cites its record (commit sha / ledger / mlflow /
doc). Nothing is dropped for being unflattering. [verify] marks cells to confirm
against session history before the paper draft.

## Phase 0 — Origin (2026-11-23 evening)

| step | what | record |
|---|---|---|
| 21:13 | Repo `drishyam-1` created public (MIT); kickoff = the Gate-1 spec (see ORIGIN_PROMPT.md): decoupled T5Gemma-2-270M decision engine on `LocalLLaMA/typed-decisions` (4 workflows, primitives noul/choice/score) | dca405e |
| 21:16 | Gate-1 planning suite: roadmap, exec engine, tooling laws (uv/ruff/ast-grep) | aa7914d |
| 21:29–21:37 | User decisions applied: real-time vision framing; resident teacher = **local ollama gemma4:31b** (GPU co-residency law) | 334d7c2, 9104d39, a3b42b7 |
| 22:11–22:28 | Same-night build: exec engine + adversarial self-tests; schemas, device probe, MLflow tracker; **decoupled model** (slot token, LoRA r16, slot head, fp32 head boundary); data inspection; **zero-leakage splits** (unseen workflow hashed, never loaded); **deterministic stress suites** (jitter n=300, multiq n=100) | 05a60d3…bfb3f3f |
| 22:21–22:39 | Sanity gate PASS (20 steps, finite); real-model smoke PASS (7.8M trainable, VRAM 2.0 GB); strictly proper scoring losses (CE+Brier, CE+RPS) with hand-computed fixtures | d77a503, 86ee350, 6e74ff4, 236de55 |
| 22:48–23:13 | Trivial baselines (unseen AUC 0.491 = no skill); per-primitive TS; external reference `ktcapraz/t5gemma2-decision-270m-v2` evaluated through its own engine: test 0.759/0.919, **jitter flip 5.67% FAIL, multiq invariance FAIL, ECE 0.183 FAIL** — the reference model's weaknesses, measured | 1431fbc, 71eb00e, fa91c32, 5ae69db |

## Phase 1 — Gate 1 verdict night (2026-11-24 00:00–08:30)

| step | what | record |
|---|---|---|
| 00:49 | **VERDICT: NO-GO (0/5)** — in-domain beats the reference (0.792/0.941/0.054 vs 0.759/0.919/0.161) but unseen workflow collapses (acc 0.274, AUC 0.499; blank-state 0.268 ≈ real 0.276 → state ignored off-domain) | 6dde069 |
| 00:54 | fp32 eval law: CPU repro proved batch-invariance exact at fp32; bf16 batch-shape noise broke the delta metric | 5304433 |
| 04:33 | **The invariance "failure" was an eval bug**: q1_batched walked slot scalars, softmax(scalar)=1.0 — the model was exact all along: **100/100 cases, delta 0.0000** (criterion 2 PASS) | 3bd8962, 0d970da |
| 05:08–06:14 | Night-chain engineering: ragged-safe TS fit; teacher coercion (nulls/arrays); resumable queue + idempotent watchdog ("lanes must survive arbitrary kills"); **teacher bake-off**: gemma4:31b KL 1.22 / agree 10.5% vs lfm3-omni KL 1.38 / 0% — both below random argmax → labeling prompt needs iteration | 4dff4e1, a2d3381, b42f02c, a77e1dc |
| 07:23–08:21 | **Teacher audit vindicated**: suspect referees built (fail-closed normalizer, alias table rank-NAME→numeric gold), few-shot added: **KL 1.220→0.223, agreement 65–81%** | 43/44/45 (2cd019b, 935d55d, 33b3c2a) |

## Phase 2 — The Hermes harness + the synthetic factory (2026-11-24 10:35–23:45)

| step | what | record |
|---|---|---|
| 10:35 | **Hermes-driven teacher interface** (user law: "make an interface a hermes agent can drive; direct wiring is not preferred"): work orders + contract + deterministic scoring | 6613edd |
| 10:38 | **First harness exam PASS: GLM-via-Hermes KL 0.1387 (best measured), noul 79.4%, ~436 cases/h** — beats both local teachers | 5492e9a |
| 11:09–12:10 | Contract variant race on seed-42: **v2a wins** (score +15pts, noul +6pts); **h500a final n=500: KL 0.157, score 68.6, noul 85.0, 500/500 valid, 13 waves zero failures** — factory contract locked | b763fe1, f18435e, b3bf158 |
| 17:01–17:49 | **paper_rig synthetic document factory born**: readable ICICI content generator + GPU doc-photo rows; thermal family with smudge/legibility coupled to gold labels; **JIT data pipeline + training-parity proof (PASS +0.8%)**; 200k rows rebuilt; producer/trainer with SigLIP frozen pre-opt; micro-step GREEN | 702dfbc…1a93727 |
| 19:12 | Diversity layer v1: 7 families, multi-institution | 29dec7c |
| 22:39–23:45 | Overnight embedding trainer (decoder-only, projector frozen), LOIO eval, camera web app ($0 positioning, post-CamScanner slot), prefetch deadlock fix, chain stage timeouts, stall watchdog | f537b5d, 100a406, 3af97a0 |

## Phase 3 — Eval-artifact hunts + Gate 2 (2026-11-25 00:00–15:50)

| step | what | record |
|---|---|---|
| 01:29–12:17 | lm_head-logits eval + softcapping; **bos-vs-pad decoder artifact found and fixed in BOTH eval and proof gate** (was producing 0% preds); gap-aware shard resume; **pixel-proof gate: fresh-seed full-pipeline eval vs gt across 7 families**; generate slice-by-[1:] fix; self-contained checkpoints (full processor persisted) | 47e79fc, db5cba0, e517082, a0a8fd7, 037879a, fccec54 |
| 15:01–15:21 | **Gate 2 built**: decision labels from deterministic factory gt; cached-page-token state (encode_state_embeds/forward_embeds); train loop on cached states; **six pre-registered bars** + TS-on-val + **fp32 eval law**; LOFO unseen = salary (zero table pages — measured) | 4c5ebd9…a49c94a |
| 15:39 | RLCD stage designed + reward math TDD (proper-scoring rewards, REINFORCE group baseline, soft-CE anchor, pre-registered abort rails) | 88/89 |

## Phase 4 — Gate 2.5 row pathway + the K-experiments (2026-11-25 19:24 → 2026-11-26)

| step | what | record |
|---|---|---|
| 11-25 19:24 | **K1: forward_row encoder-row binding** — each question is one encoder row over cached page tokens; row smoke PASS | 44ada8f |
| 11-25 19:52 | Row-path full pipeline: train_gate2_row + eval --path row (all six bars through forward_row); **plan execution contract pre-registered** | 07aa5f6 |
| 11-25 (formal v1 row) | **g2_1 0.150 FAIL / g2_2 0.0 PASS / g2_3 0.197 FAIL (T pinned at ceiling 5.0) / g2_4 0.174 FAIL / g2_5 0.047 PASS** | GATE25_CHECKPOINT_REPORT.md §5 |
| 11-25 22:03 | **Morning playbook**: per-failure branches pre-committed before the verdict; "tune and rerun" forbidden | 6fc08f7 |
| 11-26 00:25 | Standalone checkpoint report for a context-free reader | aa9abfb |
| 11-26 08:10–09:19 | **K5 counterfactual presence twins**: omission registry (render+labels one source), salary omit guards, 17 randint inclusive-high crash paths clamped; registry extended to all 8 salary fields; unseen selection forces twins | a4a363c, 9395217 |
| 11-26 09:19 | **JevBench text-row track**: bev-decision loader pinned, text-row trainer init from g2-row head, public-231 evaluator; 16 CPU tests | b57476d |
| 11-26 15:12 | **"The exam is broken, not the student" — the diagnosis, measured**: g2_4 answers were predictable from question wording alone (salary split always-Yes, receipt split always-No). Smoking gun: **v1 and v2 were different architectures (val acc 0.79 vs 0.89) yet scored g2_4 identically to three decimals: 0.174** — two different brains, same wrong number → the wrongness lives in the exam. Banked meta-lesson: *when two differently-built models score identically on a benchmark, the benchmark is the suspect — check the answer-key statistics before blaming the students.* K5 twins designed as the fix; v4 balanced-pair contract pre-registered the same hour. Generative-wall context (why decision-model framing): a 270M generative model transcribes digits at ~1–1.7% accuracy | session 56308 (11-26 15:12), GATE25_CHECKPOINT_REPORT.md:16 |
| 11-26 16:14 | **K5-v4 prep**: omission support for all 4 train families + `--balance-presence` (opt-in, train-only Yes≤2×No cap, qid-sorted determinism) + OMITTABLE_FIELDS contract + gt-parity fixtures | fa70951 |
| 11-26 18:07 | JevBench dict-of-documents crash fix (`_state_text` canonical JSON form); 231/231 verified offline | bebdf8b |
| 11-26 19:09 | **ImajevBench stock OOD eval**: dev 34.10% typed @τ=0 (abstention never wins), Brier 0.989/ECE 0.407 — photo domain measured and CLOSED | 1feba17 |
| 11-26 19:56 | **SROIE verification-v1 evaluator** — first in-distribution PUBLIC image benchmark: 361/400 official receipts (public Task-3 gold), contamination check recorded (0 test-ID overlap), 2,166 rows 50/50 seed 2026, rows digest 373d16ee; **ledger born**: SROIE 51.2 acc / AUC 0.516 / ECE 0.252 — reads real receipts (TRUE-Yes 98.3% dates), cannot reject false claims (2.5%) — the v3 prior lesson on public data | b85a525, a6aff0f |

## Phase 5 — The falsification and the fix (2026-11-27)

| step | what | record |
|---|---|---|
| 07:35 | Async-embed wedge root-caused (deterministic 600 s death) → sync path + expandable_segments; **GPU-util watchdog** built (fires <10% ×5 s while expected-GPU job alive) — user directive: "you constantly monitor… you will get triggered with that cron" | 8daa759 |
| 07:52 | vision_data parallel pass-1, **byte-identical labels proven** vs serial | 5214e8b |
| 12:08 | **v4 verdict**: balanced-pair retrain (17,776 surplus Yes rows dropped) — flip 0.387 best-ever, blank 0.50, unseen AUC ×2.25 vs champion; ECE 0.089 blocks internal promotion | 7bec037 |
| 12:40 | **SROIE v4: PUBLIC FALSIFICATION** — always-No collapse (true-Yes 0.28%), AUC unchanged 0.517: balance flipped the prior, not signal; TS refit cancelled per bridge rule | 24002fe |
| 13:18 | **v5-A built + contract pre-registered**: 625 SROIE-trainval receipts → 3,150/600 rows 50/50, template parity asserted, PASS = AUC ≥0.65 AND rejection ≥0.40 AND internal bars hold | 93234a4 |
| 19:00–19:08 | Two launch crashes fixed in one cycle each: ART-relative label paths (load_cases joins onto artifacts/); eval fixed_len widened to splits' max (ckpt stores training-only length) | c3a7d1f, 3978067 |
| 19:51 | **V5-A VERDICT: PASS — SROIE 95.89% / AUC 0.9878 / ECE 0.0203** (per-field AUC total 0.984 / date 0.966 / company 0.9998); internal bars preserved (flip 0.373, blank 0.505, unseen 0.379, test 0.794); internal ECE 0.1106 caveat disclosed, TS polish in flight | 75a42cd |

## User steering (verbatim, verifiable in session history)

- "if the GPU utilization dies… you will get triggered with that cron so that we don't waste time" → watchdog built (11-27).
- "You did not check current date and time and just committed that things are happening or not" → live-`date`-first law (11-27).
- "Everything started from this prompt… you are actually diluting everything that we did in the middle. We have to actually make it publication worthy… know what we did at every step and report honestly" → this chronology (11-27).
- "if this work is properly done then we can prepare ourselves for publishing this work universally or in a proper place… where we publish is another problem but we have to prepare the data and everything" → publication-prep lane (11-27).
- "let my CPU work on this and the GPU to be turned over to my Chintak" → CPU-only law for all remaining Drishyam work (11-27).

## Open measurement debts (to close before paper draft)

1. ECE-optimal TS refit on CPU (in flight, proc_a6e5882e21e5) → closes the v5-A internal ECE caveat or documents its persistence.
2. ~~Session-history verification of [verify] cells~~ — RESOLVED 11-27 20:2x: generative wall = ~1–1.7% digit transcription (GATE25_CHECKPOINT_REPORT.md:16); exam-flaw discovery = v1≡v2 g2_4 0.174 identical-to-3dp across architectures (session 56308). Chronology cells updated with citations.
3. JevBench 39.8 + ImajevBench 34.1 ledger rows already cite shas (bebdf8b, 1feba17) — confirmed in b85a525 message.
4. ~~HF publish of v5-A~~ — DONE + VERIFIED 20:22: remote listing carries v5a/best.pt + 3 tables + chronology README (commits 350194cd / f9de966a / 30e8b394 / 98121e1b / 7718a13e); v2 files preserved alongside.
