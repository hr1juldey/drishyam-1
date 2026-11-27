# From Coin-Flip to Verification: Four Days, One GPU, and Three Measured Failure Modes
## Building a 270M Document-Decision Model That Reads Real Scanned Receipts

**Draft v0 — written from the record** (`docs/PUBLICATION/CHRONOLOGY.md`, commit 3070126+).
Every number is a dated ledger row, eval-table JSON, or commit; the failures are
chapters, not footnotes, per the project's honesty contract.

---

## Abstract

We document the complete, four-day construction of a 270M-parameter document-decision
model — frozen SigLIP vision tower, cached page-embedding tokens, and a typed row head
over a T5Gemma-2-270m encoder — trained on a single RTX 3060 12 GB and taken from
coin-flip (51.3% accuracy, AUC 0.516) to **95.89% accuracy (AUC 0.9878, ECE 0.020)**
on a public receipt-verification protocol built on SROIE. The path there was not a
scaling curve but a sequence of three measured failure modes, each isolated by
pre-registered contracts with explicit DEAD bands: (1) a **degenerate benchmark**,
detected by the signature that two differently-built models scored the failing bar
*identically to three decimal places* (0.174) while differing everywhere else —
"the exam, not the student"; (2) an **answer-prior defect**, where minority
counterfactual augmentation taught the model a better prior instead of reading
(blank-state accuracy *rose* to 0.65), and exact answer-balancing fixed the prior
internally yet **publicly falsified** as an always-No collapse (true-claim acceptance
0.28%, AUC unchanged); and (3) the **synthetic-to-scan domain gap**, which — not the
prior, not the architecture — was the true public blocker: 6 minutes of fine-tuning on
625 real trainval receipts lifted the public benchmark by +44.6 points while the
internal robustness bars held. We release the model, all evaluation tables, a dated
benchmark ledger, and the full decision chronology, and argue the *methodology* —
gates that can fail, one variable per run, falsification-friendly contracts, honest
public ledgers — is the transferable contribution for small-model decision systems.

## 1. Introduction

**The question** (2026-11-23, verbatim from the kickoff spec): can a 270M model make
typed document decisions — not generate text — on one consumer GPU, with calibration
and adversarial robustness measured rather than assumed? The origin prompt specified
a decoupled architecture (state-only encoder, parallel non-autoregressive slot
decoder), zero-leakage unseen-task splits, deterministic stress suites, mandatory
dead-weight ablations, and GO/NO-GO bars, against a public reference checkpoint
(`ktcapraz/t5gemma2-decision-270m-v2`).

**Contributions.**
1. A measured case that **verification-style decision heads** — not generation — are
   the correct target for sub-300M multimodal document models (the generative wall:
   a 270M generative model transcribes digits at ~1–1.7% exact match; §3).
2. **The identical-score signature** for detecting degenerate benchmarks: when two
   differently-built models produce the same wrong number (0.174, 3 d.p.) on one bar
   while diverging on every other metric, the benchmark is the suspect (§5).
3. A **public falsification result**: internal answer-balancing (a correct fix for a
   measured prior defect) does not create claim-reading signal — it moved the prior
   from always-Yes to always-No with AUC unchanged (§7).
4. **Domain adaptation as the missing variable**: protocol-matched fine-tuning on 625
   real trainval receipts (3,150 rows, 2 epochs, lr 1e-5, ~6 min) closes the
   synthetic→scan gap to 95.89% / AUC 0.9878 / ECE 0.0203 on a frozen-protocol
   public test (§8), with internal robustness bars preserved.
5. A **methodology transferable to small-model campaigns**: six pre-registered bars,
   per-iteration public checkpoints, and a dated benchmark ledger with honest
   last-place rows (JevBench 39.8, ImajevBench 34.1 — recorded, not hidden).

## 2. The origin architecture (Gate 1)

From the kickoff spec, built to the letter on night one (commits 05a60d3…bfb3f3f,
2026-11-23 22:11–22:28): `DecoupledT5GemmaDecision` — state-only encoder
(`[STATE]` tokens only, zero questions), cached `H_enc`, decoder rows
`Question: {q} | Type: {primitive} | Options: {opt} <slot> …` with slots *after*
option text, merged cross-attention to the cached state, LayerNorm→MLP slot head
(640→320→1), strictly proper scoring losses (CE+Brier; CE+RPS for ordinal),
anti-shortcut augmentations, per-primitive temperature scaling fitted on val only.

Evaluation discipline from the same night: zero-leakage splits with the unseen
workflow hashed out of training; jitter suite (n=300) targeting the known 17–27%
encoder-stuffed flip bug; multi-question invariance suite (n=100) requiring
max-probability delta 0.0000 with cache speedup; mandatory blank-state ablation;
trivial baselines (unseen AUC 0.491 — no skill) and the external reference.

## 3. Gate 1: NO-GO honored, eval bugs caught by the gates

Night verdict (commit 6dde069): **NO-GO (0/5)** — in-domain beat the reference
(0.792/0.941/0.054 vs 0.759/0.919/0.161 ECE) but the unseen workflow collapsed
(acc 0.274, AUC 0.499; blank-state 0.268 ≈ real 0.276 → state ignored off-domain).

Two measurement lessons fell out of that night, both now standing laws:
- **fp32 eval law**: bf16 GEMM-order noise near softmax ties broke the invariance
  metric; CPU reproduction proved the model exactly invariant at fp32 (5304433).
- **The metric, not the model**: the multi-q suite walked slot scalars after a bare
  tensor was returned, computing |p−1| instead of invariance; fixed, the model was
  **exact: 100/100 cases, delta 0.0000** (3bd8962). Criterion 2 passed all along.

The decision pivot is recorded in the checkpoint report: a 270M *generative* model
transcribes digits at ~1–1.7% exact match — generation is the wrong head for this
scale; decisions (noul/choice/score with calibrated probabilities) are the right one.

## 4. Teachers, harness, and the synthetic factory (Gate 2 substrate)

Teacher pipeline (user-directed: "make an interface a hermes agent can drive — direct
wiring is not preferred"): work-order contract + deterministic scoring, evaluated by
exam. GLM-via-Hermes achieved the best measured teacher quality (KL 0.1387,
noul 79.4%, 500/500 valid, 13 waves zero failures) after a variant race (v2a won:
score +15 pts) and a teacher audit that took agreement from 10.5% to 65–81%
(fail-closed normalizer + alias referee + few-shot; commits 2cd019b, 935d55d,
33b3c2a).

The **paper_rig synthetic document factory** (commits 702dfbc…1a93727): seven
document families (thermal receipts, card statements, savings, electricity, salary,
telecom), multi-institution, smudge/legibility coupled to gold labels, JIT
micro-batches with per-call render seeding, and a **training-parity proof** (fresh
JIT pipeline vs pre-banked: +0.8%). Every label is derived from the factory ground
truth — exact by construction, zero leakage — and proof gates re-derive the full
pixel pipeline against gt on fresh seeds across all families (e517082).

## 5. Gate 2.5 and the broken exam

The row pathway (K1, commit 44ada8f) binds each question as one encoder row over
cached page tokens — the final architecture. Six pre-registered bars with fp32 eval
law (§3). The formal v1 row (checkpoint report §5): flip 0.150 FAIL, multiq 0.0 PASS,
ECE 0.197 FAIL (T pinned at ceiling 5.0), unseen-family AUC **0.174** FAIL, jitter
0.047 PASS. The retrained v2 (different wiring, different training outcome — val acc
0.89 vs 0.79) hit **g2_4 = 0.174 again, to three decimals**.

That identity is the paper's central diagnostic anecdote: two different brains
produced the same wrong number on one bar while diverging everywhere else, because
the bar's answers were predictable from the question wording alone (the salary split
always-Yes, the receipt split always-No). **The exam, not the student.** The banked
rule: *when two differently-built models score identically on a benchmark, check the
answer-key statistics before blaming the students.* The fix — K5 counterfactual
presence twins (same page family, one gt quantity flipped so gold flips; render and
labels share one omission registry) — is exact by construction (a4a363c, 9395217).

## 6. Going public: the ledger (2026-11-26)

The campaign rule (user, verbatim): *"Try to compete in those benchmarks and tell me
what is the thing and record all the benchmark results date by date relative to
other models, then we will improve."* The ledger was born with three honest rows:
JevBench public-231 **39.8** (text head, recorded last place), ImajevBench **34.1**
(photo domain — measured, closed as out-of-scope with evidence), and the first
in-distribution public entry: **SROIE verification-v1**, a protocol we define and
freeze (361/400 official test receipts with public Task-3 gold, zero test-ID overlap
with the adaptation snapshot, verified by receipt-ID join; fields {total, date,
company} × {TRUE gold-verbatim, FALSE cross-receipt perturbation}, seed 2026, exactly
50/50, 2,166 rows). v2 scored **51.2% / AUC 0.516 / ECE 0.252** — reads real scans
(TRUE-claim acceptance 98.3% on dates, 95.6% on companies) but cannot reject false
claims (20.1%). The defect was now public and pre-registered as the v4 target.

## 7. v4: the balance fix and its public falsification (2026-11-27)

v3 (600 counterfactual twins added as a minority) taught a *better prior*, not
reading — the blank-state smoking gun: with pages hidden, v3 (0.65) beat v2 (0.55),
i.e. it depended on the page *less*. DEAD band honored, champion preserved.

v4 made presence answers balanced at training time (omission guards in all four train
family generators + `--balance-presence`, Yes ≤ 2× No per family+template; 17,776
surplus Yes rows dropped). Internally it worked: **flip 0.387 (best-ever), blank 0.5005
(the ablation is finally blind-correct), ECE 0.089 post-TS, multiq 0.0000, test 0.798**
(7bec037). Publicly it **falsified the pre-registered prediction**: always-No collapse
— true-claim acceptance **0.28%**, false-rejection 99.7%, **AUC unchanged at 0.517**
(24002fe, mlflow d44498f7). The temperature-scaling refit was correctly *cancelled*
per the pre-registered bridge rule: no monotone transform on an uninformative score.
Balance corrects priors; it does not create signal.

## 8. v5-A: the domain was the blocker (2026-11-27)

One variable: the training distribution gains real scans. 625 SROIE *trainval*
receipts (525/100 carve, seed 2027) → 3,150 train / 600 val verification rows built
with the *eval's exact* question template, exact 50/50, cross-receipt false draws
(never equal to own gold — verified 0), zero test-pool overlap. v4-init, lr 1e-5,
2 epochs, **~6 minutes on the 3060**.

| SROIE verification-v1 | v2 | v4 | **v5-A** |
|---|---|---|---|
| accuracy | 51.2 | 49.9 | **95.89** |
| AUC | 0.516 | 0.517 | **0.9878** |
| ECE (15-bin) | 0.252 | 0.316 | **0.0203** |
| per-field AUC t/d/c | ~coin | 0.48/0.50/0.61 | **0.984 / 0.966 / 0.9998** |

Internal bars essentially preserved: flip 0.373 (v4: 0.387), blank 0.505, unseen
AUC 0.379 (v4: 0.394 — within noise), test 0.794. Disclosed caveat: internal
synthetic-benchmark ECE 0.1106 (TS fitted on the synthetic val split) — an
ECE-optimal temperature refit is in flight and its result will be attached either
way. The causal chain is complete: **balance fixed the prior (v4), real scans fixed
the domain (v5-A), and neither alone sufficed.**

## 9. Methodology (the transferable part)

1. **Gates that can fail.** Six bars, thresholds pre-registered before training;
   verdicts binding (DEAD bands honored twice: v3, and the bridge rule in §7).
2. **One variable per run.** Balance OR twins OR domain — never two at once — so
   every verdict attributes its cause. This is why §7's falsification was clean.
3. **Falsification-friendly contracts.** v4's PASS band included a public condition;
   its failure was recorded as a *result*, not an embarrassment, and v5-A was
   designed from the falsification's residue (the one signal cell, company AUC
   0.614, pointed at domain).
4. **The identical-score signature** (§5) as a cheap benchmark-validity check.
5. **Dated public ledger + per-iteration checkpoints.** Every claim has a date, a
   protocol, a commit, and (since 11-26) a public HF artifact; last places stay in.

## 10. Limitations

- **Unseen document families remain unsolved**: leave-one-family-out AUC 0.379 vs
  the 0.75 bar (salary transfer) — the adaptation recipe is per-domain, not a
  general transfer mechanism.
- **Calibration is domain-specific**: temperatures fitted on synthetic val do not
  transfer (internal ECE 0.1106 vs SROIE 0.0203); deployment requires per-domain
  temperature fitting — an honest operating cost, measured.
- **No abstention head**; the ImajevBench τ-sweep found no useful threshold.
- **Single-domain adaptation scale**: 625 receipts proved the mechanism; CORD and
  multi-domain adaptation are untested at draft time.
- The internal benchmarks are synthetic (by construction exact, but stylized); the
  public claim rests on the frozen SROIE protocol.

## 11. Reproducibility

Public repo (MIT, github.com/hr1juldey/drishyam-1, 116+ commits with dated verdicts);
public checkpoints (HF, license `gemma`): `riju279/drishyam-vision-decision-270m`
(v2 at repo root; v5-A under `v5a/` with all eval tables + probability dumps),
`riju279/drishyam-text-decision-270m`; ledger (`docs/BENCHMARK_LEDGER.md` +
`artifacts/benchmark_ledger.jsonl`); MLflow (sqlite) runs for every training and
eval; protocol documents (SROIE verification-v1 with rows digest `373d16ee`);
the full decision chronology (`docs/PUBLICATION/`). Hardware: one RTX 3060 12 GB;
the entire campaign's GPU compute is measured in hours, not days.

---

### Draft TODO (before this leaves draft status)
1. TS-polish number lands → §8 caveat resolves (either direction).
2. Related-work section (encoder-stuffed decision baselines, KIE literature incl.
   Qwen2-VL-72B 75.5 KIE / best 90.5 Hmean on ICDAR; Clever Hans / shortcut-learning
   literature for §5; benchmark-validity literature).
3. Figures: (a) the causal chain diagram (exam→prior→domain), (b) v2/v4/v5-A SROIE
   bars, (c) the ledger timeline.
4. Ablation appendix from existing tables (v3 twins-as-minority, v4 balance-only).
5. Venue decision is the user's; package made submission-ready first.
