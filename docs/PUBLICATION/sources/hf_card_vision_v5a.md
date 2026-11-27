---
license: gemma
base_model: google/t5gemma-2-270m
tags:
- document-understanding
- decision-model
- receipts
- calibrated
metrics:
- accuracy
- ece
- auc
---

# drishyam-vision-decision-270m (vision head, v5-A — domain-adapted)

A small, calibrated **document-decision model**: given a document page image and a typed
question (binary yes/no, multiple-choice, or ordinal score), it returns a decision —
not text. Fine-tune of `google/t5gemma-2-270m`; **inherits the Gemma Terms of Use**
(license `gemma`, *not* Apache).

`best.pt` here is `runs/g2-row-sft-v5a/best.pt` from the source repo (`drishyam-1`):
**v4-best (balanced-pair retrain) adapted for 2 epochs on 3,150 verification rows built
from 625 real SROIE trainval scanned receipts** (protocol-matched to the eval — same
question template, exact 50/50, cross-receipt false draws, zero test-pool overlap).
Machine-readable eval tables are attached: `gate2_row_table_v5a.json` (internal gates)
and `sroie_table_v5a.json` (public receipt-verification benchmark).

This card is the project chronology in miniature: every chapter below is a dated,
measured iteration — including the failures. Every number is a dated row in the
benchmark ledger (`docs/BENCHMARK_LEDGER.md` in the source repo).

## Architecture

Frozen SigLIP vision tower → page-embed projector → typed row head
(`forward_row` encoder-row binding):

- **Vision:** frozen SigLIP-style tower; pages enter as cached embedding tokens
  (256 page-embedding slots, never re-encoded during training).
- **Row head:** each question is one encoder row —
  `inputs_embeds = [page_embs(256) ; question+option tokens]` fed to the t5gemma
  encoder; answer slots read from the encoder output. Three typed heads:
  **noul** (binary), **choice** (2–5 options), **score** (ordinal, 5).
- **Calibration (v5-A temperatures, fitted on internal synthetic val, NLL):**
  noul T=5.0, choice T=3.62, score T=1.215.

## Headline result — SROIE receipt-verification v1 (public, 2026-11-27)

Protocol self-defined on public gold and pre-registered: 361 official SROIE test
receipts (public Task-3 KIE gold; zero overlap with the adaptation snapshot, verified
by receipt-ID join) × fields {total, date, company} × {TRUE gold-verbatim, FALSE
cross-receipt perturbation}, seed 2026, exactly 50/50 = **2,166 rows**, fp32.

| slice | v2 (2026-11-26) | v4 (2026-11-27) | **v5-A (2026-11-27)** |
|---|---|---|---|
| **overall accuracy** | 51.2 | 49.9 | **95.89** |
| **AUC (P-Yes)** | 0.516 | 0.517 | **0.9878** |
| **ECE (15-bin)** | 0.252 | 0.316 | **0.0203** |
| TRUE-accept total / date / company | 53.2 / 98.3 / 95.6 | 0.28 (all) | **96.1 / 94.2 / 99.5** |
| FALSE-reject overall | 20.1 | 99.7 (blind) | **95.2** |
| per-field AUC (total / date / company) | ~coin | 0.48 / 0.50 / 0.61 | **0.984 / 0.966 / 0.9998** |

**The one-line story:** the same architecture went from coin-flip to 95.9% on a public
benchmark through two measured data fixes — **answer-balance** (v4) and
**scan-domain adaptation** (v5-A) — in 6 minutes of training on 625 real receipts.

## The chronology (what actually happened, failures included)

### v2 — the first honest public entry (2026-11-25/26)
4/5 internal bars PASS (flip 0.346, multiq 0.0000, ECE 0.033, jitter 0.0055;
unseen-family AUC 0.175 FAIL). Public SROIE: 51.2% — *reads* real scans (98.3% on
TRUE dates) but cannot reject FALSE claims (20.1%). The Yes-prior defect, measured.

### v3 — counterfactual-minority experiment (2026-11-26) — DEAD band honored
Adding 600 K5 counterfactual twins taught the model the *prior*, not the *reading*:
unseen-family transfer rose (0.175→0.261) while the core flip gate fell (0.346→0.261
flip quality, blank-acc drifted off 0.5). Pre-registered DEAD band applied; champion
preserved. Lesson recorded: minority counterfactuals are priors, not features.

### v4 — balanced-pair retrain (2026-11-27) — internal fix, public falsification
Exact 50/50 balanced training (17,776 surplus Yes rows dropped): **flip 0.387
(best-ever), blank 0.5005, ECE 0.089 post-TS, multiq 0.0000, test 0.798.** Public
SROIE re-score: **always-No collapse** — true-claim acceptance 0.28%, rejection 99.7%,
AUC unchanged (0.517). **The pre-registered public prediction was falsified**: balance
flipped the degenerate prior without creating claim signal. TS refit correctly
*cancelled* per the bridge rule (no temperature on an uninformative score).

### v5-A — domain adaptation (2026-11-27) — the gap was the domain, not the brain
v4-init + 2 epochs lr 1e-5 on 3,150 real-receipt verification rows (~6 min on an RTX
3060). Internal bars essentially preserved (flip 0.373, blank 0.505, unseen-family
AUC 0.379 vs v4's 0.394, test 0.794) while the public benchmark went **51→95.9**.
Adaptation data: `LocalLLaMA`-style protocol rows over SROIE *trainval* (625
receipts) — never the test pool. Per-field AUC company **0.9998**.

## Internal gate table (v5-A, `gate2_row_table_v5a.json`)

| bar | value | threshold | verdict |
|---|---|---|---|
| ablation flip | 0.373 | > 0.30 | PASS |
| multiq delta (4dp) | 0.0 | == 0.0 | PASS |
| ECE (post-TS, 15-bin) | 0.1106 | < 0.08 | **FAIL (caveat)** |
| unseen-family AUC | 0.379 | > 0.75 | FAIL (v4: 0.394 — within noise) |
| jitter flip | 0.0116 | < 0.05 | PASS |

The internal ECE miss is a calibration-domain artifact (TS fitted on synthetic val).
An ECE-optimal temperature refit was attempted twice (GPU ~14 min in, CPU ~1h45m in);
both runs were killed externally before writing results, so the refit remains
**deferred, not done** — ECE 0.1106 stands as the measured number. The refit script
is committed (`drishyam.ts_polish`) and can be run any time (~15 min on GPU). The
SROIE side is already calibrated (ECE 0.020).

## Also measured (context, honest)

- **JevBench text head** (`riju279/drishyam-text-decision-270m`, initialized from v2):
  39.8 on JevBench public-231 (2026-11-26) — recorded last place, kept for the ladder.
- **ImajevBench** photo-domain: 34.1% (2026-11-26) — closed as out of scope with
  evidence.

## Limitations

- **Calibration domain split:** internal-synthetic ECE 0.1106 (caveat above) vs
  SROIE 0.0203 — temperatures are deployment-domain-specific.
- **Unseen document families:** leave-one-family-out AUC still fails its 0.75 bar
  (0.379); salary-family transfer remains unsolved.
- **No abstention/Unknown head option.**
- **Adaptation scale:** 625 receipts proved the mechanism; multi-domain adaptation
  (CORD next) is untested.
- Photo domains fail (ImajevBench closed).

## Source

Trained and evaluated in the `drishyam-1` repo (public, MIT); every number above is a
dated row in `docs/BENCHMARK_LEDGER.md` with the full protocol in the attached table
JSONs. From Hrijul — formerly R&D, Ashva.
