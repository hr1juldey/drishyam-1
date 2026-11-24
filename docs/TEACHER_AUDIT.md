# TEACHER AUDIT — why did the bake-off score below random?

Recorded 2026-11-24, morning session. Style: problem → suspects → experiment →
evidence → verdict, one row per finding. Gate-2 (Banking77) uses the same
teacher, so every verdict here gates that pipeline's trust too.

## Problem

Bake-off (20 gold cases × 3 samples, dataset's own human answer key):

| Drafter | soft-KL ↓ | argmax agree ↑ | throughput |
|---|---|---|---|
| gemma4:31b | 1.220 | 10.5% | 23.7 cases/h |
| lfm3-omni | 1.380 | 0.0% | 646 cases/h |

Random guessing on 2–4-option questions lands ~25–33%. **Both drafters scored
below random.** Below-random is not "weak" — it is structured disagreement,
which means something systematic is broken (teacher OR referee).

## Suspects

1. **Referee is deaf** — the scorer silently zero-fills unmatched option keys,
   so any key-format mismatch becomes a garbage distribution with a biased argmax.
2. **Interpretation gap** — the teacher reads the questions differently than the
   dataset's authors did (polarity/house-style).
3. **Genuine disagreement** — the judgments are ambiguous and the teacher
   honestly differs. (Some of this is desirable in soft labels.)

## Phase A — referee audit (evidence: artifacts/teacher_audit/phase_a_raw.json)

Dumped RAW teacher answers on 3 gold cases, compared keys vs the answer key.

**Finding A1 — the teacher's key format is UNSTABLE.** Same prompt, same model:

- security_incidents: echoed FULL strings `close_benign: Expected, explainable...`
  (5/5 questions) — matches gold-by-accident? NO — see Finding A2.
- invoice_processing: 3/5 questions stripped to BARE keys (`approve`, `0`...)
- customer_service: 5/5 questions stripped to BARE keys

**Finding A2 — gold convention resolved: the answer key uses BARE keys**
(`approve`, `0`, `close_benign`) — verified directly on the dataset's
`gold.probabilities`. The audit's first comparison (vs full option strings) was
itself wrong; gold is bare.

**Verdict A: Suspect 1 CONFIRMED as a major contributor.** Whenever the teacher
echoed full `key: description` strings, the old normalizer zero-filled every
option → all-zero distribution → argmax = first gold key, KL huge. That is a
systematic bias proportional to how often the teacher echoes full strings —
exactly the below-random artifact. The teacher was never answering "nothing";
the referee just could not hear bare-vs-full differences and punished them as
emptiness.

## Phase B — referee fixed, prompt hardened, re-measure (running)

Fixes applied (committed):

1. `answers_to_gold_keys` v2: full strings are stripped to their prefix before
   matching; **any unknown key rejects the whole question (fail-closed)** — the
   scorer can no longer invent zeros. Unit-tested (bare, strip, reject, missing).
2. Prompt now mandates: "ANSWER KEYS MUST BE EXACTLY THE BARE OPTION KEY — the
   part BEFORE the ':'".
3. Bake-off now reports per-question-type agreement (noul / choice / score) —
   the Phase D polarity map falls out of the same run.

Result: pending (running).

## Phase B/C/D — RESULTS (bake-off, 20 gold cases × 1 sample, gemma4:31b)

| Configuration | case agreement | mean KL | valid cases | wall |
|---|---|---|---|---|
| broken referee (last night) | 10.5% (below random) | 1.220 | 20/20 — a lie | 75 min |
| v2b: alias referee, no few-shot | 0% case / 72.7% choice, 73.1% noul, **18.8% score** | 1.582 | 16/20 | 13 min |
| v3: alias referee + few-shot | **35.0% case** / **80.8% choice, 73.5% noul, 65.0% score** | **0.223** | 20/20 | 13 min |

Random case-level agreement on 5 questions ≈ 0.2%. We measure **35%**.
Per-question random floor is 25–33%; we measure **65–81%**.

**Suspect 1 — CONFIRMED, two layers.**
(a) The old scorer silently zero-filled unmatched keys — any format mismatch
became an all-zero distribution with a biased argmax. (b) The teacher keys by
the human-readable rank NAME (`Routine`, `Material`, `Critical`) while gold
keys are numeric ranks (`1`, `3`, `4`) — same level, different naming. Fixed by
a deterministic alias table built from the dataset's own option strings
('1: Routine; handle...' ⇒ Routine→1); fail-closed kept for truly unknown keys.

**Suspect 2 — CONFIRMED, localized to score-type questions.**
Without few-shot the score rubric agreed only 18.8% (and 4/20 cases were
rejected outright); two worked gold examples moved score to 65.0% and
choice 72.7→80.8% (noul flat 73%). The dataset's graders have a strictness
house-style that a bare rubric does not convey; examples do.

**Suspect 3 — PARTIALLY REAL, and it is character, not defect.**
Phase E eyeball of residual disagreements (evidence: NIGHT_LOG 08:15 run):
- `urgency` gold 1@0.35 vs teacher 2@0.40 — gold itself is nearly FLAT
  (0.35/0.32/0.30): genuinely ambiguous case, disagreement is honest gray.
- `discrepancy_severity` gold 0@0.64 vs teacher 2@0.70 — but gold puts 27% on
  '2' itself; the teacher leans further into the gold's own second choice.
- `disposition` gold approve@0.65 vs teacher hold@0.40 + manual_review@0.40 —
  a SYSTEMATIC CAUTION SHIFT: the teacher is more conservative than the gold
  annotators (holds more, escalates more, scores severity higher).
- `matches_order` gold true@0.68 vs teacher false@0.90 — the one concerning
  pattern: occasional CONFIDENT flips on yes/no questions (~25% of noul).

## Harness interface (user directive: NO direct API wiring)

User ruling 2026-11-24 ~08:30: the teacher must be an interface a Hermes agent
DRIVES — Hermes is the harness; direct API wiring is not preferred; yes/no
questions use the same path (no direct route for anything).

Built `scripts/harness.py`:
- `prepare` — emits work-order files (`drafts/harness/wo_*.json`, 5 cases
  each, gold absent) + `CONTRACT.md` (output schema, key rules, the 2 worked
  examples as strictness anchors, "no confident flips on close binaries").
- labeling — Hermes subagents (GLM via Hermes's own provider; children inherit
  the parent model) read contract + work order, label, self-verify, write
  `draft_*.json`. 4 workers in parallel.
- `score` — the SAME deterministic alias-table referee validates and scores
  the drafts; provenance stamped from the manifest.

Dry-tested the score path with synthetic near-gold drafts BEFORE dispatching
(20/20 valid, referee behaved), then ran the real exam on the same seed-42
20 gold cases gemma4 was measured on.

## Three-way table (same 20 cases, same referee, same gold)

| Teacher | valid | mean KL ↓ | case agree | choice | noul | score | throughput |
|---|---|---|---|---|---|---|---|
| gemma4, broken referee | 20/20* | 1.220 | 10.5% | — | — | — | 23.7 cph |
| gemma4, fixed ref + few-shot | 20/20 | 0.223 | 35.0% | 80.8% | 73.5% | 65.0% | 92 cph |
| **GLM via Hermes harness** | **20/20** | **0.1387** | 20.0% | 73.1% | **79.4%** | 57.5% | **~436 cph @ 4 workers** |

*lie (silent zero-fill).

**Reading:**
1. **The harness contract + self-verification loop achieved PERFECT schema
   compliance on the first try** (20/20 valid, zero referee rejects, numeric
   and named keys all correct) — no alias table even needed; the referee only
   validates. The "do not confidently flip close binaries" rule lifted noul to
   79.4%, the best yes/no number measured all day.
2. **GLM-via-Hermes wins the KL metric — 0.139 vs 0.223.** KL is the actual
   training target for soft-label distillation, so the distributions the
   harness workers produce are the CLOSEST to the dataset's answer key of any
   teacher measured. gemma4 keeps the edge on per-question argmax (choice,
   score) and all-5 case agreement (35% vs 20%) — a harsher all-or-nothing
   read, but not the training-loss metric.
3. **Throughput: 20 cases / 165 s with only 4 workers ≈ 436 cases/h, and
   workers scale horizontally** (10 per dispatch, multiple dispatches).
   vs gemma4's 92 cph — ~4.7× at 4 workers.
4. **2-lakh math (updated):** 200k rows ÷ 436 cph ≈ 19 days at 4 workers;
   ≈ 8 days at 10; compressible further with parallel dispatch cycles or by
   raising batch size. The gemma4 local route (~90 days) is no longer the
   plan of record.
5. **Caveat:** n=20, single sample. Before factory commitment, re-run both
   teachers on 50–100 cases to confirm the KL ordering holds.

## Final verdicts


1. **The teacher was never broken — the referee was.** With translation +
   house-style anchor: KL 0.223 (was 1.220), per-question agreement 65–81%.
2. The residual is a **cautious annotator**, not a wrong one. For soft-label
   distillation this is usable as-is (KL 0.22 is close), and a conservative
   auto-approver is arguably the safer failure mode. WATCH ITEM: confident
   yes/no flips (~25% of noul) — if Gate-2's decisions are binary-heavy,
   add a "when unsure between false/true, reflect uncertainty, do not flip"
   instruction and re-measure.
3. **Factory math (measured, warm model): 20 cases / 13 min ≈ 92 cases/h.**
   2 lakh rows on gemma4 alone ≈ 90 days — not viable locally; the API teacher
   (GLM flash) must pass THIS SAME 20-case bake-off (the rig is now the quality
   gate) before factory use. Provenance per row stays mandatory.

## Gate-2 (Banking77) implication

The same rig gates Banking77 labels: bake 20 gold cases → require KL ≤ ~0.3
and per-question agreement ≥ ~60% before any mass labeling. If the banking
rubric disagrees structurally, it will show here FIRST, at 20-case cost,
not after 2 lakh rows.


## Morning proposal (after verdicts)

The 2-lakh factory: Hermes subagent supervisor over batched API calls, with
per-batch quality gates (schema + sum-to-1 + spot-check vs gold where gold
exists + failure ledger). Designed only after Phases B–E say the measurement
itself is trustworthy.
