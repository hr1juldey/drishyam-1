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

## Phase C — few-shot anchor (pending)

Two worked examples from held-out train rows (not the test 20) prepended to the
prompt. Measures Suspect 2: if agreement jumps again, the gap is house-style,
not model quality.

## Phase D — polarity map (pending, falls out of Phase B/C output)

Per-question-type agreement. Reading guide: if yes/no (noul) agrees well but
score/choice do not, the gap lives in rubric nuance; if everything is low after
the referee fix, it is interpretation; if all are high, suspects 1+2 explained
everything and the teacher is fine.

## Phase E — human eyeball (pending)

3 side-by-side cases (teacher reasoning-free answers vs gold) read by the
agent, judged for defensibility. Whatever disagreement remains after B+C gets
classified here: defensible gray vs wrong.

## Morning proposal (after verdicts)

The 2-lakh factory: Hermes subagent supervisor over batched API calls, with
per-batch quality gates (schema + sum-to-1 + spot-check vs gold where gold
exists + failure ledger). Designed only after Phases B–E say the measurement
itself is trustworthy.
