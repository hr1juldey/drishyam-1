# EXEC ENGINE — the loop protocol

An agent with zero context and dead memory must be able to resume this project
from DISK alone. `docs/ROADMAP.md` is the catalog; this file is the loop;
`docs/EXEC_STATE.json` is the machine state; `docs/PROGRESS.md` is the human
ledger; `scripts/next_step.py` (built in G1-E0) is the single entry point.

## State schema (EXEC_STATE.json)

Per sub-task ID: `status` ∈ todo | in_progress | done | parked, `attempts`,
`note`, `commit` (HEAD sha at done), `artifact` (path that proves it).

## Loop (one iteration = one sub-task attempt)

1. `uv run python scripts/next_step.py` → prints next sub-task + spec + commands
   + PASS criterion, marks it in_progress.
2. Execute per the ROADMAP spec exactly (diff-based edits; long GPU jobs as
   separate background processes, never chained with tests).
3. Run the PASS command(s).
4. PASS → commit with the sub-task ID in the message →
   `next_step.py --done ID --note "<measured result>"` (captures HEAD, updates
   state + appends `docs/EXEC_LOG.md`, auto-commits both) → report one paragraph
   with numbers.
5. FAIL → `next_step.py --fail ID --note "<exact error + diagnosis>"` FIRST
   (record before touching code), classify the failure (mechanical vs design),
   change ONE variable, retry. 3 failures → PARKED. Parked sub-tasks need a
   human; they are never silently retried and are listed in every report.

## Verify invariants (`--verify`)

V1: a committed sha in git ⇒ that ID is `done` in state (no drift).
V2: state IDs ⊆ ROADMAP IDs (no orphans).
V3: `done` ⇒ artifact path exists (no false completions).
V4: `parked` is never returned as next.
V5: parked list == parked statuses.
Violations print the exact repair command.

## Documentation law

Results docs are RENDERED from measurement artifacts by script. A result doc may
describe future work, but every number in it comes from an artifact on disk;
missing artifact ⇒ `MISSING` cell, never 0, never prose.

## Refinement policy

A near-miss gate result tempts a "quick rerun". Allowed ONLY where the ROADMAP
pre-names the condition and the single variable to change (e.g. S0's
ManualAdamW fallback, T1's OOM batch-halving). Anything else is scope creep →
surface to the user.
