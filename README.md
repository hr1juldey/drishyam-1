# drishyam-1

> **drishyam** (Sanskrit दृश्यम्) — "that which is seen." The project becomes a
> vision decision model at Gate 2; before that, everything it scores is
> real-time-first (user directive 2026-11-23).

**Decoupled T5Gemma-2-270M System-1 decision engine** — a state-only encoder whose
hidden states are cached once per decision state, feeding a parallel
non-autoregressive slot decoder that scores every question/option in one forward
pass. Base model: `google/t5gemma-2-270m-270m` (18 enc / 18 dec layers, d=640,
tied embeddings, merged decoder attention).

The architectural claim under test: if questions and options never touch the
encoder, one cached state serves unlimited questions with **exact** cross-question
invariance, measurable cache speedup, and no case/whitespace flip bugs — the
failure class documented by the benchmark below for encoder-stuffed models.

The real-time thesis: **look once, decide many.** The expensive look — one
encoder pass over the state (later: the 896×896 scene) — runs once and is
cached; every question after it is a short parallel decode off the cache. That
is the shape that makes multi-question decision serving real-time, and it is
why the same architecture carries the vision gate.

## Benchmark

[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(Apache-2.0): 4 workflows × (300 train + 100 test) cases, 5 typed questions per
case, primitives `noul` (yes/no), `choice` (one of N), `score` (ordered rubric),
full gold distributions. Metrics mirror the benchmark's official scorer
(accuracy / KL / Brier / ECE) for leaderboard comparability.

## Gate ladder

| Gate | Scope | Status |
|---|---|---|
| **Gate 1** | Text-only architectural proof on typed-decisions: decoupled encoder/decoder + slot head + calibration, vs trivial and external-reference baselines, with state-ablation and invariance diagnostics. GPU: RTX 3060 12 GB (CUDA). | **planning complete — execution pending** |
| **Gate 2** | Full multi-dataset training + native 896×896 multimodal vision decisions + Banking77 (>70-option) scaling. Unlocked only if Gate 1 is a GO. | locked |

Gate 1 go/no-go criteria, evaluation suites, and the full execution roadmap:
[`docs/ROADMAP.md`](docs/ROADMAP.md). Every deviation from the original spec and
every open question: [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Tooling law (non-negotiable in this repo)

- **Python only via uv** — never bare `python3`, never system Python. `uv sync`,
  then `uv run python ...`. `ast-grep scan` enforces this on shell scripts.
- **ruff** gates every commit; **ast-grep** structural checks around every code
  edit (diff-based edits, no blind rewrites).
- **pydantic** validates every config and every `results/drishyam_results.jsonl` row.
- **Hugging Face libraries** (`transformers`, `datasets`, `peft`, `accelerate`,
  `hf-transfer`) do all model/data work.
- Training runs are tracked in **MLflow** (`sqlite:///mlflow.db`), smoke-run
  before any full run, checkpointed for resume, and every results document is
  rendered from artifacts by script — never hand-typed.

## Quickstart (after Gate-1 execution starts)

```bash
uv sync
uv run python -m drishyam.probe        # GPU / bf16 capability check
uv run python scripts/next_step.py          # prints the next sub-task to execute
```

## License

MIT — see [LICENSE](LICENSE).
