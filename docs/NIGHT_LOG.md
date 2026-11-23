- 23:46:42 === NIGHT GPU QUEUE start (pid 569585) ===
- 23:46:42 waiting for runs/g1-t1.log EXIT marker
- 23:46:42 [cpu] === NIGHT CPU LANE start (pid 569711) — waiting for training EXIT ===
G1-T1: done
- 23:57:42 STEP C1 start: uv run python -m drishyam.calibrate
- 23:57:42 [cpu] training EXIT seen — CPU lane starting (num_thread=8 capped so GPU lane keeps headroom)
- 23:57:42 [cpu] bake-off start (lfm3-omni vs gemma4:31b, 50 cases x 3 samples)
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 65, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 58, in main
    table = fit_all()
            ^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 51, in fit_all
    p = np.stack([np.asarray(x, dtype=float) for x in probs[sel]])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/numpy/_core/shape_base.py", line 458, in stack
    raise ValueError('all input arrays must have the same shape')
ValueError: all input arrays must have the same shape
- 23:57:43 STEP C1 fail#1 rc=1 — retrying
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 65, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 58, in main
    table = fit_all()
            ^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 51, in fit_all
    p = np.stack([np.asarray(x, dtype=float) for x in probs[sel]])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/numpy/_core/shape_base.py", line 458, in stack
    raise ValueError('all input arrays must have the same shape')
ValueError: all input arrays must have the same shape
- 23:58:03 STEP C1 fail#2 rc=1 — retrying
- 23:58:23 STEP C1 PARKED (3-strikes)
- 23:58:23 STEP V1 start: uv run python -m drishyam.evaluate --ckpt runs/g1-t1/best.pt --out artifacts/gate_table.json
2026/11/23 23:58:27 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 10758.68it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
NIGHT_QUEUE_EXIT=143
- 23:59:58 === NIGHT GPU QUEUE start (pid 601283) ===
- 23:59:58 waiting for runs/g1-t1.log EXIT marker
G1-T1: done
- 23:59:58 STEP C1 start: uv run python -m drishyam.calibrate
{
  "noul": {
    "T": 0.94,
    "n": 300
  },
  "choice": {
    "T": 0.955,
    "n": 240
  },
  "score": {
    "T": 0.9375,
    "n": 360
  }
}
- 23:59:59 STEP C1 PASS (rc=0)
- 23:59:59 STEP V1 start: uv run python -m drishyam.evaluate --ckpt runs/g1-t1/best.pt --out artifacts/gate_table.json
2026/11/24 00:00:03 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 10797.17it/s]
Traceback (most recent call last):
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 179, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 171, in main
    rep = bakeoff(a.model.split("+"), a.cases, a.samples, a.seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 107, in bakeoff
    acc_p = {q: {k: float(np.mean([p.get(q, {}).get(k, 0.0) for p in sample_ps]))
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py", line 3862, in mean
    return _methods._mean(a, axis=axis, dtype=dtype,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/numpy/_core/_methods.py", line 132, in _mean
    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'
- 00:00:09 [cpu] bake-off PARKED (rc=1) — drafter still runs with gemma4:31b default
- 00:00:10 [cpu] night drafter start (gemma4:31b, budget 6h, max 400 cases, drafts/ quarantined)
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
- 00:03:44 [cpu] night drafter end — drafts at drafts/seen3_drafts.jsonl (0 cases)
- 00:03:44 [cpu] === NIGHT CPU LANE end ===
NIGHT_CPU_EXIT=0
