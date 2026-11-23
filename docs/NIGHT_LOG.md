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
Traceback (most recent call last):
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 184, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 176, in main
    rep = bakeoff(a.model.split("+"), a.cases, a.samples, a.seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 112, in bakeoff
    acc_p = {q: {k: float(np.mean([_num(p.get(q, {}).get(k)) for p in sample_ps]))
                                        ^^^^^^^^^^^^^^^^
AttributeError: 'float' object has no attribute 'get'
- 00:10:34 [cpu] bakeoff PARKED rc=0
drafted 0 cases (3 consecutive fails at stop)
- 00:14:35 [cpu] drafter end (0 cases)
- 00:14:35 [cpu] lane done
- 00:15:16 === NIGHT GPU QUEUE start (pid 638553) ===
G1-T1: done
- 00:15:16 STEP V1 start: uv run python -m drishyam.evaluate --ckpt runs/g1-t1/best.pt --out artifacts/gate_table.json
- 00:15:16 [cpu] === NIGHT CPU LANE start (pid 638595) ===
- 00:15:16 [cpu] bake-off start (lfm3-omni vs gemma4:31b, 50 cases x 3 samples)
2026/11/24 00:15:21 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/evaluate.py", line 205, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/evaluate.py", line 159, in main
    model, tok, fixed_len = load_model(ROOT / a.ckpt)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/evaluate.py", line 23, in load_model
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 1616, in load
    return _load(
           ^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2260, in _load
    result = unpickler.load()
             ^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2208, in persistent_load
    typed_storage = load_tensor(
                    ^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2173, in load_tensor
    wrap_storage = restore_location(storage, location)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 1969, in restore_location
    return default_restore_location(storage, map_location)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 735, in default_restore_location
    result = fn(storage, location)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 669, in _deserialize
    return obj.to(device=device)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/storage.py", line 298, in to
    return _to(self, device, non_blocking)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/_utils.py", line 108, in _to
    untyped_storage = torch.UntypedStorage(self.size(), device=device)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11.61 GiB of which 76.81 MiB is free. Process 2627 has 36.53 MiB memory in use. Process 2630 has 34.66 MiB memory in use. Process 3261 has 70.28 MiB memory in use. Process 372545 has 34.16 MiB memory in use. Process 486650 has 154.16 MiB memory in use. Process 3874839 has 232.00 MiB memory in use. Process 3506424 has 232.00 MiB memory in use. Process 601359 has 3.19 GiB memory in use. Process 626876 has 6.59 GiB memory in use. Including non-PyTorch memory, this process has 474.00 MiB memory in use. Of the allocated memory 368.58 MiB is allocated by PyTorch, and 1.42 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
- 00:15:22 STEP V1 fail#1 rc=1 — retrying
2026/11/24 00:15:47 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 9449.01it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
- 00:16:50 STEP V1 fail#1 rc=137 — retrying
2026/11/24 00:17:15 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 11338.01it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
NIGHT_QUEUE_EXIT=137
