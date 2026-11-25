"""MLflow tracking (user law: sqlite backend, PRIMARY tracker)."""

from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]


def mlflow_run(name: str):
    (ROOT / "runs").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'runs' / 'mlflow.db'}")
    # 2026-11-25: g2-sft/g2-labels landed in Default — the dashboard watches
    # drishyam-doc and showed nothing. Experiment set HERE so every stage is
    # born in the watched experiment, not moved by hand afterwards.
    mlflow.set_experiment("drishyam-doc")
    return mlflow.start_run(run_name=name, tags={"project": "drishyam-1", "gate": "1"})
