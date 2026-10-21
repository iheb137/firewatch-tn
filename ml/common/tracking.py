"""Journalisation MLflow optionnelle.

Si ``MLFLOW_TRACKING_URI`` n'est pas défini ou si mlflow n'est pas installé,
les appels deviennent des no-op : les scripts tournent aussi sur Colab/Kaggle.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from ml.common.config import flatten


class Tracker:
    def __init__(self, experiment: str, run_name: str | None = None):
        self.enabled = False
        self._mlflow = None
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return
        try:
            import mlflow
        except ImportError:
            return
        mlflow.set_experiment(experiment)
        self._mlflow = mlflow
        self._run_name = run_name
        self.enabled = True

    @contextlib.contextmanager
    def run(self):
        if not self.enabled:
            yield self
            return
        with self._mlflow.start_run(run_name=self._run_name):
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            flat = {k: str(v)[:500] for k, v in flatten(params).items()}
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            clean = {k: float(v) for k, v in metrics.items() if v == v}  # filtre NaN
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str) -> None:
        if self.enabled and os.path.exists(path):
            self._mlflow.log_artifact(path)
