"""Métriques masquées pour l'entraînement et l'évaluation.

L'accuracy est volontairement absente : avec ~1-2 % de pixels en feu, un modèle
qui prédit « pas de feu » partout dépasserait 98 %.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


@keras.saving.register_keras_serializable(package="firewatch")
class MaskedMetric(keras.metrics.Metric):
    """Enveloppe une métrique Keras binaire : sigmoïde sur les logits + masque des -1."""

    def __init__(self, kind: str = "auc_pr", threshold: float = 0.5, name: str | None = None,
                 **kwargs):
        super().__init__(name=name or kind, **kwargs)
        self.kind = kind
        self.threshold = threshold
        if kind == "auc_pr":
            self.inner = keras.metrics.AUC(curve="PR", num_thresholds=500)
        elif kind == "precision":
            self.inner = keras.metrics.Precision(thresholds=threshold)
        elif kind == "recall":
            self.inner = keras.metrics.Recall(thresholds=threshold)
        else:
            raise ValueError(f"Métrique inconnue : {kind}")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        self.inner.update_state(
            tf.reshape(tf.clip_by_value(y_true, 0.0, 1.0), [-1]),
            tf.reshape(tf.sigmoid(tf.cast(y_pred, tf.float32)), [-1]),
            sample_weight=tf.reshape(mask, [-1]),
        )

    def result(self):
        return self.inner.result()

    def reset_state(self):
        self.inner.reset_state()

    def get_config(self):
        return {**super().get_config(), "kind": self.kind, "threshold": self.threshold}


def training_metrics() -> list[keras.metrics.Metric]:
    return [MaskedMetric("auc_pr"), MaskedMetric("precision"), MaskedMetric("recall")]


# ---------- évaluation hors-ligne (numpy) ----------

def flatten_valid(y_true: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = y_true.reshape(-1)
    probs = probs.reshape(-1)
    valid = y_true != -1
    return y_true[valid].astype(np.int8), probs[valid].astype(np.float32)


def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "iou": iou}


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """AUC-PR (average precision, sans interpolation), calcul exact par tri."""
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order] == 1
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float(np.sum(precision[y_sorted]) / n_pos)


def best_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Seuil maximisant le F1 — à choisir sur la VALIDATION, jamais sur le test."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    scores = [confusion_at(y, p, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(scores))])
