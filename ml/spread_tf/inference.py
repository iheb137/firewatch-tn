"""Inférence : prédiction simple et Monte Carlo Dropout (carte d'incertitude)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from ml.spread_tf.model import set_mc_dropout


def predict_proba(model, x: np.ndarray | tf.Tensor) -> np.ndarray:
    return tf.sigmoid(model(x, training=False)).numpy()


def mc_dropout_predict(model, x, n_samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (moyenne, écart-type) des probabilités sur ``n_samples`` passes."""
    if set_mc_dropout(model, True) == 0:
        raise ValueError("Ce modèle n'a pas de couche de dropout : MC Dropout impossible.")
    try:
        samples = np.stack([predict_proba(model, x) for _ in range(n_samples)], axis=0)
    finally:
        set_mc_dropout(model, False)
    return samples.mean(axis=0), samples.std(axis=0)
