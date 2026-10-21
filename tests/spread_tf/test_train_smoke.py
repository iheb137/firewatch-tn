"""Entraînement de fumée de bout en bout sur données synthétiques (≈1 min CPU)."""

import json

import keras
import pytest

from ml.common.config import load_config
from ml.spread_tf.dataset import DataConfig
from ml.spread_tf.evaluate import evaluate, persistence
from ml.spread_tf.train import train


@pytest.mark.slow
def test_train_save_reload_evaluate(ndws_dir, tmp_path):
    cfg = load_config("configs/spread_tf.yaml", [
        f"data.data_dir={ndws_dir}", "data.batch_size=16", "data.shuffle_buffer=64",
        "model.filters=[8,16,32]", "train.epochs=3", f"train.output_dir={tmp_path}",
    ])
    run = train(cfg)
    assert (run / "best.keras").exists()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["best_val_auc_pr"] > 0

    model = keras.models.load_model(run / "best.keras")  # objets personnalisés enregistrés
    import tensorflow as tf

    result = evaluate(lambda x: tf.sigmoid(model(x)).numpy(), DataConfig(**cfg["data"]))
    base = evaluate(persistence, DataConfig(**cfg["data"]), tune_threshold=False)
    assert 0 <= result["test"]["f1"] <= 1
    assert base["test"]["auc_pr"] > 0
