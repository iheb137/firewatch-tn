"""Entraînement du Module A.

    python -m ml.spread_tf.train --config configs/spread_tf.yaml
    python -m ml.spread_tf.train --config configs/spread_tf.yaml --set model.name=logreg

Produit dans ``<output_dir>/<run>/`` : ``best.keras``, ``history.csv``,
``config.yaml`` et les journaux TensorBoard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import keras
import tensorflow as tf
import yaml

from ml.common.config import load_config, seed_everything
from ml.common.tracking import Tracker
from ml.spread_tf.dataset import DataConfig, count_records, make_dataset
from ml.spread_tf.losses import MaskedBceDiceLoss
from ml.spread_tf.metrics import training_metrics
from ml.spread_tf.model import ModelConfig, build_model


class _MlflowCallback(keras.callbacks.Callback):
    def __init__(self, tracker: Tracker):
        super().__init__()
        self.tracker = tracker

    def on_epoch_end(self, epoch, logs=None):
        self.tracker.log_metrics(logs or {}, step=epoch)


def build_from_config(cfg: dict) -> tuple[keras.Model, DataConfig]:
    data_cfg = DataConfig(**cfg["data"])
    m = cfg["model"]
    model_cfg = ModelConfig(name=m["name"], filters=tuple(m["filters"]),
                            dropout=m["dropout"], l2=m["l2"])
    return build_model(model_cfg), data_cfg


def train(cfg: dict) -> Path:
    seed_everything(cfg["seed"])
    if cfg["train"].get("mixed_precision"):
        keras.mixed_precision.set_global_policy("mixed_float16")

    model, data_cfg = build_from_config(cfg)
    train_ds = make_dataset(data_cfg, "train")
    val_ds = make_dataset(data_cfg, "eval")

    t = cfg["train"]
    steps_per_epoch = count_records(data_cfg.data_dir, "train") // data_cfg.batch_size
    total_steps = max(steps_per_epoch * t["epochs"], 1)
    schedule = keras.optimizers.schedules.CosineDecay(
        t["learning_rate"], decay_steps=total_steps, alpha=0.01
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=schedule, weight_decay=t["weight_decay"]),
        loss=MaskedBceDiceLoss(**cfg["loss"]),
        metrics=training_metrics(),
    )

    run_name = f"{cfg['model']['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(t["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)

    tracker = Tracker("firewatch-spread", run_name)
    callbacks = [
        keras.callbacks.ModelCheckpoint(out_dir / "best.keras", monitor="val_auc_pr",
                                        mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max",
                                      patience=t["early_stopping_patience"],
                                      restore_best_weights=True),
        keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
        keras.callbacks.TerminateOnNaN(),
        _MlflowCallback(tracker),
    ]
    if importlib.util.find_spec("tensorboard") is not None:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=str(out_dir / "tb")))

    with tracker.run():
        tracker.log_params(cfg)
        history = model.fit(train_ds, validation_data=val_ds, epochs=t["epochs"],
                            callbacks=callbacks, verbose=2)
        best = max(history.history["val_auc_pr"])
        summary = {"run": run_name, "best_val_auc_pr": best,
                   "epochs_run": len(history.history["loss"]),
                   "params": int(model.count_params())}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        tracker.log_metrics({"best_val_auc_pr": best})
        tracker.log_artifact(str(out_dir / "best.keras"))
    print(json.dumps(summary, indent=2))
    if not math.isfinite(best):
        raise RuntimeError("Entraînement divergent (val_auc_pr non fini).")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/spread_tf.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="clé=valeur")
    args = parser.parse_args()
    tf.get_logger().setLevel("ERROR")
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
