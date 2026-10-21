"""Évaluation rigoureuse du Module A.

    python -m ml.spread_tf.evaluate --run runs/spread_tf/<run>
    python -m ml.spread_tf.evaluate --baseline persistence --config configs/spread_tf.yaml

Protocole :
* pixels -1 exclus de toutes les métriques ;
* seuil de décision choisi sur la **validation** (max F1), puis appliqué tel quel au test ;
* AUC-PR (indépendante du seuil) rapportée en métrique principale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from ml.common.config import load_config
from ml.spread_tf import losses, metrics, model  # noqa: F401  (enregistre les objets Keras)
from ml.spread_tf.constants import MODEL_CHANNELS
from ml.spread_tf.dataset import DataConfig, make_dataset
from ml.spread_tf.metrics import average_precision, best_threshold, confusion_at, flatten_valid

PREV_FIRE_IDX = MODEL_CHANNELS.index("PrevFireMask")


def collect(predict_fn, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for x, y in ds:
        ys.append(y.numpy())
        ps.append(predict_fn(x))
    return np.concatenate(ys), np.concatenate(ps)


def persistence(x: tf.Tensor) -> np.ndarray:
    """Baseline : le feu de demain est le feu d'aujourd'hui."""
    return np.clip(x[..., PREV_FIRE_IDX : PREV_FIRE_IDX + 1].numpy(), 0.0, 1.0)


def evaluate(predict_fn, data_cfg: DataConfig, tune_threshold: bool = True) -> dict:
    y_val, p_val = flatten_valid(*collect(predict_fn, make_dataset(data_cfg, "eval")))
    y_test, p_test = flatten_valid(*collect(predict_fn, make_dataset(data_cfg, "test")))
    thr = best_threshold(y_val, p_val) if tune_threshold else 0.5
    return {
        "threshold_from_val": thr,
        "val": {"auc_pr": average_precision(y_val, p_val), **confusion_at(y_val, p_val, thr)},
        "test": {"auc_pr": average_precision(y_test, p_test),
                 **confusion_at(y_test, p_test, thr)},
        "test_positive_rate": float(np.mean(y_test == 1)),
        "test_valid_pixels": int(y_test.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="Dossier d'un entraînement (contient best.keras)")
    parser.add_argument("--baseline", choices=["persistence"])
    parser.add_argument("--config", default="configs/spread_tf.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    if args.run:
        run = Path(args.run)
        cfg = load_config(run / "config.yaml", args.set)
        net = keras.models.load_model(run / "best.keras")
        def predict_fn(x):
            return tf.sigmoid(net(x, training=False)).numpy()
        out_path = run / "eval.json"
        result = evaluate(predict_fn, DataConfig(**cfg["data"]))
    elif args.baseline == "persistence":
        cfg = load_config(args.config, args.set)
        result = evaluate(persistence, DataConfig(**cfg["data"]), tune_threshold=False)
        out_path = Path(cfg["train"]["output_dir"]) / "baseline_persistence.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        parser.error("--run ou --baseline requis")

    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
