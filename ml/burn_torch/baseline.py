"""Baseline sans apprentissage : seuillage du dNBR.

    python -m ml.burn_torch.baseline --config configs/burn_torch.yaml

Le seuil est choisi sur les folds d'entraînement (max IoU), puis évalué sur le
fold de test — même protocole que le réseau, pour une comparaison honnête.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml.burn_torch.dataset import BurnDataConfig, CaBuArDataset
from ml.burn_torch.metrics import SegMeter
from ml.burn_torch.train import fold_split
from ml.common.config import load_config

DNBR_CHANNEL = 26  # dernier canal en mode pre_post


def evaluate_threshold(ds: CaBuArDataset, threshold: float) -> dict[str, float]:
    meter = SegMeter(threshold=threshold)
    for i in range(len(ds)):
        x, y = ds[i]
        meter.update(x[DNBR_CHANNEL][None, None], y[None])
    return meter.compute()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/burn_torch.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--max-patches", type=int, default=400,
                   help="patchs d'entraînement utilisés pour régler le seuil")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    if cfg["data"]["mode"] != "pre_post":
        raise SystemExit("La baseline dNBR nécessite le mode pre_post.")

    data_cfg = BurnDataConfig(**{**cfg["data"], "crop": None, "augment": False})
    train_f, _, test_f = fold_split(cfg["split"]["test_fold"], cfg["split"]["val_fold"])
    train_ds = CaBuArDataset(data_cfg, train_f, training=False)
    if len(train_ds) > a.max_patches:
        rng = np.random.default_rng(cfg["seed"])
        train_ds.keys = sorted(rng.choice(train_ds.keys, a.max_patches, replace=False).tolist())

    grid = np.round(np.arange(0.0, 0.66, 0.02), 2)
    scores = {float(t): evaluate_threshold(train_ds, float(t))["iou"] for t in grid}
    best = max(scores, key=scores.get)
    with torch.no_grad():
        test = evaluate_threshold(CaBuArDataset(data_cfg, test_f, training=False), best)
    result = {"method": "dNBR threshold", "threshold_from_train": best, "test": test}

    out = Path(cfg["train"]["output_dir"]) / f"baseline_dnbr_f{cfg['split']['test_fold']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
