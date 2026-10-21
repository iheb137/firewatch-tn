"""Chargement des configurations YAML avec surcharges en ligne de commande.

Exemple : ``--set train.epochs=5 model.dropout=0.2``
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Surcharge invalide '{item}' (attendu clé=valeur)")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def merged(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    out.update(extra)
    return out


def seed_everything(seed: int) -> None:
    """Fixe les graines des frameworks **déjà chargés** uniquement.

    Ne jamais importer TensorFlow et PyTorch dans le même processus : leurs
    bibliothèques natives entrent en conflit (segfault silencieux).
    """
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        import keras

        keras.utils.set_random_seed(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Aplatissement pour la journalisation MLflow."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
