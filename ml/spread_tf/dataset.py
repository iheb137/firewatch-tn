"""Pipeline tf.data pour Next Day Wildfire Spread.

Chaque exemple TFRecord contient 13 variables float32 de forme (64, 64) :
12 entrées + ``FireMask`` (cible à J+1). Le pipeline :

1. parse l'exemple ;
2. (train) recadrage aléatoire + rotations/miroirs avec correction du vent ;
3. découpe la direction du vent en sin/cos, coupe et normalise les autres variables ;
4. renvoie ``(x, y)`` avec x : (H, W, 13) et y : (H, W, 1) où y ∈ {-1, 0, 1}.

Les pixels ``-1`` (nuages, données manquantes) sont conservés dans ``y`` :
la perte et les métriques les masquent.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass

import tensorflow as tf

from ml.spread_tf import augment
from ml.spread_tf.constants import (
    DATA_STATS,
    INPUT_FEATURES,
    SPLIT_PATTERNS,
    TARGET_FEATURE,
    TILE_SIZE,
)

ALL_FEATURES = (*INPUT_FEATURES, TARGET_FEATURE)


@dataclass(frozen=True)
class DataConfig:
    data_dir: str
    batch_size: int = 64
    train_crop: int | None = 32  # None = tuile complète 64×64
    eval_crop: int | None = None
    augment: bool = True
    shuffle_buffer: int = 2048
    seed: int = 42
    cache: bool = True


def feature_spec(tile_size: int = TILE_SIZE) -> dict[str, tf.io.FixedLenFeature]:
    return {
        name: tf.io.FixedLenFeature(shape=[tile_size, tile_size], dtype=tf.float32)
        for name in ALL_FEATURES
    }


def parse_example(serialized: tf.Tensor, tile_size: int = TILE_SIZE) -> dict[str, tf.Tensor]:
    return tf.io.parse_single_example(serialized, feature_spec(tile_size))


def normalize(features: dict[str, tf.Tensor]) -> tf.Tensor:
    """Transforme le dict brut en tenseur d'entrée (H, W, 13) normalisé."""
    channels = []
    for name in INPUT_FEATURES:
        value = features[name]
        if name == "th":
            rad = value * (math.pi / 180.0)
            channels += [tf.sin(rad), tf.cos(rad)]
        elif name == "PrevFireMask":
            channels.append(value)  # {-1, 0, 1} gardé tel quel
        else:
            lo, hi, mean, std = DATA_STATS[name]
            channels.append((tf.clip_by_value(value, lo, hi) - mean) / std)
    return tf.stack(channels, axis=-1)


def to_model_inputs(
    features: dict[str, tf.Tensor],
    *,
    crop: int | None,
    augment_geometry: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    if crop is not None and crop < TILE_SIZE:
        features = augment.random_crop(features, crop)
    if augment_geometry:
        features = augment.random_geometric(features)
    x = normalize(features)
    y = features[TARGET_FEATURE][..., None]
    return x, y


def center_crop(x: tf.Tensor, y: tf.Tensor, size: int) -> tuple[tf.Tensor, tf.Tensor]:
    off = (TILE_SIZE - size) // 2
    return x[off : off + size, off : off + size], y[off : off + size, off : off + size]


def list_files(data_dir: str, split: str) -> list[str]:
    pattern = os.path.join(data_dir, SPLIT_PATTERNS[split])
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier '{SPLIT_PATTERNS[split]}' dans {data_dir}. "
            "Lancez `make data-ndws` (voir README)."
        )
    return files


def count_records(data_dir: str, split: str) -> int:
    """Compte les exemples sans les décoder (rapide)."""
    return sum(1 for _ in tf.data.TFRecordDataset(list_files(data_dir, split)))


def make_dataset(cfg: DataConfig, split: str) -> tf.data.Dataset:
    training = split == "train"
    files = list_files(cfg.data_dir, split)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if cfg.cache:
        ds = ds.cache()  # cache des exemples bruts : l'augmentation reste aléatoire
    if training:
        ds = ds.shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda f: to_model_inputs(f, crop=cfg.train_crop, augment_geometry=cfg.augment),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda f: to_model_inputs(f, crop=None, augment_geometry=False),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if cfg.eval_crop is not None and cfg.eval_crop < TILE_SIZE:
            ds = ds.map(lambda x, y: center_crop(x, y, cfg.eval_crop))
    return ds.batch(cfg.batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)
