"""Génère des TFRecords synthétiques au **schéma exact** de NDWS.

Usage : tests automatisés, CI, et démo de l'interface sans télécharger 2 Go.

    python -m ml.spread_tf.synthetic --out data/synthetic/ndws --train 256 --eval 64 --test 64

Le feu de J+1 est le feu de J étendu dans la direction **vers laquelle** souffle
le vent, avec une probabilité qui croît avec la sécheresse (ERC) : un modèle
correct doit donc battre la persistance et exploiter le vent.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

from ml.spread_tf.constants import INPUT_FEATURES, SPLIT_PATTERNS, TARGET_FEATURE, TILE_SIZE


def _smooth_noise(rng: np.random.Generator, size: int, scale: int = 8) -> np.ndarray:
    coarse = rng.normal(size=(size // scale + 2, size // scale + 2))
    img = np.kron(coarse, np.ones((scale, scale)))[:size, :size]
    return (img - img.mean()) / (img.std() + 1e-6)


def make_tile(rng: np.random.Generator, size: int = TILE_SIZE) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    th = rng.uniform(0, 360)  # d'où vient le vent
    vs = rng.uniform(1.0, 9.0)
    erc = rng.uniform(10, 100)

    prev = np.zeros((size, size), np.float32)
    for _ in range(rng.integers(1, 4)):
        cy, cx = rng.integers(12, size - 12, size=2)
        r = rng.uniform(2, 5)
        prev[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = 1.0

    # Le feu progresse dans la direction OPPOSÉE à l'origine du vent.
    # Vecteur « vers » en coordonnées image : est = +colonne, nord = -ligne.
    to = np.deg2rad(th + 180.0)
    dx, dy = np.sin(to), -np.cos(to)
    steps = int(1 + vs / 3 + erc / 50)
    nxt = prev.copy()
    for s in range(1, steps + 1):
        shifted = np.roll(np.roll(prev, int(round(dy * s)), axis=0), int(round(dx * s)), axis=1)
        keep = rng.random((size, size)) < 0.75
        nxt = np.maximum(nxt, shifted * keep)
    burnt_out = (prev == 1) & (rng.random((size, size)) < 0.3)
    nxt[burnt_out] = 0.0

    unknown = rng.random((size, size)) < 0.02
    nxt[unknown] = -1.0
    prev_unknown = rng.random((size, size)) < 0.01
    prev[prev_unknown] = -1.0

    n = lambda: _smooth_noise(rng, size)  # noqa: E731
    return {
        "elevation": np.clip(600 + 400 * n(), 0, 3000),
        "th": np.full((size, size), th) + rng.normal(0, 5, (size, size)),
        "vs": np.full((size, size), vs) + rng.normal(0, 0.2, (size, size)),
        "tmmn": 285 + 4 * n(),
        "tmmx": 300 + 5 * n(),
        "sph": np.clip(0.007 + 0.002 * n(), 0, 1),
        "pr": np.clip(rng.exponential(0.5, (size, size)), 0, 40),
        "pdsi": np.clip(-2 + 1.5 * n(), -6, 7),
        "NDVI": np.clip(5000 + 2000 * n(), -9000, 9900),
        "population": np.clip(rng.exponential(10, (size, size)), 0, 2500),
        "erc": np.full((size, size), erc) + rng.normal(0, 2, (size, size)),
        "PrevFireMask": prev,
        TARGET_FEATURE: nxt,
    }


def serialize(tile: dict[str, np.ndarray]) -> bytes:
    feats = {
        k: tf.train.Feature(float_list=tf.train.FloatList(value=v.astype(np.float32).ravel()))
        for k, v in tile.items()
    }
    return tf.train.Example(features=tf.train.Features(feature=feats)).SerializeToString()


def write_split(out_dir: str, split: str, n: int, seed: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SPLIT_PATTERNS[split].replace("*", "00"))
    rng = np.random.default_rng(seed)
    with tf.io.TFRecordWriter(path) as w:
        for _ in range(n):
            w.write(serialize(make_tile(rng)))
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic/ndws")
    p.add_argument("--train", type=int, default=256)
    p.add_argument("--eval", type=int, default=64)
    p.add_argument("--test", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert set(INPUT_FEATURES) | {TARGET_FEATURE} == set(make_tile(np.random.default_rng(0)))
    for i, (split, n) in enumerate([("train", a.train), ("eval", a.eval), ("test", a.test)]):
        print(write_split(a.out, split, n, a.seed + i))


if __name__ == "__main__":
    main()
