"""Fichier HDF5 synthétique au format CaBuAr (tests, CI, démo).

    python -m ml.burn_torch.synthetic --out data/synthetic/cabuar.hdf5 --patches 40 --size 64

Les zones brûlées font chuter le proche infrarouge (B08) et monter le SWIR (B12)
dans l'image post-incendie, comme dans la réalité.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

from ml.burn_torch.constants import BAND


def make_patch(rng: np.random.Generator, size: int):
    base = rng.normal(1500, 200, (size, size, 12)).clip(0, None)
    base[..., BAND["B08"]] += 2000  # végétation : NIR élevé
    pre = base + rng.normal(0, 50, base.shape)
    post = base + rng.normal(0, 50, base.shape)
    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size, 1), np.uint16)
    for _ in range(rng.integers(1, 3)):
        cy, cx = rng.integers(0, size, 2)
        r = rng.uniform(size / 10, size / 4)
        mask[((yy - cy) ** 2 + (xx - cx) ** 2 <= r**2), 0] = 1
    burnt = mask[..., 0] == 1
    post[burnt, BAND["B08"]] *= 0.45
    post[burnt, BAND["B12"]] *= 1.8
    return pre.clip(0, 65535).astype(np.uint16), post.clip(0, 65535).astype(np.uint16), mask


def write(path: str, n: int, size: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        for i in range(n):
            pre, post, mask = make_patch(rng, size)
            grp = f.create_group(f"uid{i}_0")
            grp.attrs["fold"] = i % 5
            grp["post_fire"], grp["mask"] = post, mask
            if i % 7 != 0:  # comme dans CaBuAr, certains patchs n'ont pas d'image avant
                grp["pre_fire"] = pre
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic/cabuar.hdf5")
    p.add_argument("--patches", type=int, default=40)
    p.add_argument("--size", type=int, default=64)
    a = p.parse_args()
    print(write(a.out, a.patches, a.size))


if __name__ == "__main__":
    main()
