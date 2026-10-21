"""Dataset PyTorch pour CaBuAr (lecture directe du HDF5, sans script Hugging Face).

Entrée du modèle, pour le mode ``pre_post`` (par défaut) :
    12 bandes avant + 12 bandes après + NBR avant + NBR après + dNBR = 27 canaux.
Mode ``post`` : 12 bandes après + NBR après = 13 canaux (utilise tous les patchs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.burn_torch.constants import REFLECTANCE_SCALE
from ml.burn_torch.indices import nbr


def num_channels(mode: str) -> int:
    return {"pre_post": 27, "post": 13}[mode]


def _open_h5(path: str):
    import h5py
    import hdf5plugin  # noqa: F401  (enregistre le filtre BZip2)

    return h5py.File(path, "r")


def index_patches(path: str, folds: Sequence[int], mode: str) -> list[str]:
    """Liste des clés des patchs appartenant aux folds demandés."""
    keys = []
    with _open_h5(path) as f:
        for key, grp in f.items():
            if int(grp.attrs["fold"]) not in folds:
                continue
            if mode == "pre_post" and "pre_fire" not in grp:
                continue
            keys.append(key)
    return sorted(keys)


def build_input(post: np.ndarray, pre: np.ndarray | None) -> torch.Tensor:
    """(H, W, 12) uint16 → (C, H, W) float32, réflectance + indices."""
    post_t = torch.from_numpy(post.astype(np.float32)).permute(2, 0, 1) / REFLECTANCE_SCALE
    post_t = post_t.clamp(0.0, 1.5)
    if pre is None:
        return torch.cat([post_t, nbr(post_t)[None]], dim=0)
    pre_t = torch.from_numpy(pre.astype(np.float32)).permute(2, 0, 1) / REFLECTANCE_SCALE
    pre_t = pre_t.clamp(0.0, 1.5)
    nbr_pre, nbr_post = nbr(pre_t), nbr(post_t)
    return torch.cat([pre_t, post_t, nbr_pre[None], nbr_post[None],
                      (nbr_pre - nbr_post)[None]], dim=0)


def random_flip_rot(x: torch.Tensor, y: torch.Tensor, gen: torch.Generator):
    """Rotations 90° et miroirs : sans risque ici (pas de variable orientée)."""
    k = int(torch.randint(0, 4, (1,), generator=gen))
    x, y = torch.rot90(x, k, dims=(1, 2)), torch.rot90(y, k, dims=(1, 2))
    if torch.rand(1, generator=gen) < 0.5:
        x, y = x.flip(2), y.flip(2)
    if torch.rand(1, generator=gen) < 0.5:
        x, y = x.flip(1), y.flip(1)
    return x, y


def random_crop(x: torch.Tensor, y: torch.Tensor, size: int, gen: torch.Generator):
    h, w = x.shape[1:]
    if size >= h:
        return x, y
    i = int(torch.randint(0, h - size + 1, (1,), generator=gen))
    j = int(torch.randint(0, w - size + 1, (1,), generator=gen))
    return x[:, i : i + size, j : j + size], y[:, i : i + size, j : j + size]


@dataclass
class BurnDataConfig:
    h5_path: str
    mode: str = "pre_post"
    crop: int | None = 256
    augment: bool = True
    seed: int = 42


class CaBuArDataset(Dataset):
    def __init__(self, cfg: BurnDataConfig, folds: Sequence[int], training: bool):
        self.cfg = cfg
        self.training = training
        self.keys = index_patches(cfg.h5_path, folds, cfg.mode)
        if not self.keys:
            raise ValueError(f"Aucun patch pour les folds {list(folds)} en mode {cfg.mode}")
        self._h5 = None  # ouvert paresseusement : un handle par worker DataLoader
        self._gen = torch.Generator().manual_seed(cfg.seed)

    def __len__(self) -> int:
        return len(self.keys)

    def _file(self):
        if self._h5 is None:
            self._h5 = _open_h5(self.cfg.h5_path)
        return self._h5

    def __getitem__(self, i: int):
        grp = self._file()[self.keys[i]]
        pre = grp["pre_fire"][...] if self.cfg.mode == "pre_post" else None
        x = build_input(grp["post_fire"][...], pre)
        y = torch.from_numpy((grp["mask"][...] > 0).astype(np.float32)).permute(2, 0, 1)
        if self.training:
            if self.cfg.crop:
                x, y = random_crop(x, y, self.cfg.crop, self._gen)
            if self.cfg.augment:
                x, y = random_flip_rot(x, y, self._gen)
        return x.contiguous(), y.contiguous()


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds._h5 = None
    ds._gen = torch.Generator().manual_seed(info.seed % 2**31)  # graine distincte par worker
