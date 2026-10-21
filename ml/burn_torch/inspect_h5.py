"""Inspection rapide du fichier CaBuAr après téléchargement (à lancer une fois).

    python -m ml.burn_torch.inspect_h5 data/raw/cabuar/512x512.hdf5

Affiche le nombre de patchs par fold, la part avec image pré-incendie, les
formes/dtypes et des statistiques par bande pour valider l'ordre des bandes
(B08 doit être nettement plus élevée que B04 sur la végétation saine).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from ml.burn_torch.constants import S2_BANDS
from ml.burn_torch.dataset import _open_h5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--sample", type=int, default=20)
    a = p.parse_args()

    folds, with_pre, shapes = Counter(), Counter(), {}
    band_means = []
    with _open_h5(a.path) as f:
        for i, grp in enumerate(f.values()):
            fold = int(grp.attrs["fold"])
            folds[fold] += 1
            with_pre[fold] += int("pre_fire" in grp)
            if i < a.sample:
                for name in grp:
                    shapes[name] = (grp[name].shape, str(grp[name].dtype))
                post = grp["post_fire"][...].astype(np.float32)
                band_means.append(post.reshape(-1, post.shape[-1]).mean(axis=0))
                if i == 0:
                    mask = grp["mask"][...]
                    shapes["mask_values"] = sorted(np.unique(mask).tolist())[:10]
    report = {
        "patches_per_fold": dict(sorted(folds.items())),
        "with_pre_fire_per_fold": dict(sorted(with_pre.items())),
        "arrays": {k: str(v) for k, v in shapes.items()},
        "post_fire_band_means": dict(zip(S2_BANDS, np.mean(band_means, 0).round(1).tolist())),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
