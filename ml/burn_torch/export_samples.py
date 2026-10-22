"""Exporte des patchs CaBuAr représentatifs en .npz pour l'interface (analyse post-incendie).

    python -m ml.burn_torch.export_samples --h5 data/raw/cabuar/512x512.hdf5 \
        --out data/scenarios/burn

Critères : fold de test, image avant réellement disponible (pas un patch rempli de
zéros) et surface brûlée partielle, pour que le contour du feu soit visible.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ml.burn_torch.constants import PIXEL_AREA_HA
from ml.burn_torch.dataset import _open_h5


def valid_fraction(img: np.ndarray) -> float:
    """Part des pixels non nuls sur toutes les bandes (0 = image absente)."""
    return float((img.max(axis=-1) > 0).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--min-burned", type=float, default=0.15)
    p.add_argument("--max-burned", type=float, default=0.75)
    p.add_argument("--source", default="CaBuAr — Sentinel-2 L2A, Californie (CAL FIRE)")
    p.add_argument("--out", default="data/scenarios/burn")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    picked, skipped_empty = [], 0
    with _open_h5(a.h5) as f:
        candidates = []
        for key, grp in f.items():
            if int(grp.attrs["fold"]) != a.fold or "pre_fire" not in grp:
                continue
            frac = float((grp["mask"][...] > 0).mean())
            if a.min_burned <= frac <= a.max_burned:
                candidates.append((abs(frac - 0.45), key, frac))
        candidates.sort()
        for _, key, frac in candidates:
            grp = f[key]
            pre, post = grp["pre_fire"][...], grp["post_fire"][...]
            if valid_fraction(pre) < 0.98 or valid_fraction(post) < 0.98:
                skipped_empty += 1
                continue
            mask = (grp["mask"][...] > 0).astype(np.uint8)
            name = f"cabuar-f{a.fold}-{key}"
            np.savez_compressed(os.path.join(a.out, f"{name}.npz"), pre=pre, post=post, mask=mask)
            picked.append({"id": name, "file": f"{name}.npz", "burned_fraction": round(frac, 3),
                           "burned_area_ha": round(float(mask.sum()) * PIXEL_AREA_HA, 1),
                           "shape": list(post.shape[:2])})
            if len(picked) >= a.n:
                break

    index = {"source": a.source, "fold": a.fold,
             "pixel_area_ha": PIXEL_AREA_HA, "samples": picked}
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)
    print(f"{len(picked)} patchs exportés dans {a.out} "
          f"({skipped_empty} écartés : image avant ou après incomplète)")


if __name__ == "__main__":
    main()
