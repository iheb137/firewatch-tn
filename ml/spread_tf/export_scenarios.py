"""Exporte des tuiles NDWS en scénarios JSON pour l'interface (simulateur de propagation).

    python -m ml.spread_tf.export_scenarios --data-dir <dossier tfrecord> \
        --out data/scenarios/spread/ndws_test.json

Chaque scénario contient les 12 variables brutes (unités physiques, telles que
l'API les attend) et le masque réel du feu à J+1, pour comparer prédit vs réel.
Les tuiles retenues sont celles du split demandé où le feu progresse le plus.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from ml.spread_tf.constants import INPUT_FEATURES, TARGET_FEATURE
from ml.spread_tf.dataset import list_files, parse_example

# Décimales conservées par variable : fichiers légers sans perte utile.
DECIMALS = {"elevation": 0, "th": 1, "vs": 2, "tmmn": 2, "tmmx": 2, "sph": 5, "pr": 2,
            "pdsi": 2, "NDVI": 0, "population": 1, "erc": 1, "PrevFireMask": 0}


def tile_stats(prev: np.ndarray, nxt: np.ndarray, th: np.ndarray, vs: np.ndarray) -> dict:
    return {
        "fire_today_px": int((prev == 1).sum()),
        "fire_tomorrow_px": int((nxt == 1).sum()),
        "new_fire_px": int(((nxt == 1) & (prev != 1)).sum()),
        "unknown_px": int((nxt == -1).sum()),
        "wind_from_deg": float(np.round(np.degrees(np.arctan2(
            np.sin(np.radians(th)).mean(), np.cos(np.radians(th)).mean())) % 360, 1)),
        "wind_speed_ms": float(np.round(vs.mean(), 2)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "eval", "test"])
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--min-fire-today", type=int, default=5)
    p.add_argument("--source", default="NDWS — Google Research (USA, 2012-2020)")
    p.add_argument("--out", default="data/scenarios/spread/ndws_test.json")
    a = p.parse_args()

    candidates = []
    ds = tf.data.TFRecordDataset(list_files(a.data_dir, a.split)).map(parse_example)
    for idx, ex in enumerate(ds):
        prev, nxt = ex["PrevFireMask"].numpy(), ex[TARGET_FEATURE].numpy()
        if (prev == 1).sum() < a.min_fire_today:
            continue
        new_fire = int(((nxt == 1) & (prev != 1)).sum())
        candidates.append((new_fire, idx, {k: v.numpy() for k, v in ex.items()}))
    candidates.sort(key=lambda c: -c[0])

    scenarios = []
    for _new_fire, idx, ex in candidates[: a.n]:
        features = {k: np.round(ex[k].astype(np.float64), DECIMALS[k]).tolist()
                    for k in INPUT_FEATURES}
        scenarios.append({
            "id": f"{a.split}-{idx:05d}",
            "index": idx,
            "features": features,
            "truth": ex[TARGET_FEATURE].astype(np.int8).tolist(),
            "stats": tile_stats(ex["PrevFireMask"], ex[TARGET_FEATURE], ex["th"], ex["vs"]),
        })

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"source": a.source, "split": a.split, "tile_km": 64,
                   "scenarios": scenarios}, fh, separators=(",", ":"))
    size = os.path.getsize(a.out) / 1e6
    print(f"{len(scenarios)} scénarios ({len(candidates)} candidats) -> {a.out} ({size:.1f} Mo)")


if __name__ == "__main__":
    main()
