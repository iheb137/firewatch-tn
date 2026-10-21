"""Analyse exploratoire de NDWS : statistiques par canal, déséquilibre, exemples.

    python -m ml.spread_tf.eda --data-dir data/raw/ndws --out reports/eda_ndws

Produit ``stats.json`` et des figures PNG (distributions, exemples de tuiles).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.spread_tf.constants import INPUT_FEATURES, TARGET_FEATURE
from ml.spread_tf.dataset import list_files, parse_example


def load_raw(data_dir: str, split: str, limit: int | None) -> dict[str, np.ndarray]:
    ds = tf.data.TFRecordDataset(list_files(data_dir, split)).map(parse_example)
    if limit:
        ds = ds.take(limit)
    buf: dict[str, list] = {k: [] for k in (*INPUT_FEATURES, TARGET_FEATURE)}
    for ex in ds:
        for k in buf:
            buf[k].append(ex[k].numpy())
    return {k: np.stack(v) for k, v in buf.items()}


def label_stats(mask: np.ndarray) -> dict[str, float]:
    n = mask.size
    return {"fire": float(np.sum(mask == 1) / n), "no_fire": float(np.sum(mask == 0) / n),
            "unknown": float(np.sum(mask == -1) / n)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw/ndws")
    p.add_argument("--out", default="reports/eda_ndws")
    p.add_argument("--limit", type=int, default=3000, help="tuiles par split (0 = toutes)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {}
    raw_train = None
    for split in ("train", "eval", "test"):
        raw = load_raw(a.data_dir, split, a.limit or None)
        raw_train = raw if split == "train" else raw_train
        prev, nxt = raw["PrevFireMask"], raw[TARGET_FEATURE]
        both = (prev == 1) & (nxt == 1)
        report[split] = {
            "tiles": int(prev.shape[0]),
            "target": label_stats(nxt),
            "prev_fire": label_stats(prev),
            # part des pixels en feu à J+1 déjà en feu à J (borne de la persistance)
            "next_fire_already_burning": float(both.sum() / max((nxt == 1).sum(), 1)),
        }
    report["channels"] = {
        k: {"min": float(v.min()), "p01": float(np.percentile(v, 0.1)),
            "mean": float(v.mean()), "std": float(v.std()),
            "p999": float(np.percentile(v, 99.9)), "max": float(v.max())}
        for k, v in raw_train.items() if k not in ("PrevFireMask", TARGET_FEATURE)
    }
    (out / "stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({s: report[s] for s in ("train", "eval", "test")}, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : figures ignorées")
        return

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for ax, k in zip(axes.ravel(), [f for f in INPUT_FEATURES if f != "PrevFireMask"]):
        ax.hist(raw_train[k].ravel()[::50], bins=60, color="#E8590C")
        ax.set_title(k)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=120)

    idx = np.argsort(-(raw_train["PrevFireMask"] == 1).sum(axis=(1, 2)))[:4]
    cmap = matplotlib.colors.ListedColormap(["#BDBDBD", "#212529", "#E8590C"])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for row, i in enumerate(idx):
        for col, (k, cm) in enumerate([("elevation", "terrain"), ("NDVI", "Greens"),
                                        ("PrevFireMask", cmap), (TARGET_FEATURE, cmap)]):
            kw = {"vmin": -1, "vmax": 1} if "Mask" in k else {}
            axes[row, col].imshow(raw_train[k][i], cmap=cm, **kw)
            axes[row, col].set_title(k if row == 0 else "")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=120)
    print(f"Figures écrites dans {out}")


if __name__ == "__main__":
    main()
