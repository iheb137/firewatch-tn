"""Export du meilleur modèle pour le service d'inférence.

    python -m ml.spread_tf.export --run runs/spread_tf/<run> --out models/spread

Écrit ``model.keras`` (avec logits) et ``meta.json`` (canaux, seuil, métriques).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ml.spread_tf.constants import INPUT_FEATURES, MODEL_CHANNELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default="models/spread")
    args = parser.parse_args()

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "best.keras", out / "model.keras")
    evaluation = json.loads((run / "eval.json").read_text()) if (run / "eval.json").exists() else {}
    meta = {
        "run": run.name,
        "raw_features": list(INPUT_FEATURES),
        "model_channels": list(MODEL_CHANNELS),
        "threshold": evaluation.get("threshold_from_val", 0.5),
        "test_metrics": evaluation.get("test", {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Modèle exporté dans {out}")


if __name__ == "__main__":
    main()
