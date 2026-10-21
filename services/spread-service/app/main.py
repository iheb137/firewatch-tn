"""Service d'inférence du Module A (TensorFlow).

POST /predict : variables brutes d'une tuile (unités physiques, comme dans NDWS)
→ carte de probabilité de feu à J+1, masque binaire et, en option, incertitude MC Dropout.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ml.spread_tf.constants import INPUT_FEATURES

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/spread"))
MAX_MC_SAMPLES = 50

state: dict = {"model": None, "meta": {}}


def load_model() -> None:
    path = MODEL_DIR / "model.keras"
    if not path.exists():
        return
    import keras

    from ml.spread_tf import losses, metrics, model  # noqa: F401  (objets Keras enregistrés)

    state["model"] = keras.models.load_model(path, compile=False)
    meta = MODEL_DIR / "meta.json"
    state["meta"] = json.loads(meta.read_text()) if meta.exists() else {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="FireWatch TN — spread-service", version="0.1.0", lifespan=lifespan)


class SpreadRequest(BaseModel):
    features: dict[str, list[list[float]]] = Field(
        description=f"Une grille H×W par variable : {', '.join(INPUT_FEATURES)}"
    )
    mc_samples: int = Field(0, ge=0, le=MAX_MC_SAMPLES,
                            description="0 = prédiction simple ; >0 = MC Dropout")

    @field_validator("features")
    @classmethod
    def check_features(cls, v):
        missing = set(INPUT_FEATURES) - set(v)
        if missing:
            raise ValueError(f"Variables manquantes : {sorted(missing)}")
        shapes = {np.asarray(v[k]).shape for k in INPUT_FEATURES}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError("Toutes les variables doivent être des grilles 2D de même taille")
        h, w = next(iter(shapes))
        if h % 8 or w % 8:
            raise ValueError("Hauteur et largeur doivent être multiples de 8")
        return v


class SpreadResponse(BaseModel):
    probability: list[list[float]]
    mask: list[list[int]]
    threshold: float
    uncertainty: list[list[float]] | None = None
    model_run: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model")
def model_info():
    if state["model"] is None:
        raise HTTPException(503, "Aucun modèle chargé (MODEL_DIR/model.keras absent)")
    return {"framework": "tensorflow", **state["meta"]}


@app.post("/predict", response_model=SpreadResponse)
def predict(req: SpreadRequest):
    net = state["model"]
    if net is None:
        raise HTTPException(503, "Aucun modèle chargé")
    import tensorflow as tf

    from ml.spread_tf.dataset import normalize
    from ml.spread_tf.inference import mc_dropout_predict, predict_proba

    raw = {k: tf.constant(np.asarray(req.features[k], np.float32)) for k in INPUT_FEATURES}
    x = normalize(raw)[None]
    if req.mc_samples:
        mean, std = mc_dropout_predict(net, x, req.mc_samples)
        prob, unc = mean[0, ..., 0], std[0, ..., 0]
    else:
        prob, unc = predict_proba(net, x)[0, ..., 0], None
    thr = float(state["meta"].get("threshold", 0.5))
    return SpreadResponse(
        probability=np.round(prob, 4).tolist(),
        mask=(prob >= thr).astype(int).tolist(),
        threshold=thr,
        uncertainty=None if unc is None else np.round(unc, 4).tolist(),
        model_run=state["meta"].get("run"),
    )
