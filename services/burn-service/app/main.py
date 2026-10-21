"""Service d'inférence du Module B (PyTorch).

POST /segment : fichier ``.npz`` contenant ``post`` (H, W, 12, uint16) et, selon
le modèle, ``pre`` (même forme) → masque brûlé (PNG base64) et surface en hectares.
"""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from ml.burn_torch.constants import PIXEL_AREA_HA

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/models/burn/best.pt"))
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

state: dict = {"model": None, "config": None}


def load_model() -> None:
    if not MODEL_PATH.exists():
        return
    import torch

    from ml.burn_torch.dataset import num_channels
    from ml.burn_torch.model import BurnModelConfig, build_model

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(BurnModelConfig(**{**cfg["model"], "encoder_weights": None},
                                        in_channels=num_channels(cfg["data"]["mode"])))
    model.load_state_dict(ckpt["model"])
    state["model"], state["config"] = model.eval(), cfg


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="FireWatch TN — burn-service", version="0.1.0", lifespan=lifespan)


def mask_to_png_b64(mask: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model")
def model_info():
    if state["model"] is None:
        raise HTTPException(503, "Aucun modèle chargé (MODEL_PATH absent)")
    cfg = state["config"]
    return {"framework": "pytorch", "arch": cfg["model"]["arch"],
            "encoder": cfg["model"]["encoder"], "mode": cfg["data"]["mode"],
            "pixel_area_ha": PIXEL_AREA_HA}


@app.post("/segment")
async def segment(file: UploadFile = File(...), threshold: float = 0.5):
    if state["model"] is None:
        raise HTTPException(503, "Aucun modèle chargé")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Fichier trop volumineux")
    try:
        arrays = np.load(io.BytesIO(data), allow_pickle=False)
        post = arrays["post"]
        pre = arrays["pre"] if "pre" in arrays.files else None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Fichier .npz invalide : {exc}") from exc

    mode = state["config"]["data"]["mode"]
    if mode == "pre_post" and pre is None:
        raise HTTPException(400, "Ce modèle requiert aussi l'image avant incendie ('pre')")
    if post.ndim != 3 or post.shape[-1] != 12 or (pre is not None and pre.shape != post.shape):
        raise HTTPException(400, "Formes attendues : (H, W, 12)")
    h, w = post.shape[:2]
    if h % 32 or w % 32:
        raise HTTPException(400, "Hauteur et largeur doivent être multiples de 32")

    import torch

    from ml.burn_torch.dataset import build_input

    x = build_input(post, pre if mode == "pre_post" else None)[None]
    with torch.no_grad():
        prob = torch.sigmoid(state["model"](x))[0, 0].numpy()
    mask = (prob >= threshold).astype(np.uint8)
    return {
        "burned_area_ha": round(float(mask.sum()) * PIXEL_AREA_HA, 2),
        "burned_fraction": round(float(mask.mean()), 4),
        "threshold": threshold,
        "mask_png_base64": mask_to_png_b64(mask),
        "shape": [h, w],
    }
