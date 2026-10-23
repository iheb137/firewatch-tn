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


def _png_b64(img, mode: str) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img, mode=mode).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def mask_to_png_b64(mask: np.ndarray) -> str:
    return _png_b64((mask * 255).astype(np.uint8), "L")


def rgb_preview(img: np.ndarray, scale: float | None = None) -> tuple[str, float]:
    """Couleurs naturelles B04/B03/B02, étirées au 99e centile (partagé avant/après)."""
    rgb = img[..., [3, 2, 1]].astype(np.float32)
    if scale is None:
        valid = rgb[rgb.max(axis=-1) > 0]
        scale = float(np.percentile(valid, 99)) if valid.size else 1.0
    out = np.clip(rgb / max(scale, 1.0), 0, 1) ** 0.8
    return _png_b64((out * 255).astype(np.uint8), "RGB"), scale


def mask_edges(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Pixels du masque touchant l'extérieur (contour d'environ `width` pixels)."""
    inner = mask.copy()
    for _ in range(width):
        p = np.pad(inner, 1, constant_values=False)
        inner = inner & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~inner


def overlay_png_b64(prob: np.ndarray, threshold: float) -> str:
    """Calque RGBA : remplissage léger (le relief brûlé reste visible) et contour opaque."""
    mask = prob >= threshold
    alpha = np.where(mask, 70, 0)
    alpha[mask_edges(mask)] = 255
    alpha = alpha.astype(np.uint8)
    rgba = np.zeros((*prob.shape, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 239, 68, 35, alpha
    return _png_b64(rgba, "RGBA")


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
async def segment(file: UploadFile = File(...), threshold: float = 0.5,
                  previews: bool = False):
    if state["model"] is None:
        raise HTTPException(503, "Aucun modèle chargé")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Fichier trop volumineux")
    try:
        arrays = np.load(io.BytesIO(data), allow_pickle=False)
        post = arrays["post"]
        pre = arrays["pre"] if "pre" in arrays.files else None
        truth = arrays["mask"] if "mask" in arrays.files else None
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
    result = {
        "burned_area_ha": round(float(mask.sum()) * PIXEL_AREA_HA, 2),
        "burned_fraction": round(float(mask.mean()), 4),
        "threshold": threshold,
        "mask_png_base64": mask_to_png_b64(mask),
        "shape": [h, w],
    }
    if truth is not None:  # vérité terrain fournie (échantillons de démo) : on note le résultat
        t = truth.reshape(h, w) > 0
        m = mask.astype(bool)
        tp, fp, fn = int((m & t).sum()), int((m & ~t).sum()), int((~m & t).sum())
        result["truth"] = {
            "burned_area_ha": round(float(t.sum()) * PIXEL_AREA_HA, 2),
            "iou": round(tp / max(tp + fp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "mask_png_base64": mask_to_png_b64(t.astype(np.uint8)),
        }
    if previews:
        post_png, scale = rgb_preview(post)
        result["post_rgb_png_base64"] = post_png
        if pre is not None:
            result["pre_rgb_png_base64"] = rgb_preview(pre, scale)[0]
        result["overlay_png_base64"] = overlay_png_b64(prob, threshold)
    return result
