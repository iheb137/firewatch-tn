"""Passerelle API de FireWatch TN : point d'entrée unique pour l'interface React.

Relaie les requêtes vers les services d'inférence et agrège leur état.
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

SPREAD_URL = os.environ.get("SPREAD_SERVICE_URL", "http://spread-service:8000")
BURN_URL = os.environ.get("BURN_SERVICE_URL", "http://burn-service:8000")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
TIMEOUT = httpx.Timeout(60.0, connect=5.0)

app = FastAPI(title="FireWatch TN API", version="0.1.0",
              description="Prédiction de propagation et cartographie des incendies de forêt")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])


# Remplaçable dans les tests par un httpx.MockTransport.
TRANSPORT: httpx.AsyncBaseTransport | None = None


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT, transport=TRANSPORT)


async def _forward(resp: httpx.Response):
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise HTTPException(resp.status_code, detail)
    return resp.json()


@app.get("/api/v1/health")
async def health():
    out = {"api": "ok"}
    async with client() as c:
        for name, url in (("spread", SPREAD_URL), ("burn", BURN_URL)):
            try:
                out[name] = (await c.get(f"{url}/health")).json()
            except httpx.HTTPError:
                out[name] = {"status": "unreachable"}
    return out


@app.get("/api/v1/models")
async def models():
    out = {}
    async with client() as c:
        for name, url in (("spread", SPREAD_URL), ("burn", BURN_URL)):
            try:
                r = await c.get(f"{url}/model")
                out[name] = r.json() if r.status_code == 200 else {"loaded": False}
            except httpx.HTTPError:
                out[name] = {"loaded": False, "status": "unreachable"}
    return out


@app.post("/api/v1/spread/predict")
async def spread_predict(request: Request):
    async with client() as c:
        try:
            resp = await c.post(f"{SPREAD_URL}/predict", content=await request.body(),
                                headers={"content-type": "application/json"})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"spread-service injoignable : {exc}") from exc
    return await _forward(resp)


@app.post("/api/v1/burn/segment")
async def burn_segment(file: UploadFile = File(...), threshold: float = 0.5):
    async with client() as c:
        try:
            resp = await c.post(f"{BURN_URL}/segment", params={"threshold": threshold},
                                files={"file": (file.filename, await file.read(),
                                                "application/octet-stream")})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"burn-service injoignable : {exc}") from exc
    return await _forward(resp)
