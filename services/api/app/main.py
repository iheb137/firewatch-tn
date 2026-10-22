"""Passerelle API de FireWatch TN : point d'entrée unique pour l'interface React.

Relaie les requêtes vers les services d'inférence et agrège leur état.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

SPREAD_URL = os.environ.get("SPREAD_SERVICE_URL", "http://spread-service:8000")
BURN_URL = os.environ.get("BURN_SERVICE_URL", "http://burn-service:8000")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
TIMEOUT = httpx.Timeout(60.0, connect=5.0)
SCENARIOS_DIR = Path(os.environ.get("SCENARIOS_DIR", "/data/scenarios"))
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")

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
            resp = await c.post(f"{BURN_URL}/segment",
                                params={"threshold": threshold, "previews": "true"},
                                files={"file": (file.filename, await file.read(),
                                                "application/octet-stream")})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"burn-service injoignable : {exc}") from exc
    return await _forward(resp)


# ---------- scénarios de démonstration (fichiers produits par les scripts d'export) ----------

@lru_cache(maxsize=8)
def _spread_files() -> dict[str, dict]:
    """Charge data/scenarios/spread/*.json une seule fois : {id: scénario + source}."""
    out: dict[str, dict] = {}
    for path in sorted((SCENARIOS_DIR / "spread").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for sc in doc.get("scenarios", []):
            out[sc["id"]] = {**sc, "source": doc.get("source", path.stem)}
    return out


@app.get("/api/v1/scenarios/spread")
async def spread_scenarios():
    return [{"id": k, "source": v["source"], "stats": v.get("stats", {})}
            for k, v in _spread_files().items()]


@app.get("/api/v1/scenarios/spread/{scenario_id}")
async def spread_scenario(scenario_id: str):
    sc = _spread_files().get(scenario_id)
    if sc is None:
        raise HTTPException(404, f"Scénario inconnu : {scenario_id}")
    return sc


def _burn_index() -> dict:
    path = SCENARIOS_DIR / "burn" / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"samples": []}


@app.get("/api/v1/scenarios/burn")
async def burn_samples():
    return _burn_index()


@app.post("/api/v1/scenarios/burn/{sample_id}/segment")
async def burn_sample_segment(sample_id: str, threshold: float = 0.5):
    if not SAFE_ID.match(sample_id):
        raise HTTPException(400, "Identifiant invalide")
    sample = next((s for s in _burn_index()["samples"] if s["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(404, f"Échantillon inconnu : {sample_id}")
    data = (SCENARIOS_DIR / "burn" / sample["file"]).read_bytes()
    async with client() as c:
        try:
            resp = await c.post(f"{BURN_URL}/segment",
                                params={"threshold": threshold, "previews": "true"},
                                files={"file": (sample["file"], data, "application/octet-stream")})
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"burn-service injoignable : {exc}") from exc
    return {**(await _forward(resp)), "sample": sample}
