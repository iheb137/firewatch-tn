"""Passerelle API : relais et propagation des erreurs (services simulés)."""

import httpx
from fastapi.testclient import TestClient

from tests.api._helpers import load_app


def test_gateway_forwards_and_reports_errors():
    mod = load_app("api", "api_main")

    def handler(request: httpx.Request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model_loaded": True})
        if request.url.path == "/predict":
            return httpx.Response(503, json={"detail": "Aucun modèle chargé"})
        return httpx.Response(404)

    mod.TRANSPORT = httpx.MockTransport(handler)
    c = TestClient(mod.app)
    assert c.get("/api/v1/health").json()["spread"]["model_loaded"] is True
    r = c.post("/api/v1/spread/predict", json={"features": {}})
    assert r.status_code == 503 and "modèle" in r.json()["detail"]
