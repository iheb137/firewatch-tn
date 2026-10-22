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


def test_gateway_serves_spread_scenarios(tmp_path, monkeypatch):
    import json as _json

    (tmp_path / "spread").mkdir()
    (tmp_path / "spread" / "demo.json").write_text(_json.dumps({
        "source": "test",
        "scenarios": [{"id": "s1", "features": {}, "truth": [], "stats": {"a": 1}}]}))
    mod = load_app("api", "api_main_scen")
    mod.SCENARIOS_DIR = tmp_path
    mod._spread_files.cache_clear()
    c = TestClient(mod.app)
    listing = c.get("/api/v1/scenarios/spread").json()
    assert listing == [{"id": "s1", "source": "test", "stats": {"a": 1}}]
    assert c.get("/api/v1/scenarios/spread/s1").json()["source"] == "test"
    assert c.get("/api/v1/scenarios/spread/nope").status_code == 404


def test_gateway_burn_sample_forwards_npz(tmp_path):
    import json as _json

    (tmp_path / "burn").mkdir()
    (tmp_path / "burn" / "x.npz").write_bytes(b"NPZDATA")
    (tmp_path / "burn" / "index.json").write_text(_json.dumps(
        {"samples": [{"id": "x", "file": "x.npz"}]}))
    mod = load_app("api", "api_main_burn")
    mod.SCENARIOS_DIR = tmp_path
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = request.read()
        seen["previews"] = request.url.params.get("previews")
        return httpx.Response(200, json={"burned_area_ha": 1.0})

    mod.TRANSPORT = httpx.MockTransport(handler)
    c = TestClient(mod.app)
    r = c.post("/api/v1/scenarios/burn/x/segment")
    assert r.status_code == 200 and r.json()["sample"]["id"] == "x"
    assert b"NPZDATA" in seen["body"] and seen["previews"] == "true"
    assert c.post("/api/v1/scenarios/burn/..%2Fetc/segment").status_code in (400, 404)
