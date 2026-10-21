"""spread-service (TensorFlow) — exécuté dans un processus sans PyTorch."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.api._helpers import load_app


def raw_tile(size=32):
    from ml.spread_tf.synthetic import make_tile

    tile = make_tile(np.random.default_rng(0), size)
    tile.pop("FireMask")
    return {k: v.tolist() for k, v in tile.items()}



@pytest.fixture(scope="module")
def spread_client(tmp_path_factory):
    pytest.importorskip("tensorflow")
    from ml.spread_tf.model import ModelConfig, build_model

    d = tmp_path_factory.mktemp("spread_model")
    build_model(ModelConfig(filters=(8, 16, 32))).save(d / "model.keras")
    (d / "meta.json").write_text(json.dumps({"run": "test", "threshold": 0.4}))
    mod = load_app("spread-service", "spread_main")
    mod.MODEL_DIR = d
    with TestClient(mod.app) as c:
        yield c


def test_spread_health_and_predict(spread_client):
    assert spread_client.get("/health").json()["model_loaded"] is True
    r = spread_client.post("/predict", json={"features": raw_tile(), "mc_samples": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert np.asarray(body["probability"]).shape == (32, 32)
    assert body["threshold"] == 0.4 and body["uncertainty"] is not None


def test_spread_rejects_bad_input(spread_client):
    feats = raw_tile()
    feats.pop("th")
    assert spread_client.post("/predict", json={"features": feats}).status_code == 422
