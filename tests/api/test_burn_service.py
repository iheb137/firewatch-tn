"""burn-service (PyTorch) — exécuté dans un processus sans TensorFlow."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.api._helpers import load_app


@pytest.fixture(scope="module")
def burn_client(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from ml.burn_torch.model import BurnModelConfig, build_model

    d = tmp_path_factory.mktemp("burn_model")
    cfg = {"model": {"arch": "unet", "encoder": "resnet18", "encoder_weights": None},
           "data": {"mode": "pre_post"}}
    model = build_model(BurnModelConfig(**cfg["model"], in_channels=27))
    torch.save({"model": model.state_dict(), "config": cfg}, d / "best.pt")
    mod = load_app("burn-service", "burn_main")
    mod.MODEL_PATH = d / "best.pt"
    with TestClient(mod.app) as c:
        yield c


def npz(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_burn_segment(burn_client):
    img = np.random.default_rng(0).integers(0, 5000, (64, 64, 12), dtype=np.uint16)
    r = burn_client.post("/segment", files={"file": ("x.npz", npz(pre=img, post=img))})
    assert r.status_code == 200, r.text
    assert r.json()["shape"] == [64, 64] and r.json()["burned_area_ha"] >= 0
    r = burn_client.post("/segment", files={"file": ("x.npz", npz(post=img))})
    assert r.status_code == 400  # image avant manquante
