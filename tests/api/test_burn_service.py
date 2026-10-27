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


def test_burn_segment_previews_and_truth(burn_client):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 5000, (64, 64, 12), dtype=np.uint16)
    mask = np.zeros((64, 64, 1), np.uint8)
    mask[:32] = 1
    r = burn_client.post("/segment", params={"previews": "true"},
                         files={"file": ("x.npz", npz(pre=img, post=img, mask=mask))})
    assert r.status_code == 200, r.text
    body = r.json()
    for k in ("pre_rgb_png_base64", "post_rgb_png_base64", "overlay_png_base64"):
        assert len(body[k]) > 100
    assert body["truth"]["burned_area_ha"] == 32 * 64 * 0.04
    assert 0 <= body["truth"]["iou"] <= 1


def test_overlay_fill_is_light_and_edges_opaque():
    import base64
    import io

    from PIL import Image

    mod = load_app("burn-service", "burn_main")
    mask_edges, overlay_png_b64 = mod.mask_edges, mod.overlay_png_b64

    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    edges = mask_edges(mask, width=2)
    assert edges[5, 10] and edges[6, 10] and not edges[7, 10] and not edges[0, 0]
    prob = mask.astype(np.float32)
    rgba = np.array(Image.open(io.BytesIO(base64.b64decode(overlay_png_b64(prob, 0.5)))))
    assert rgba[5, 10, 3] == 255 and 0 < rgba[10, 10, 3] < 128 and rgba[0, 0, 3] == 0
