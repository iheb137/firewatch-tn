import json

import numpy as np
import pytest
import torch

from ml.burn_torch.constants import PIXEL_AREA_HA
from ml.burn_torch.dataset import BurnDataConfig, CaBuArDataset, build_input, num_channels
from ml.burn_torch.indices import dnbr
from ml.burn_torch.metrics import SegMeter
from ml.burn_torch.model import BurnModelConfig, DiceFocalLoss, build_model
from ml.burn_torch.synthetic import write
from ml.burn_torch.train import fold_split, train
from ml.common.config import load_config


@pytest.fixture(scope="module")
def h5(tmp_path_factory):
    return write(str(tmp_path_factory.mktemp("cabuar") / "cabuar.hdf5"), n=30, size=64)


def test_fold_split_is_disjoint():
    tr, va, te = fold_split(0, 1)
    assert set(tr) | set(va) | set(te) == {0, 1, 2, 3, 4}
    assert not set(tr) & (set(va) | set(te))
    with pytest.raises(ValueError):
        fold_split(2, 2)


@pytest.mark.parametrize("mode", ["pre_post", "post"])
def test_dataset_reads_cabuar_layout(h5, mode):
    ds = CaBuArDataset(BurnDataConfig(h5_path=h5, mode=mode, crop=32), [0, 1, 2], training=True)
    x, y = ds[0]
    assert x.shape == (num_channels(mode), 32, 32) and y.shape == (1, 32, 32)
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
    if mode == "pre_post":  # les patchs sans image avant sont exclus
        assert all(int(k[3:].split("_")[0]) % 7 != 0 for k in ds.keys)


def test_dnbr_is_high_on_burned_pixels(h5):
    ds = CaBuArDataset(BurnDataConfig(h5_path=h5, crop=None, augment=False), [0], training=False)
    x, y = ds[0]
    d = x[26]
    assert d[y[0] > 0].mean() > d[y[0] == 0].mean() + 0.2
    pre, post = x[:12], x[12:24]
    torch.testing.assert_close(dnbr(pre, post), d)


def test_build_input_scales_reflectance():
    post = np.full((4, 4, 12), 10_000, np.uint16)
    x = build_input(post, None)
    assert x.shape == (13, 4, 4) and float(x[0, 0, 0]) == pytest.approx(1.0)


def test_meter_area_and_iou():
    m = SegMeter()
    target = torch.zeros(1, 1, 10, 10)
    target[..., :5, :] = 1
    probs = target.clone()
    probs[..., 4, :] = 0  # rate une ligne
    m.update(probs, target)
    r = m.compute()
    assert r["iou"] == pytest.approx(40 / 50)
    assert r["area_true_ha"] == pytest.approx(50 * PIXEL_AREA_HA)


def test_model_and_loss_forward():
    model = build_model(BurnModelConfig(encoder="resnet18", encoder_weights=None, in_channels=27))
    logits = model(torch.zeros(2, 27, 64, 64))
    assert logits.shape == (2, 1, 64, 64)
    loss = DiceFocalLoss()(logits, torch.zeros_like(logits))
    assert torch.isfinite(loss)


@pytest.mark.slow
def test_train_smoke(h5, tmp_path):
    cfg = load_config("configs/burn_torch.yaml", [
        f"data.h5_path={h5}", "data.crop=32", "model.encoder=resnet18",
        "model.encoder_weights=null", "train.epochs=2", "train.batch_size=4",
        "train.num_workers=0", f"train.output_dir={tmp_path}",
    ])
    run = train(cfg)
    result = json.loads((run / "eval.json").read_text())
    assert 0 <= result["test"]["iou"] <= 1
    assert (run / "best.pt").exists()


def test_burn_code_never_loads_tensorflow():
    """TF + PyTorch dans un même processus = segfault : le Module B ne doit jamais charger TF."""
    import subprocess
    import sys

    code = ("import sys, ml.burn_torch.train, ml.burn_torch.baseline; "
            "from ml.common.config import seed_everything; seed_everything(0); "
            "sys.exit(int('tensorflow' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
