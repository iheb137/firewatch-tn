import numpy as np
import pytest
import tensorflow as tf

from ml.spread_tf.constants import MODEL_CHANNELS, NUM_MODEL_CHANNELS
from ml.spread_tf.dataset import DataConfig, count_records, make_dataset
from ml.spread_tf.losses import MaskedBceDiceLoss
from ml.spread_tf.metrics import average_precision, best_threshold, confusion_at, flatten_valid
from ml.spread_tf.model import ModelConfig, build_model, set_mc_dropout


def test_dataset_shapes_and_labels(ndws_dir):
    cfg = DataConfig(data_dir=ndws_dir, batch_size=8, train_crop=32)
    assert count_records(ndws_dir, "train") == 96
    x, y = next(iter(make_dataset(cfg, "train")))
    assert x.shape == (8, 32, 32, NUM_MODEL_CHANNELS)
    assert y.shape == (8, 32, 32, 1)
    assert set(np.unique(y.numpy())) <= {-1.0, 0.0, 1.0}
    x, y = next(iter(make_dataset(cfg, "eval")))
    assert x.shape[1:] == (64, 64, NUM_MODEL_CHANNELS)


def test_wind_encoded_as_unit_vector(ndws_dir):
    x, _ = next(iter(make_dataset(DataConfig(data_dir=ndws_dir, batch_size=4), "eval")))
    s = x[..., MODEL_CHANNELS.index("th_sin")]
    c = x[..., MODEL_CHANNELS.index("th_cos")]
    np.testing.assert_allclose((s**2 + c**2).numpy(), 1.0, atol=1e-5)


def test_loss_ignores_unknown_pixels():
    loss = MaskedBceDiceLoss()
    y = tf.constant([[[[1.0], [0.0], [-1.0], [-1.0]]]])
    logits_a = tf.constant([[[[5.0], [-5.0], [5.0], [-5.0]]]])
    logits_b = tf.constant([[[[5.0], [-5.0], [-9.0], [9.0]]]])  # ne change que les -1
    np.testing.assert_allclose(float(loss(y, logits_a)), float(loss(y, logits_b)), rtol=1e-6)


def test_offline_metrics():
    y = np.array([1, 0, 1, 0, -1, 1])
    p = np.array([0.9, 0.2, 0.8, 0.7, 0.99, 0.1])
    yv, pv = flatten_valid(y, p)
    assert yv.size == 5
    m = confusion_at(yv, pv, 0.5)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert average_precision(np.array([1, 0]), np.array([0.9, 0.1])) == pytest.approx(1.0)
    assert 0.05 <= best_threshold(yv, pv) <= 0.95


@pytest.mark.parametrize("name", ["res_unet", "attention_unet", "logreg"])
def test_models_are_fully_convolutional(name):
    model = build_model(ModelConfig(name=name, filters=(8, 16, 32)))
    for size in (32, 64):
        out = model(tf.zeros((2, size, size, NUM_MODEL_CHANNELS)))
        assert out.shape == (2, size, size, 1)


def test_mc_dropout_produces_variance():
    model = build_model(ModelConfig(filters=(8, 16, 32), dropout=0.3))
    x = tf.random.normal((1, 32, 32, NUM_MODEL_CHANNELS))
    a, b = model(x, training=False), model(x, training=False)
    np.testing.assert_allclose(a.numpy(), b.numpy())  # déterministe par défaut
    assert set_mc_dropout(model, True) > 0
    assert not np.allclose(model(x).numpy(), model(x).numpy())
    set_mc_dropout(model, False)


def test_spread_code_never_loads_torch():
    import subprocess
    import sys

    code = ("import sys, ml.spread_tf.train, ml.spread_tf.evaluate; "
            "from ml.common.config import seed_everything; seed_everything(0); "
            "sys.exit(int('torch' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
