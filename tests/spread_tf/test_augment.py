"""La correction du vent doit rester physiquement cohérente avec la transformation de l'image."""

import numpy as np
import pytest
import tensorflow as tf

from ml.spread_tf import augment


def wind_vector_image(th_deg):
    """Vecteur « d'où vient le vent » en coordonnées image (colonne, ligne)."""
    r = np.deg2rad(th_deg)
    return np.array([np.sin(r), -np.cos(r)])  # est = +colonne, nord = -ligne


def transform_vector(v, k, lr, ud):
    """Applique au vecteur la même transformation que rot90^k puis miroirs."""
    col, row = v
    for _ in range(k):  # rot90 antihoraire : (col, row) -> (row, -col)
        col, row = row, -col
    if lr:
        col = -col
    if ud:
        row = -row
    return np.array([col, row])


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("lr", [False, True])
@pytest.mark.parametrize("ud", [False, True])
@pytest.mark.parametrize("th", [0.0, 37.0, 90.0, 200.0, 315.0])
def test_wind_correction_matches_geometry(k, lr, ud, th):
    new = augment.rotate_wind(tf.constant(th), k)
    if lr:
        new = augment.flip_lr_wind(new)
    if ud:
        new = augment.flip_ud_wind(new)
    expected = transform_vector(wind_vector_image(th), k, lr, ud)
    np.testing.assert_allclose(wind_vector_image(float(new)), expected, atol=1e-5)


def test_image_rotation_moves_pixels_like_vectors():
    """Un pixel au nord du centre doit finir à l'ouest après rot90 (k=1)."""
    img = np.zeros((5, 5, 1), np.float32)
    img[0, 2] = 1  # nord
    rotated = tf.image.rot90(img, 1).numpy()[..., 0]
    assert rotated[2, 0] == 1  # ouest


def test_random_geometric_keeps_layers_aligned():
    rng = np.random.default_rng(0)
    base = rng.random((16, 16)).astype(np.float32)
    feats = {"a": tf.constant(base), "b": tf.constant(base * 2), "th": tf.fill((16, 16), 10.0)}
    for i in range(10):
        out = augment.random_geometric(feats, seed=tf.constant([i, 0]))
        np.testing.assert_allclose(out["b"].numpy(), 2 * out["a"].numpy(), rtol=1e-6)
        assert np.all((out["th"].numpy() >= 0) & (out["th"].numpy() < 360))
