"""Augmentations géométriques physiquement cohérentes.

Tourner ou retourner une tuile change l'orientation du terrain : la direction du
vent ``th`` doit être corrigée en conséquence, sinon le modèle apprend sur des
situations physiquement impossibles (vent « venant du nord » qui pousse le feu
vers l'est).

Conventions :
* les tuiles sont orientées nord en haut (ligne 0 = nord, colonne 0 = ouest) ;
* ``th`` est la direction d'où vient le vent, en degrés, sens horaire depuis le nord.

Corrections (``tf.image.rot90`` tourne dans le sens antihoraire) :
* rotation de k × 90° antihoraire : th' = th − 90·k
* miroir gauche/droite (est ↔ ouest) : th' = −th
* miroir haut/bas (nord ↔ sud) : th' = 180 − th
"""

from __future__ import annotations

import tensorflow as tf


def _wrap_degrees(th: tf.Tensor) -> tf.Tensor:
    return tf.math.floormod(th, 360.0)


def rotate_wind(th: tf.Tensor, k: tf.Tensor) -> tf.Tensor:
    return _wrap_degrees(th - 90.0 * tf.cast(k, th.dtype))


def flip_lr_wind(th: tf.Tensor) -> tf.Tensor:
    return _wrap_degrees(-th)


def flip_ud_wind(th: tf.Tensor) -> tf.Tensor:
    return _wrap_degrees(180.0 - th)


def random_geometric(
    features: dict[str, tf.Tensor], seed: tf.Tensor | None = None
) -> dict[str, tf.Tensor]:
    """Applique rotation 90°·k + miroirs aléatoires à toutes les couches (H, W).

    ``features`` contient des tenseurs bruts (non normalisés) de forme (H, W),
    dont ``th`` en degrés. Retourne un nouveau dict.
    """
    keys = sorted(features)
    stack = tf.stack([features[k] for k in keys], axis=-1)  # (H, W, C)

    if seed is None:
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        flip_lr = tf.random.uniform([]) < 0.5
        flip_ud = tf.random.uniform([]) < 0.5
    else:
        seeds = tf.random.experimental.stateless_split(seed, 3)
        k = tf.random.stateless_uniform([], seeds[0], 0, 4, dtype=tf.int32)
        flip_lr = tf.random.stateless_uniform([], seeds[1]) < 0.5
        flip_ud = tf.random.stateless_uniform([], seeds[2]) < 0.5

    # 1) transformations spatiales, identiques pour toutes les couches
    stack = tf.image.rot90(stack, k)
    stack = tf.cond(flip_lr, lambda: tf.image.flip_left_right(stack), lambda: stack)
    stack = tf.cond(flip_ud, lambda: tf.image.flip_up_down(stack), lambda: stack)
    out = {k_: stack[..., i] for i, k_ in enumerate(keys)}

    # 2) correction des valeurs de direction du vent, dans le même ordre
    th = rotate_wind(out["th"], k)
    th = tf.where(flip_lr, flip_lr_wind(th), th)
    th = tf.where(flip_ud, flip_ud_wind(th), th)
    out["th"] = th
    return out


def random_crop(
    features: dict[str, tf.Tensor], size: int, seed: tf.Tensor | None = None
) -> dict[str, tf.Tensor]:
    """Recadrage aléatoire identique pour toutes les couches."""
    keys = sorted(features)
    stack = tf.stack([features[k] for k in keys], axis=-1)
    shape = [size, size, len(keys)]
    if seed is None:
        stack = tf.image.random_crop(stack, shape)
    else:
        stack = tf.image.stateless_random_crop(stack, shape, seed)
    return {k: stack[..., i] for i, k in enumerate(keys)}
