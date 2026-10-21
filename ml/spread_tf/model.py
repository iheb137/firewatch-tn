"""Modèles du Module A : baseline logistique et U-Net résiduel (option attention).

Toutes les sorties sont des **logits** (pas de sigmoïde) pour la stabilité
numérique ; la sigmoïde est appliquée dans les métriques et à l'export.
Les modèles sont entièrement convolutionnels : entraînés sur des recadrages
32×32, ils s'appliquent directement aux tuiles 64×64 (taille multiple de 2^profondeur).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import keras
from keras import layers

from ml.spread_tf.constants import NUM_MODEL_CHANNELS


@dataclass(frozen=True)
class ModelConfig:
    name: str = "res_unet"  # "res_unet" | "attention_unet" | "logreg"
    filters: tuple[int, ...] = field(default=(32, 64, 128))
    dropout: float = 0.1
    l2: float = 1e-5


@keras.saving.register_keras_serializable(package="firewatch")
class MCSpatialDropout2D(layers.SpatialDropout2D):
    """Dropout spatial activable à l'inférence pour le Monte Carlo Dropout.

    ``layer.mc_active = True`` force le dropout même en mode inférence, sans
    toucher à la BatchNorm (contrairement à ``model(x, training=True)``).
    """

    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.mc_active = False

    def call(self, inputs, training=None):
        return super().call(inputs, training=bool(training) or self.mc_active)


def _conv_bn_relu(x, filters, l2, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2), name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def residual_block(x, filters: int, dropout: float, l2: float, name: str):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", name=f"{name}_proj")(x)
    y = _conv_bn_relu(x, filters, l2, f"{name}_a")
    if dropout > 0:
        y = MCSpatialDropout2D(dropout, name=f"{name}_drop")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2), name=f"{name}_b_conv")(y)
    y = layers.BatchNormalization(name=f"{name}_b_bn")(y)
    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.ReLU(name=f"{name}_out")(y)


def attention_gate(skip, gating, filters: int, name: str):
    """Porte d'attention additive (Oktay et al., 2018) : pondère la connexion de saut."""
    theta = layers.Conv2D(filters, 1, name=f"{name}_theta")(skip)
    phi = layers.Conv2D(filters, 1, name=f"{name}_phi")(gating)
    act = layers.ReLU(name=f"{name}_relu")(layers.Add(name=f"{name}_add")([theta, phi]))
    psi = layers.Conv2D(1, 1, activation="sigmoid", name=f"{name}_psi")(act)
    return layers.Multiply(name=f"{name}_gate")([skip, psi])


def build_res_unet(cfg: ModelConfig, in_channels: int = NUM_MODEL_CHANNELS,
                   attention: bool = False) -> keras.Model:
    inputs = keras.Input(shape=(None, None, in_channels), name="features")
    x = inputs
    skips = []
    for i, f in enumerate(cfg.filters[:-1]):
        x = residual_block(x, f, cfg.dropout, cfg.l2, name=f"enc{i}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)

    x = residual_block(x, cfg.filters[-1], cfg.dropout, cfg.l2, name="bottleneck")

    for i, f in reversed(list(enumerate(cfg.filters[:-1]))):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up{i}")(x)
        skip = skips[i]
        if attention:
            skip = attention_gate(skip, x, max(f // 2, 8), name=f"att{i}")
        x = layers.Concatenate(name=f"cat{i}")([x, skip])
        x = residual_block(x, f, cfg.dropout, cfg.l2, name=f"dec{i}")

    logits = layers.Conv2D(1, 1, dtype="float32", name="logits")(x)
    return keras.Model(inputs, logits, name="attention_unet" if attention else "res_unet")


def build_logreg(in_channels: int = NUM_MODEL_CHANNELS) -> keras.Model:
    """Régression logistique pixel à pixel = une convolution 1×1."""
    inputs = keras.Input(shape=(None, None, in_channels), name="features")
    logits = layers.Conv2D(1, 1, dtype="float32", name="logits")(inputs)
    return keras.Model(inputs, logits, name="logreg")


def build_model(cfg: ModelConfig) -> keras.Model:
    if cfg.name == "logreg":
        return build_logreg()
    if cfg.name in ("res_unet", "attention_unet"):
        return build_res_unet(cfg, attention=cfg.name == "attention_unet")
    raise ValueError(f"Modèle inconnu : {cfg.name}")


def set_mc_dropout(model: keras.Model, active: bool) -> int:
    n = 0
    for layer in model.layers:
        if isinstance(layer, MCSpatialDropout2D):
            layer.mc_active = active
            n += 1
    return n
