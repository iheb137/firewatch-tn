"""Pertes masquées : les pixels étiquetés -1 n'entrent ni au numérateur ni au dénominateur."""

from __future__ import annotations

import keras
import tensorflow as tf


def _mask_and_labels(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    labels = tf.clip_by_value(y_true, 0.0, 1.0)
    return mask, labels


@keras.saving.register_keras_serializable(package="firewatch")
class MaskedBceDiceLoss(keras.losses.Loss):
    """BCE pondérée (sur logits) + Dice doux, sur les pixels valides uniquement.

    * ``pos_weight`` > 1 compense la rareté des pixels en feu ;
    * le terme Dice optimise directement le recouvrement du front de feu.
    """

    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 1.0,
                 smooth: float = 1.0, name: str = "masked_bce_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def call(self, y_true, y_pred):
        logits = tf.cast(y_pred, tf.float32)
        mask, labels = _mask_and_labels(y_true)
        n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)

        bce = tf.nn.weighted_cross_entropy_with_logits(
            labels=labels, logits=logits, pos_weight=self.pos_weight
        )
        bce = tf.reduce_sum(bce * mask) / n_valid

        probs = tf.sigmoid(logits) * mask
        inter = tf.reduce_sum(probs * labels * mask)
        denom = tf.reduce_sum(probs) + tf.reduce_sum(labels * mask)
        dice = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        # Keras attend une perte par échantillon : on la diffuse sur le batch.
        total = bce + self.dice_weight * dice
        return tf.fill([tf.shape(logits)[0]], total)

    def get_config(self):
        return {**super().get_config(), "pos_weight": self.pos_weight,
                "dice_weight": self.dice_weight, "smooth": self.smooth}
