"""Modèles du Module B (segmentation des surfaces brûlées)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BurnModelConfig:
    arch: str = "unet"  # "unet" | "segformer"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"  # None = initialisation aléatoire
    in_channels: int = 27


def build_model(cfg: BurnModelConfig) -> nn.Module:
    """U-Net / SegFormer via segmentation_models_pytorch.

    Avec des poids ImageNet et ``in_channels`` ≠ 3, smp adapte automatiquement
    la première convolution (réutilisation des poids RVB sur les canaux supplémentaires).
    """
    import segmentation_models_pytorch as smp

    if cfg.arch == "unet":
        return smp.Unet(encoder_name=cfg.encoder, encoder_weights=cfg.encoder_weights,
                        in_channels=cfg.in_channels, classes=1)
    if cfg.arch == "segformer":
        return smp.Segformer(encoder_name="mit_b0", encoder_weights=cfg.encoder_weights,
                             in_channels=cfg.in_channels, classes=1)
    raise ValueError(f"Architecture inconnue : {cfg.arch}")


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="binary")
        self.wd, self.wf = dice_weight, focal_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.wd * self.dice(logits, target) + self.wf * self.focal(logits, target)
