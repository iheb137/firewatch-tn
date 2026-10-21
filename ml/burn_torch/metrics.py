"""Métriques de segmentation accumulées sur tout le jeu (micro-moyenne) + erreur de surface."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ml.burn_torch.constants import PIXEL_AREA_HA


@dataclass
class SegMeter:
    threshold: float = 0.5
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    area_true_ha: list[float] = field(default_factory=list)
    area_pred_ha: list[float] = field(default_factory=list)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        pred = probs >= self.threshold
        tgt = target > 0.5
        self.tp += float((pred & tgt).sum())
        self.fp += float((pred & ~tgt).sum())
        self.fn += float((~pred & tgt).sum())
        dims = tuple(range(1, pred.ndim))
        self.area_pred_ha += (pred.sum(dim=dims).float() * PIXEL_AREA_HA).tolist()
        self.area_true_ha += (tgt.sum(dim=dims).float() * PIXEL_AREA_HA).tolist()

    def compute(self) -> dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        true_total, pred_total = sum(self.area_true_ha), sum(self.area_pred_ha)
        area_err = abs(pred_total - true_total) / true_total if true_total else float("nan")
        return {"iou": iou, "f1": f1, "precision": precision, "recall": recall,
                "area_true_ha": true_total, "area_pred_ha": pred_total,
                "area_rel_error": area_err}
