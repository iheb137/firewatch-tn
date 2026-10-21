"""Indices spectraux de brûlure (fonctionnent sur tenseurs torch ou tableaux numpy)."""

from __future__ import annotations

from ml.burn_torch.constants import BAND

EPS = 1e-6


def nbr(img, nir: str = "B08", swir: str = "B12"):
    """Normalized Burn Ratio : (NIR − SWIR) / (NIR + SWIR). ``img`` : (..., 12, H, W)."""
    a = img[..., BAND[nir], :, :]
    b = img[..., BAND[swir], :, :]
    return (a - b) / (a + b + EPS)


def dnbr(pre, post):
    """Différence de NBR : forte valeur positive = végétation brûlée."""
    return nbr(pre) - nbr(post)
