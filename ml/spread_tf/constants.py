"""Constantes du dataset Next Day Wildfire Spread (NDWS).

Les noms de variables, l'ordre et les statistiques proviennent du code de
référence publié par Google Research avec le dataset
(google-research/simulation_research/next_day_wildfire_spread, dataset_demo.ipynb).
Les statistiques ont été calculées par les auteurs sur le jeu d'entraînement :
les réutiliser respecte la règle « normalisation calculée sur le train uniquement ».
"""

from __future__ import annotations

TILE_SIZE = 64  # pixels, 1 km / pixel

# Ordre identique au notebook de référence (format Kaggle).
INPUT_FEATURES: tuple[str, ...] = (
    "elevation",  # m
    "th",  # direction du vent, degrés depuis le nord, sens horaire (d'où vient le vent)
    "vs",  # vitesse du vent, m/s
    "tmmn",  # température min, K
    "tmmx",  # température max, K
    "sph",  # humidité spécifique, kg/kg
    "pr",  # précipitations, mm
    "pdsi",  # Palmer Drought Severity Index
    "NDVI",  # indice de végétation (x 10 000)
    "population",  # habitants / km²
    "erc",  # Energy Release Component (NFDRS)
    "PrevFireMask",  # masque du feu au jour J : 1 feu, 0 pas de feu, -1 inconnu
)
TARGET_FEATURE = "FireMask"  # masque du feu au jour J+1, même encodage

# (min_clip, max_clip, mean, std) — valeurs du notebook de référence.
DATA_STATS: dict[str, tuple[float, float, float, float]] = {
    "elevation": (0.0, 3141.0, 657.3003, 649.0147),
    "pdsi": (-6.1298, 7.8760, -0.0053, 2.6823),
    "NDVI": (-9821.0, 9996.0, 5157.625, 2466.6677),
    "pr": (0.0, 44.5304, 1.7398051, 4.4828),
    "sph": (0.0, 1.0, 0.0071658953, 0.0042835088),
    "th": (0.0, 360.0, 190.3298, 72.5985),
    "tmmn": (253.15, 298.9489, 281.08768, 8.9824),
    "tmmx": (253.15, 315.0923, 295.17383, 9.8155),
    "vs": (0.0, 10.0243, 3.8501, 1.4110),
    "erc": (0.0, 106.2489, 37.3263, 20.8460),
    "population": (0.0, 2534.0630, 25.5314, 154.7233),
}

# La direction du vent est une variable circulaire : une normalisation
# moyenne/écart-type la casse (359° et 1° seraient « éloignés »). On la remplace
# par ses composantes sin/cos. Les canaux d'entrée du modèle sont donc :
MODEL_CHANNELS: tuple[str, ...] = tuple(
    c for f in INPUT_FEATURES for c in (("th_sin", "th_cos") if f == "th" else (f,))
)
NUM_MODEL_CHANNELS = len(MODEL_CHANNELS)  # 13

UNKNOWN_LABEL = -1.0

# Noms des fichiers dans l'archive Kaggle « fantineh/next-day-wildfire-spread ».
SPLIT_PATTERNS = {
    "train": "next_day_wildfire_spread_train_*.tfrecord",
    "eval": "next_day_wildfire_spread_eval_*.tfrecord",
    "test": "next_day_wildfire_spread_test_*.tfrecord",
}
