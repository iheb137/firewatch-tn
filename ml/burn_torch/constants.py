"""Constantes du dataset CaBuAr (California Burned Areas).

Format vérifié sur le script de chargement officiel du dépôt Hugging Face
``DarthReca/california_burned_areas`` :

* fichier ``raw/patched/512x512.hdf5`` (≈ 5,9 Go, compression BZip2 → ``hdf5plugin`` requis) ;
* une entrée par patch ``uidN_x`` avec ``post_fire`` (512, 512, 12) uint16,
  ``mask`` (512, 512, 1) uint16 et, pour une partie des patchs, ``pre_fire`` ;
* l'attribut ``fold`` ∈ {0, 1, 2, 3, 4} donne le fold de validation croisée.
"""

from __future__ import annotations

PATCH_SIZE = 512

# Ordre standard des 12 bandes Sentinel-2 L2A (B10 « cirrus » n'existe pas en L2A).
# À confirmer au premier chargement avec `python -m ml.burn_torch.inspect`.
S2_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",
)
BAND = {name: i for i, name in enumerate(S2_BANDS)}

# Les scènes CaBuAr font 5490 × 5490 pixels pour une tuile Sentinel-2 de 109,8 km
# → 20 m par pixel → 1 pixel = 0,04 ha.
PIXEL_SIZE_M = 20.0
PIXEL_AREA_HA = PIXEL_SIZE_M**2 / 10_000

REFLECTANCE_SCALE = 10_000.0  # valeurs L2A entières → réflectance [0, 1]
FOLDS = (0, 1, 2, 3, 4)
