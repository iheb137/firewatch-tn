// Résultats mesurés sur Kaggle (GPU T4), jeux de test, octobre 2026.

export const SPREAD_RESULTS = {
  dataset: "Next Day Wildfire Spread — test : 1 689 tuiles, 6,7 M pixels valides, 1,25 % en feu",
  rows: [
    { name: "U-Net, tuiles complètes 64×64", aucpr: 0.270, f1: 0.354, precision: 0.275, recall: 0.498, production: true },
    { name: "U-Net avec portes d'attention", aucpr: 0.264, f1: 0.345, precision: 0.266, recall: 0.492 },
    { name: "U-Net large (64-128-256 filtres)", aucpr: 0.256, f1: 0.348, precision: 0.271, recall: 0.486 },
    { name: "U-Net, recadrages 32×32", aucpr: 0.256, f1: 0.347, precision: 0.269, recall: 0.489 },
    { name: "Persistance (le feu de demain = celui d'aujourd'hui)", aucpr: 0.116, f1: 0.309, precision: 0.357, recall: 0.273, baseline: true },
  ],
  notes: [
    "Seuil de décision choisi sur la validation (max F1), jamais sur le test.",
    "Un seul entraînement par variante : les écarts entre U-Net (≤ 0,014 d'AUC-PR) restent modestes.",
    "Un modèle 4× plus gros n'apporte rien : la limite vient de l'information disponible dans les données.",
  ],
};

export const BURN_RESULTS = {
  dataset: "CaBuAr — Sentinel-2, Californie, fold de test 0 (validation : fold 1)",
  rows: [
    { name: "U-Net ResNet-34 pré-entraîné", iou: 0.631, f1: 0.774, precision: 0.810, recall: 0.740, areaErr: 0.086, production: true },
    { name: "Seuillage dNBR (seuil 0,24 réglé sur l'entraînement)", iou: 0.527, f1: 0.691, precision: 0.739, recall: 0.649, areaErr: 0.122, baseline: true },
  ],
  notes: [
    "Surface brûlée du fold de test : 113 398 ha réels, 103 597 ha prédits.",
    "IoU de validation (fold 1) : 0,40 contre 0,63 en test : les performances varient fortement selon la région.",
    "Certains patchs n'ont pas d'image avant incendie exploitable (remplie de zéros) : à filtrer.",
  ],
};
