# 🔥 FireWatch TN

[![CI](https://github.com/iheb137/firewatch-tn/actions/workflows/ci.yml/badge.svg)](https://github.com/iheb137/firewatch-tn/actions/workflows/ci.yml)

**Prédiction de la propagation et cartographie des incendies de forêt en Tunisie par réseaux de neurones.**

Mini-projet deep learning, ING-5-GLSI-J-D, TEK-UP University.
Réalisé par **Iheb Eddine Saafi** et **Arbi Hazbri**.

| Module | Tâche | Framework | Données d'entraînement |
|---|---|---|---|
| **A — Propagation** | Masque du feu à J+1 (segmentation 64×64, 1 km/px) | TensorFlow / Keras | [Next Day Wildfire Spread](https://www.kaggle.com/fantineh/next-day-wildfire-spread) (Google Research) |
| **B — Surfaces brûlées** | Masque brûlé + surface en ha (Sentinel-2, 20 m/px) | PyTorch | [CaBuAr](https://huggingface.co/datasets/DarthReca/california_burned_areas) (Politecnico di Torino) |

Le document de cadrage complet est dans [`docs/cadrage_firewatch_tn.pdf`](docs/cadrage_firewatch_tn.pdf).

---

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
make install-tf          # Module A
make install-torch       # Module B (CPU : pip install torch --index-url https://download.pytorch.org/whl/cpu)
make test                # ~1 min, données synthétiques au format réel
                         # (TF et PyTorch testés dans des processus séparés)
```

Sans rien télécharger, `make synthetic` génère des données au **schéma exact** des
vrais datasets : utile pour développer l'API et l'interface en parallèle de l'entraînement.

## Module A — Propagation (TensorFlow)

```bash
make data-ndws           # nécessite ~/.kaggle/kaggle.json
make eda                 # statistiques + figures -> reports/eda_ndws
make baseline-spread     # référence minimale : persistance
make train-spread        # U-Net résiduel
python -m ml.spread_tf.evaluate --run runs/spread_tf/<run>
python -m ml.spread_tf.export   --run runs/spread_tf/<run> --out models/spread
```

Variantes : `--set model.name=attention_unet`, `--set model.name=logreg` (baseline 1×1).

**Choix techniques importants**

- **Pixels `-1` masqués** (nuages, données manquantes) dans la perte *et* les métriques.
- **Direction du vent encodée en sin/cos** : 359° et 1° sont proches ; une normalisation
  moyenne/écart-type les éloignerait. Le modèle reçoit donc 13 canaux (12 variables, vent dédoublé).
- **Augmentation physiquement cohérente** : rotations/miroirs avec correction de la
  direction du vent (`ml/spread_tf/augment.py`, testé dans `tests/spread_tf/test_augment.py`).
- **Seuil de décision choisi sur la validation**, jamais sur le test ; AUC-PR en métrique principale.
- **MC Dropout** pour la carte d'incertitude, sans perturber la BatchNorm.

## Module B — Surfaces brûlées (PyTorch)

```bash
make data-cabuar         # ~5,9 Go, vérifie le contenu du fichier à la fin
make baseline-burn       # seuillage dNBR (seuil réglé sur les folds d'entraînement)
make train-burn          # U-Net ResNet-34 pré-entraîné ImageNet
python -m ml.burn_torch.train --set split.test_fold=2 split.val_fold=3   # autre fold
```

Entrée : 12 bandes avant + 12 bandes après + NBR avant/après + dNBR = **27 canaux**.
Validation croisée sur les 5 folds géographiques de CaBuAr.

> ⚠️ À faire au premier téléchargement : vérifier avec `python -m ml.burn_torch.inspect_h5`
> que l'ordre des bandes correspond à `ml/burn_torch/constants.py` (B08 nettement
> supérieure à B04 sur la végétation).

## Plateforme

```bash
make up                  # API :8000, MLflow :5000, PostGIS :5432, MinIO :9001
open http://localhost:8000/docs
```

| Service | Rôle |
|---|---|
| `services/api` | Passerelle REST unique pour l'interface (`/api/v1/...`) |
| `services/spread-service` | Inférence Module A (charge `models/spread/model.keras`) |
| `services/burn-service` | Inférence Module B (charge `models/burn/best.pt`) |

Pour suivre les entraînements dans MLflow : `export MLFLOW_TRACKING_URI=http://localhost:5000`.
Sans cette variable, les scripts tournent normalement (Colab, Kaggle).

## Entraîner sur GPU gratuit

Kaggle Notebooks (le dataset NDWS y est déjà monté) ou Google Colab :

```python
!git clone https://github.com/iheb137/firewatch-tn && cd firewatch-tn
!pip install -q -r requirements/tf.txt
!python -m ml.spread_tf.train --set data.data_dir=/kaggle/input/next-day-wildfire-spread train.mixed_precision=true
```

## Structure

```
configs/            hyperparamètres YAML (surchargeables : --set clé=valeur)
ml/common/          configuration, graines, suivi MLflow
ml/spread_tf/       Module A : dataset, augment, model, losses, metrics, train, evaluate, export
ml/burn_torch/      Module B : dataset, indices, model, metrics, train, baseline
services/           api, spread-service, burn-service (FastAPI)
deploy/             docker-compose.yml (Kubernetes : sprint 6)
tests/              pytest (synthétique), exécutés en CI
web/                interface React (sprint 4)
docs/               cadrage, rapport
```

## Références

- F. Huot et al., *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire
  Spreading From Remote-Sensing Data*, IEEE TGRS, 2022.
- D. Rege Cambrin, L. Colomba, P. Garza, *CaBuAr: California Burned Areas dataset for
  delineation*, IEEE GRSM, 2023.
