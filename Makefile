PY ?= python
COMPOSE = docker compose -f deploy/docker-compose.yml

.PHONY: help install-tf install-torch install-dev data-ndws data-cabuar synthetic eda \
        train-spread baseline-spread train-burn baseline-burn test test-fast lint up down

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[33m%-16s\033[0m %s\n",$$1,$$2}'

install-tf:  ## Dépendances du Module A (TensorFlow)
	$(PY) -m pip install -r requirements/tf.txt -r requirements/dev.txt

install-torch:  ## Dépendances du Module B (PyTorch)
	$(PY) -m pip install -r requirements/torch.txt -r requirements/dev.txt

data-ndws:  ## Télécharge Next Day Wildfire Spread (Kaggle)
	bash scripts/download_ndws.sh data/raw/ndws

data-cabuar:  ## Télécharge CaBuAr (Hugging Face)
	bash scripts/download_cabuar.sh data/raw/cabuar

synthetic:  ## Génère des données synthétiques au format réel (démo sans téléchargement)
	$(PY) -m ml.spread_tf.synthetic --out data/synthetic/ndws
	$(PY) -m ml.burn_torch.synthetic --out data/synthetic/cabuar.hdf5

eda:  ## Analyse exploratoire NDWS -> reports/eda_ndws
	$(PY) -m ml.spread_tf.eda --data-dir data/raw/ndws --out reports/eda_ndws

baseline-spread:  ## Baseline de persistance (Module A)
	$(PY) -m ml.spread_tf.evaluate --baseline persistence

train-spread:  ## Entraîne le Module A
	$(PY) -m ml.spread_tf.train --config configs/spread_tf.yaml

baseline-burn:  ## Baseline seuillage dNBR (Module B)
	$(PY) -m ml.burn_torch.baseline --config configs/burn_torch.yaml

train-burn:  ## Entraîne le Module B
	$(PY) -m ml.burn_torch.train --config configs/burn_torch.yaml

test:  ## Tous les tests (TF et PyTorch dans des processus séparés)
	bash scripts/run_tests.sh

test-fast:  ## Tests rapides uniquement
	bash scripts/run_tests.sh -m "not slow"

lint:  ## Ruff
	ruff check .

up:  ## Lance la plateforme (API, services, MLflow, PostGIS, MinIO)
	$(COMPOSE) up --build -d

down:  ## Arrête la plateforme
	$(COMPOSE) down
