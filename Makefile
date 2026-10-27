PY ?= python
COMPOSE = docker compose -f deploy/docker-compose.yml

.PHONY: help install-tf install-torch install-dev data-ndws data-cabuar synthetic scenarios scenarios-synthetic eda \
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

scenarios:  ## Scénarios de démo pour l'interface (depuis les vraies données si présentes)
	$(PY) -m ml.spread_tf.export_scenarios --data-dir data/raw/ndws --out data/scenarios/spread/ndws_test.json
	$(PY) -m ml.burn_torch.export_samples --h5 data/raw/cabuar/512x512.hdf5 --out data/scenarios/burn

case-study:  ## Cas d'étude Melloula 2023 : vraies images Sentinel-2 (pip install -r requirements/geo.txt)
	$(PY) -m ml.burn_torch.fetch_s2

scenarios-synthetic: synthetic  ## Scénarios de démo synthétiques (sans télécharger les datasets)
	$(PY) -m ml.spread_tf.export_scenarios --data-dir data/synthetic/ndws --source "Synthétique (démo)" --out data/scenarios/spread/synthetic.json
	$(PY) -m ml.burn_torch.synthetic --out data/synthetic/cabuar_512.hdf5 --patches 15 --size 512
	$(PY) -m ml.burn_torch.export_samples --h5 data/synthetic/cabuar_512.hdf5 --min-burned 0.01 --source "Synthétique (démo)" --out data/scenarios/burn

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
