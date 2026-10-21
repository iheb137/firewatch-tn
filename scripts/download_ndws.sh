#!/usr/bin/env bash
# Télécharge Next Day Wildfire Spread depuis Kaggle (~2 Go décompressé).
# Prérequis : pip install kaggle, puis ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token)
set -euo pipefail
DEST="${1:-data/raw/ndws}"
mkdir -p "$DEST"
kaggle datasets download -d fantineh/next-day-wildfire-spread -p "$DEST" --unzip
# L'archive peut contenir un sous-dossier : on ramène les .tfrecord à la racine.
find "$DEST" -mindepth 2 -name '*.tfrecord' -exec mv -n {} "$DEST"/ \;
echo "Fichiers :"; ls -1 "$DEST"/*.tfrecord | sed 's#.*/#  #'
