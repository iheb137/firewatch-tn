#!/usr/bin/env bash
# Télécharge le fichier de patchs 512x512 de CaBuAr depuis Hugging Face (~5,9 Go).
# Prérequis : pip install huggingface_hub
set -euo pipefail
DEST="${1:-data/raw/cabuar}"
mkdir -p "$DEST"
hf download DarthReca/california_burned_areas raw/patched/512x512.hdf5 \
  --repo-type dataset --local-dir "$DEST/hf"
mv -n "$DEST/hf/raw/patched/512x512.hdf5" "$DEST/512x512.hdf5"
python -m ml.burn_torch.inspect_h5 "$DEST/512x512.hdf5"
