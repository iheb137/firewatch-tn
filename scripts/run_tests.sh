#!/usr/bin/env bash
# TensorFlow et PyTorch ne doivent pas être chargés dans le même processus
# (conflit de bibliothèques natives -> segfault). Chaque groupe tourne à part,
# comme en production où chaque modèle a son propre conteneur.
set -euo pipefail
export TF_CPP_MIN_LOG_LEVEL=3
ARGS=("$@")
groups=(
  "tests/spread_tf tests/api/test_spread_service.py"
  "tests/burn_torch tests/api/test_burn_service.py"
  "tests/api/test_gateway.py"
)
status=0
for g in "${groups[@]}"; do
  echo "==> pytest $g"
  # shellcheck disable=SC2086
  python -m pytest $g -q "${ARGS[@]}" || status=1
done
exit $status
