#!/usr/bin/env bash
#
# End-to-end HawkEye model training, for the RTX 3050 laptop (CUDA).
#
# Runs the whole pipeline in order:
#   1. download + organize the action datasets from Kaggle
#   2. train the action classifier (fall / violence / normal)
#   3. download + train the weapon YOLO detector
#   4. a post-training sanity pass with eval/evaluate.py
#
# Each step is a standalone command — copy/paste any single line to re-run just
# that stage (e.g. re-train weapons without re-downloading everything). Prereqs:
#   - CUDA torch installed (see requirements-cuda.txt) + pip install -r requirements-cuda.txt
#   - ~/.kaggle/kaggle.json in place
# Manual-only datasets (UR Fall, Le2i) are NOT fetched here — see docs/training.md.
#
# Usage:  bash scripts/train_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root

ACTION_DATA="data/datasets/actions"
WEAPON_DIR="data/datasets/weapons"
ACTION_OUT="models/action_gru.pt"
WEAPON_OUT="models/weapons.pt"

# Fall back to CPU for unsupported MPS ops instead of crashing (harmless on CUDA).
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "==> [1/4] Downloading action datasets from Kaggle"
python scripts/setup_datasets.py --source kaggle --out "$ACTION_DATA"

echo "==> [2/4] Training the action classifier"
python scripts/train_action.py --data "$ACTION_DATA" --out "$ACTION_OUT" --epochs 30

echo "==> [3/4] Downloading + training the weapon detector"
python scripts/train_weapon.py --auto-download --dataset-dir "$WEAPON_DIR" \
    --out "$WEAPON_OUT" --epochs 80 --imgsz 640

echo "==> [4/4] Post-training sanity check (action classifier)"
python eval/evaluate.py test-action --data "$ACTION_DATA" --model "$ACTION_OUT" || true
python eval/evaluate.py compare     --data "$ACTION_DATA" --model "$ACTION_OUT" || true

echo
echo "Done. Next: re-tune configs/default.yaml thresholds from the metrics above"
echo "(see docs/training.md - 'Re-tuning the config'), then run a live Stage-A check."
