#!/usr/bin/env bash
#
# End-to-end HawkEye model training. Target: Apple Silicon (M2, MPS).
#
# Runs the whole pipeline in order:
#   1. download + organize the action datasets from Kaggle
#   2. train the action classifier (fall / violence / normal)
#   3. download + train the weapon YOLO detector
#   4. a post-training sanity pass with eval/evaluate.py
#
# Each step is a standalone command — copy/paste any single line to re-run just
# that stage (e.g. re-train weapons without re-downloading everything). Prereqs:
#   - pip install -r requirements.txt
#   - ~/.kaggle/kaggle.json in place
# Manual-only datasets (UR Fall, Le2i, Surveillance Camera Fight) are NOT fetched
# here — see docs/training.md for where to get them and where to put them.
#
# Time on an M2 Air: dataset download dominates. Action training is minutes once
# features are cached; the weapon YOLO is the long pole (hours at 80 epochs).
#
# Usage:  bash scripts/train_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root

ACTION_DATA="data/datasets/actions"
WEAPON_DIR="data/datasets/weapons"
ACTION_OUT="models/action_gru.pt"
WEAPON_OUT="models/weapons.pt"

# Fall back to CPU for unsupported MPS ops instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "==> [0/4] Pipeline self-check (fails fast if the feature/split logic broke)"
python tests/test_action_pipeline.py

echo "==> [1/4] Downloading action datasets from Kaggle"
python scripts/setup_datasets.py --source kaggle --out "$ACTION_DATA"

echo "==> [2/4] Training the action classifier"
python scripts/train_action.py --data "$ACTION_DATA" --out "$ACTION_OUT" \
    --epochs 40 --report reports/action_metrics.json

echo "==> [3/4] Downloading + training the weapon detector"
python scripts/train_weapon.py --auto-download --dataset-dir "$WEAPON_DIR" \
    --out "$WEAPON_OUT" --epochs 80 --imgsz 640

echo "==> [4/4] Post-training sanity check (action classifier)"
python eval/evaluate.py compare --data "$ACTION_DATA" --model "$ACTION_OUT" || true

echo
echo "Done. Before trusting these weights:"
echo "  1. Check the HELD-OUT TEST block from step 2 — especially violence→fall"
echo "     and fall→violence. The per-epoch validation numbers are not the score."
echo "  2. Check the per-source accuracy table. A large gap between sources means"
echo "     the model learned which dataset a clip came from, not what happens in it."
echo "  3. Check step 4: if the trained model does not beat the heuristic baseline,"
echo "     do not wire it into configs/default.yaml."
echo "  4. Set detect.weapon_labels from your weapon model's per-class mAP50"
echo "     (printed in step 3) — do not reuse another run's list."
