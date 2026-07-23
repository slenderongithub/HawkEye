# Training Guide — improving the models

This guide is for whoever trains HawkEye's models (e.g. on the RTX 3050). It
explains **why** the current detector confuses violence and fall, exactly what
in the codebase fixes it, and the step-by-step to train both the action
classifier and the weapon detector and plug them in.

You do **not** need to touch the app code to improve accuracy — you train a
model, drop the weights file in, and point the config at it.

> **Note:** you may notice a new `hawkeye/attendance/` module and an Attendance
> tab on the dashboard if you pull recent changes — that's an unrelated Phase 1
> feature (face-recognition attendance logging, reusing the same InsightFace
> identity DB) and needs no training. Nothing in this guide is affected by it.

This guide covers everything **from a fresh machine to a trained model plugged
in** — Section 0 is git/setup, Sections 1+ are the actual training.

---

## 0. Get the repo and set up your environment (RTX 3050 laptop)

### 0.1 Clone the repo and check out the branch

You need write access first — ask the repo owner to add you as a **Collaborator**
(GitHub → repo → Settings → Collaborators) if you haven't been added yet.

```bash
git clone https://github.com/slenderongithub/HawkEye.git
cd HawkEye
git checkout phase-1-core
git pull origin phase-1-core
```

Now branch off `phase-1-core` for your training work — don't push directly to
it, so we don't collide on the same branch:

```bash
git checkout -b train-action-model
```

### 0.2 Set up the Python environment (CUDA)

```bash
# 1) Install a CUDA build of PyTorch FIRST (match your installed CUDA version —
#    check with `nvidia-smi`; cu121 works for CUDA 12.1+):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Then the rest of the CUDA requirements:
pip install -r requirements-cuda.txt
```

Verify the GPU is visible:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If this prints `CUDA available: False`, the training scripts still run but print a
loud **"CPU/smoke-test configuration — NOT a real training run"** warning and fall
back to CPU (very slow, and hyper-parameters are not tuned for it). This fallback
exists only so the code can be import/smoke-tested on a non-CUDA machine — **fix
the CUDA/torch install before training on real data.** All the CUDA/CPU/MPS
branching lives in one place: `hawkeye/train/device_profile.py`.

### 0.3 Set up the Kaggle API token (for automatic dataset download)

Most datasets are auto-downloaded via the Kaggle API. Create a token once:

1. https://www.kaggle.com/settings/account → **API** → **Create New Token**
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

`pip install -r requirements-cuda.txt` already installs the `kaggle` package.

### 0.4 Get the datasets

See **Section 2** — one command auto-downloads the Kaggle-hosted sets into
`data/datasets/actions/<label>/` (gitignored; datasets never go through git).

### 0.5 Train, then commit + push your CODE changes

Once you've trained a model (Section 3) and are happy with it (Section 4 wiring
it in), commit and push **only the code/config changes** — never the `.pt`
weights file itself (it's gitignored on purpose):

```bash
git add -A
git commit -m "Train action classifier on RWF-2000 + fall dataset, wire into Stage A"
git push -u origin train-action-model
```

Then open a Pull Request on GitHub: **base `phase-1-core` ← compare
`train-action-model`**. Send the trained `models/action_gru.pt` file to the
repo owner directly (Drive/USB/etc.) so they can drop it in locally too.

---

## 1. Why violence and fall get confused (and the fix)

The app ships with **zero-training heuristics** (`hawkeye/detect/heuristics.py`)
that look at a *single frame* of pose keypoints. That's why it mixes up classes:

- A single frame can't tell a **fall** (a downward collapse, then stillness)
  from a **fight** (sustained two-person motion) — both just look like "limbs in
  odd positions."
- The heuristics have no learned decision boundary, so there's no *margin*
  between classes — borderline poses flip between labels.

The trained classifier fixes this with **five levers**, all already implemented
in `scripts/train_action.py` and on by default:

| Lever | Where | Why it separates the classes |
|---|---|---|
| **Separating features** | `hawkeye/detect/skeleton_features.py` | Encodes fall cues (torso verticality, hip dropping in frame, wide bbox) *and* fight cues (person count, closeness, mutual motion) as explicit dimensions. When the thing that distinguishes the classes is a literal feature, the boundary is clean. |
| **Temporal model** | `hawkeye/detect/action_model.py` (`ActionNet`: Conv1d→BiGRU→attention) | Reads ~1.5s of motion, not one frame — so it sees the *trajectory* (collapse-then-still vs sustained struggle). |
| **Angular-margin head** | `ArcMargin` in `action_model.py` (`--no-margin` to disable) | ArcFace-style additive margin literally **maximizes the angular margin between classes** — the direct answer to "the margins aren't maximized." |
| **Class balancing** | inverse-frequency weights in `train_action.py` | Stops a large violence set from swamping a small fall set (a top cause of one class being predicted as the other). |
| **Augmentation** | `augment()` in `train_action.py` | Mirror flips, jitter, noise, keypoint dropout → generalizes from few clips. |

And every epoch prints a **confusion matrix** so you can literally watch the
violence↔fall off-diagonal cells shrink as you tune.

> **Golden rule:** features are computed by the *same* module
> (`skeleton_features.py`) during training and live inference. Never change
> features in only one place — change that one file and both stay in sync.

---

## 2. Get the datasets

Datasets are gitignored and land in `data/datasets/actions/<label>/`. There are
two paths: **automated** (Kaggle) for most of the data, and **manual** for the
few sources not hosted on Kaggle.

### 2a. Automated (Kaggle) — one command

```bash
python scripts/setup_datasets.py --source kaggle
```
This downloads and auto-organizes:

| Class | Auto-downloaded source |
|---|---|
| `violence/` | **RWF-2000** (Fight split) + **Hockey/Movies** fights |
| `normal/`   | **RWF-2000** (NonFight split) + **UCF101** filtered to daily-living classes (walking, typing, mopping, …) as hard negatives |
| `fall/`     | a Kaggle **fall-detection** mirror |

Notes:
- Kaggle slugs drift (mirrors get renamed/removed). The downloader tries several
  candidates per source, logs any misses, and continues — a missing dataset never
  aborts the run. If a source is skipped, the script prints how to find a current
  slug (`kaggle datasets list -s <name>`).
- Downloads are cached under `data/datasets/actions/_raw_kaggle/`; delete that
  folder to reclaim disk once the class folders are populated.
- Restrict to specific classes with e.g. `--classes fall` (useful for re-runs).

### 2b. Manual (non-Kaggle sources) — optional accuracy boost

These are university/paper-hosted and must be downloaded by hand, then sorted:

| Class | Manual source |
|---|---|
| `fall/`     | **UR Fall Detection** (`http://fenix.ur.edu.pl/~mkepski/ds/uf.html`), **Le2i** |
| `violence/` | **Surveillance Camera Fight Dataset** (paper GitHub/Drive) |

After downloading, sort a folder into the class layout with the same script:
```bash
python scripts/setup_datasets.py --source local --local-dir <download_folder>
```

### Layout & balance

Final layout the trainer expects (folder names = class labels; `nonviolence`,
`fight`, `falling`, etc. auto-map via `_ALIASES`):
```
data/datasets/actions/
├── violence/  *.mp4
├── fall/      *.mp4
└── normal/    *.mp4
```
Check counts/balance any time with:
```bash
python scripts/setup_datasets.py --verify-only
```
**Balance matters:** aim for a similar number of clips per class. Class-weighting
compensates for imbalance, but more real clips for the minority class helps more.

---

## 3. Train the action classifier

> **One-command option:** `bash scripts/train_all.sh` runs the whole pipeline
> (download → train action → download+train weapon → eval sanity pass). Each line
> in it is a standalone command you can also run individually. The rest of this
> section covers the action step on its own.

On the RTX 3050 (CUDA auto-detected):
```bash
source .venv/bin/activate            # or the CUDA env
python scripts/train_action.py --data data/datasets/actions --epochs 30
```

The device and its hyper-parameters come from `hawkeye/train/device_profile.py`:
on **CUDA** it enables mixed-precision (AMP), pins memory, and uses a larger
batch/model capacity; on non-CUDA it drops to the safe CPU smoke-test config.
Force a device with `--device cuda|cpu` and override batch with `--batch N`.

What happens:
1. Extracts YOLO-pose features from every clip (cached to
   `data/datasets/action_features.npz` — re-runs skip this and are fast).
   **After adding new clips, delete that `.npz`** (or pass `--cache ""`) so the
   new data is actually picked up — otherwise the stale cache is reused.
2. Stratified train/val split, standardization, class weights.
3. Trains `ActionNet` with the angular-margin head.
4. Prints per-epoch **confusion matrix + precision/recall/F1**.
5. Saves the best-val checkpoint to `models/action_gru.pt`.

Useful flags:
```bash
--epochs 50 --batch 32 --lr 1e-3      # training length / size / step
--seq-len 32                          # frames per sample (~1.5s at 20 fps)
--window-stride 16                    # sliding-window step (smaller = more samples)
--no-margin                           # ablation: plain softmax instead of angular margin
--label-smoothing 0.05
```

### Reading the confusion matrix (your main tool)

Rows = true class, columns = predicted. The number at
`[violence, fall]` is "violence clips wrongly called fall." Watch those
off-diagonal cells. If they stay high:
- **More/cleaner data** for the confused pair (biggest lever).
- Increase the margin: bump `m` in `ArcMargin(...)` from `0.30` toward `0.40`.
- Longer clips: raise `--seq-len` so the temporal difference is clearer.
- Check labels aren't mixed (a mislabeled clip poisons the boundary).

Compare `--no-margin` vs default runs — the margin head should raise F1 and
shrink the violence↔fall off-diagonals. That comparison is a great thesis result.

---

## 4. Plug the trained model in (no code changes)

Edit `configs/default.yaml`:
```yaml
detect:
  action_model: "models/action_gru.pt"   # <- your trained weights
  action_min_conf: 0.5                    # raise to be stricter about triggering
```
Restart HawkEye. On startup you'll see
`Stage A action source: TRAINED model (...)` instead of `heuristics`. The
classifier now drives violence/fall detection; everything downstream (Stage B
reasoning, identity, dispatch) is unchanged.

Share the `.pt` file with teammates directly (Drive/USB) — it's gitignored and
does **not** go through the PR. Only the *code* changes are committed.

### 4.1 Re-tune the config thresholds from the run's metrics

Don't leave the thresholds at their old values — set them from what this run
actually produced (a deliberate manual step; the safety-critical thresholds
deserve a human looking at the curves):

1. **`action_min_conf`** — from the trainer's final per-class precision/recall and
   a held-out check:
   ```bash
   python eval/evaluate.py test-action --data data/datasets/actions --model models/action_gru.pt
   python eval/evaluate.py compare     --data data/datasets/actions --model models/action_gru.pt
   ```
   Pick the confidence that maximizes F1, but **favor recall on `violence`** — a
   missed fight is worse than a false alarm. `compare` must show the trained model
   beating the heuristic baseline; if it doesn't, something in the pipeline is
   wrong, not just under-trained.
2. **`weapon_conf` / `weapon_labels`** — see Section 5.

---

## 5. Train the weapon detector (knife / gun / etc.)

Weapon detection is a separate YOLO object detector, disabled until you provide
one. `scripts/train_weapon.py` fine-tunes YOLO11 and uses the same
`device_profile.py` (CUDA → `batch=-1` auto-batch to ~60% VRAM + AMP; a safe
fallback otherwise).

**Automated (Kaggle):**
```bash
python scripts/train_weapon.py --auto-download --epochs 80 --imgsz 640
```
This downloads a Kaggle weapon dataset into `data/datasets/weapons/`, and:
- if the download already ships a YOLO `data.yaml`, uses it directly;
- if it's **Pascal-VOC** (XML annotations, e.g. the "Guns Object Detection" set),
  converts it to YOLO format (`convert_voc_to_yolo`) and builds a train/val split
  + `weapons.yaml` automatically.

Weapon mirrors vary a lot; if auto-assembly can't figure out the layout it says so
and points you at the raw folder to organize by hand (`--create-yaml`).

**Manual dataset** (already in YOLO format):
```bash
python scripts/train_weapon.py --data path/to/weapons.yaml --epochs 80 --imgsz 640
```

**Plug it in** — set in `configs/default.yaml`:
```yaml
detect:
  weapon_model: "models/weapons.pt"
  weapon_conf: 0.6          # moderately high — weapon false positives are costly
  weapon_labels: ["gun", "pistol", "rifle", "shotgun", "firearm"]
```
`hawkeye/detect/weapons.py` filters detections to `weapon_labels` and raises a
high-priority `WEAPON` anomaly that routes to `police`.

> **Set `weapon_labels` from the run's per-class mAP.** After training, the script
> prints mAP50/mAP50-95 (and writes a PR curve under `runs/detect/…`). Only include
> classes that clear a real bar (mAP50 > ~0.5). **Knife detection has historically
> scored ~0% mAP50 — keep it excluded from `weapon_labels` until a run proves it
> works.** Pick `weapon_conf` from the PR curve's knee.

---

## 6. Where each piece lives (quick map)

```
hawkeye/detect/skeleton_features.py  features (train + infer share this) ← edit here
hawkeye/detect/action_model.py       ActionNet + ArcMargin head
hawkeye/detect/action_classifier.py  live inference wrapper (rolling window)
hawkeye/detect/weapons.py            weapon YOLO wrapper
hawkeye/train/device_profile.py      CUDA/CPU device + hyper-param profile (shared)
scripts/train_action.py              action-classifier training
scripts/train_weapon.py              weapon-detector training (+ --auto-download)
scripts/setup_datasets.py            dataset download/organize (--source kaggle|local)
scripts/_kaggle_utils.py             shared Kaggle download helper
scripts/train_all.sh                 one-command end-to-end pipeline
eval/evaluate.py                     test-action / rwf2000 / compare
configs/default.yaml                 detect.action_model / detect.weapon_model
```

Future upgrade (optional, for the thesis): swap `ActionNet` for a true **ST-GCN**
(graph conv over the skeleton) — it plugs into the same checkpoint interface.
