# Training Guide — improving the models

This guide explains **why** the detector used to confuse violence and fall,
exactly what in the codebase fixes it, and the step-by-step to train both the
action classifier and the weapon detector and plug them in.

Training target is **Apple Silicon (M2, MPS)** —
`hawkeye/train/device_profile.py` picks the right settings for whatever it finds.

You do **not** need to touch the app code to improve accuracy — you train a
model, drop the weights file in, and point the config at it.

This guide covers everything **from a fresh machine to a trained model plugged
in** — Section 0 is setup, Sections 1+ are the actual training.

---

## 0. Set up your environment

### 0.1 Get the repo

```bash
git clone https://github.com/slenderongithub/HawkEye.git
cd HawkEye
```

Work on a branch, not `main`:

```bash
git checkout -b train-action-model
```

### 0.2 Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the accelerator is visible:
```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```
MPS is a **first-class training target** — the action classifier is small
(a few million parameters over a 64-frame window) and trains in minutes on an
M2 once features are cached. If MPS is not available the scripts still run on
CPU, just slowly. All device branching lives in one place:
`hawkeye/train/device_profile.py`.

### 0.2b Run the self-checks first

```bash
python tests/test_action_pipeline.py
python tests/test_pipeline_shutdown.py
```

These take seconds and assert the things that silently destroyed the previous
model (see Section 1). If they fail, fix that before spending hours training.

### 0.3 Set up the Kaggle API token (for automatic dataset download)

Most datasets are auto-downloaded via the Kaggle API. Create a token once:

1. https://www.kaggle.com/settings/account → **API** → **Create New Token**
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

`pip install -r requirements.txt` already installs the `kaggle` package.

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

### 0.6 Training on Windows + NVIDIA CUDA

The trainers are device-agnostic — `hawkeye/train/device_profile.py` resolves
`cuda → mps → cpu` and picks the matching knobs. Three things are specific to a
Windows/CUDA box:

**1. Install a PyTorch wheel that matches the GPU.** Use
`requirements-cuda.txt` instead of `requirements.txt`:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
# RTX 50-series (Blackwell) needs cu128 + torch>=2.7 — older wheels have no sm_120 kernels
pip install "torch>=2.7" torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-cuda.txt
```

Verify the wheel actually drives the card — `torch.cuda.is_available()` returns
`True` even when the wheel has no kernels for it, so run a real op:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); print((torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item())"
```

A `no kernel image is available for execution on the device` error here means the
CUDA build is too old for the GPU — reinstall from the right `--index-url`.

**2. DataLoader workers are forced to 0 for the action trainer on Windows.**
Windows spawns workers instead of forking, so the DataLoader must pickle its
Dataset; `train_action.py`'s Dataset closes over the in-RAM feature cache, which
is both unpicklable and would be copied per worker. This is handled
automatically in `device_profile.py` — the GPU step dominates anyway. The weapon
trainer (ultralytics, disk-bound, picklable dataset) keeps its workers.

**3. Redirecting output to a file needs UTF-8.** These scripts print `✓ ═ →`,
and Windows uses the legacy code page for redirected (non-console) output, so
`python scripts/train_action.py ... > train.log` raises `UnicodeEncodeError`.
Either don't redirect, or force UTF-8:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

## 1. Why violence and fall got confused (and the fix)

There were two separate problems: a **modelling** one and, far more damaging,
two **silent bugs** that made the previous model useless in production while
reporting ~0.95 validation macro-F1. Read this section before changing anything.

### 1a. The two bugs that made a "0.95 F1" model behave randomly

| Bug | What happened | Fix |
|---|---|---|
| **Train/serve feature skew** | Training extracted features with `motion=0.0` hardcoded, while the live pipeline passed a real scene-motion score. Standardisation divided that dimension by its training std (≈0, floored to `1e-6`), so a live motion value of `0.35` entered the network as **~350,000**. One dimension, five orders of magnitude out, dominated every activation. | Training now computes motion with the *same* `MotionAnomaly` class the live pipeline uses. `skeleton_features.standardize()` is shared by both paths, **floors variance** at `MIN_STD` and **clips output** to ±8, so this class of bug cannot recur silently. |
| **Window-level data leakage** | Clips were cut into windows with 50% overlap and split *randomly*. Window `[0:32]` went to train and `[16:48]` — sharing half its frames — went to validation. The model memorised clips; validation applauded. Every number the old run printed was fiction, which is why nobody caught the bug above. | The split is now **grouped by source clip** (`group_split()`): every window from one video goes entirely to train, val, or test. There is also a real **held-out test set**, evaluated exactly once. |

Both are guarded by `tests/test_action_pipeline.py`. Run it.

### 1b. The modelling problem

A single frame can't tell a **fall** (a downward collapse, then stillness) from a
**fight** (sustained two-person motion) — both are "limbs in odd positions". The
current pipeline separates them by construction:

| Lever | Where | Why it separates the classes |
|---|---|---|
| **Two-actor features** | `hawkeye/detect/skeleton_features.py` | The vector carries actor A *and* the nearest actor B, plus the geometry *between* them — wrist→head and wrist→torso reach distances, closing speed, vertical dominance. Violence is a two-body interaction; encoding only one skeleton threw that away. |
| **Fall kinematics** | same module (`DYN` block) | A fall is downward hip **acceleration** followed by **stillness**. Downward velocity alone is not enough — sitting down fast has that too. Both are explicit dimensions. |
| **Occlusion awareness** | `A_KPC` block | Keypoint confidences are features. A zeroed occluded joint would otherwise read as "joint located exactly at the hip", which the network will happily learn from. |
| **Multi-scale temporal model** | `action_model.py` (`MultiScaleStem` → BiGRU → attention+mean+max pooling) | Parallel 3/5/9-frame kernels see a short ballistic fall and a long sustained fight in the same window. Falls are peaky (max pooling matters), fights are sustained (mean pooling matters). |
| **Calibrated confidence** | temperature scaling + per-class thresholds | The old ArcFace head scaled logits by 30, saturating softmax at ~1.0 — so the `action_min_conf` gate every trigger depended on was decorative. A temperature is now fitted on validation and each class gets its own threshold at a target precision. |
| **Temporal hysteresis** | `action_classifier.py` | A class must hold across several consecutive inference ticks before it is emitted. One window over threshold is noise, not an event. |
| **Anti-shortcut augmentation** | `augment()` in `train_action.py` | Camera roll, anisotropic scaling, occlusion, and **actor-B dropout**. That last one matters: without it the net shortcuts to "two skeletons ⇒ violence", which is exactly how a bystander standing over a collapsed person becomes an assault alert. |

> **Golden rule:** features and standardisation are computed by the *same* module
> (`skeleton_features.py`) in training, evaluation, and live inference. Never
> reimplement either in a second place — that is precisely how the skew bug
> happened, and it is why `scripts/run_pipeline.py` (a drifted duplicate trainer)
> was deleted.

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

```bash
source .venv/bin/activate
python scripts/train_action.py --data data/datasets/actions --epochs 40 \
    --report reports/action_metrics.json
```

The device and its hyper-parameters come from `hawkeye/train/device_profile.py`.
Force a device with `--device mps|cpu` and override batch with `--batch N`.

What happens:
1. Extracts YOLO-pose + motion features from every clip, cached **per clip** to
   `data/datasets/action_features.npz` — re-runs skip this and are fast.
   **After adding new clips, delete that `.npz`** so the new data is picked up.
2. **Grouped split by source clip** into train / val / test — no window from one
   video ever spans two splits.
3. Trains `ActionNet`, printing a confusion matrix and the violence↔fall
   confusion rate every epoch.
4. Fits a **temperature** and **per-class thresholds** on validation.
5. Evaluates the **held-out test set once**, plus a per-source accuracy table.
6. Saves everything (weights, mean/std, temperature, thresholds, test confusion)
   to `models/action_gru.pt`.

Useful flags:
```bash
--epochs 60 --batch 32 --lr 2e-3      # training length / size / step
--seq-len 64                          # frames per window (~3s at 20 fps)
--window-stride 24                    # sliding-window step (smaller = more samples)
--target-precision 0.90               # operating point each class's threshold aims for
--test-split 0.15 --val-split 0.15
--max-per-class 655                   # cap clips per class (0 = use everything)
--extract-batch 32                    # frames per pose forward pass (see below)
--window-cache-gb 4.0                 # RAM for caching built windows (0 = off)
--no-tf32                             # disable TF32 matmuls on CUDA
--margin                              # ablation: ArcFace head (uncalibrated logits)
```

#### Keeping the GPU busy

Two things dominate the wall clock, and neither is the model — it is small enough
that a modern GPU is mostly idle unless it is fed properly.

**Feature extraction** is the bigger one by far (millions of frames). Pose runs
`--extract-batch` frames per forward pass; one frame per call spends most of its
time on launch overhead. Measured on an M2 (MPS), 100 frames of a real clip:

| `--extract-batch` | throughput |
|---|---|
| 1 (the old behaviour) | 55 fps |
| 32 (default) | 159 fps |
| 64 | 163 fps |

Batching is worth ~3×, and past 32 it flattens while frame buffers keep growing,
so 32 is the default. Video decode is *not* the constraint (~3500 fps). Raise it
on a large-VRAM card, lower it if extraction hits an out-of-memory error.

**Per-epoch cost** is dominated by rebuilding windows, not by gradients. A window
is a pure function of `(clip, start)`, but it is used once per epoch for training
and again by the validation pass each epoch — so over 40 epochs each window was
being rebuilt dozens of times on the main thread while the GPU waited. They are
now cached when they fit in `--window-cache-gb` (a window is
`seq_len × 188 × 4` bytes ≈ 47 KB at the defaults). The run prints whether the
cache is on:

```
window cache: ON (61234 windows ≈ 2.75 GB, budget 4.0 GB)
```

If it says `off` because of the budget, raise `--window-cache-gb` when you have
the RAM. It is disabled automatically when the loader uses worker processes,
since each worker would fill its own copy.

`--batch` is deliberately left at the device-profile default. Raising it does
increase GPU utilisation, but it also changes the optimisation (the LR schedule
is tuned for the default), so treat it as a modelling choice, not a free speedup
— and keep it identical across runs you intend to compare.

#### Balanced vs. imbalanced — running both off one cache

Class counts are usually lopsided (falls are the scarce class). Loss is
class-weighted to compensate, but whether that beats simply training on equal
counts is an empirical question, so test it rather than assume:

```bash
# 1. full set — builds the feature cache
python scripts/train_action.py --data data/datasets/actions \
    --out models/action_gru_full.pt --cache data/datasets/action_features.npz \
    --epochs 40 --report reports/action_metrics_full.json

# 2. balanced — SAME --cache, so extraction is skipped entirely
python scripts/train_action.py --data data/datasets/actions \
    --out models/action_gru_balanced.pt --cache data/datasets/action_features.npz \
    --max-per-class 655 --epochs 40 --report reports/action_metrics_balanced.json
```

`--max-per-class` is applied **after** the cache is loaded, which is the whole
point: pose extraction is hours, training is minutes, so the second run costs
only the minutes. Do not build a second copy of the dataset for this — a
symlinked or duplicated tree just re-extracts the same features under a
different path.

Compare the two `reports/*.json` on `test_macro_f1` first, then on the
violence↔fall entries of `test_confusion`. Keep the winner's `.pt`.

### Reading the output (in order of how much it matters)

1. **The held-out TEST block, not the per-epoch validation numbers.** Validation
   picked the checkpoint and tuned the thresholds, so it is optimistic by
   construction. The test block is the honest number.
2. **`violence→fall` and `fall→violence`.** This is the pair the whole pipeline
   exists to keep apart; it is printed on its own line for that reason. Rows =
   true class, columns = predicted.
3. **The per-source accuracy table.** If accuracy is high on one source dataset
   and low on another, the model learned *which dataset a clip came from* —
   camera angle, resolution, compression — rather than what happens in it. This
   is the single most common way a surveillance model looks excellent offline and
   fails on your actual camera. A large gap means: mix sources within each class,
   or gather data from the deployment camera itself.
4. **The comparison against the heuristic baseline** (`eval/evaluate.py compare`).
   If the trained model does not beat `hawkeye/detect/heuristics.py`, it is not
   ready — ship the heuristics instead.

If violence↔fall confusion stays high:
- **More and better-mixed data** for the confused pair (biggest lever by far).
- Raise `--seq-len` so the full trajectory (collapse → stillness) fits the window.
- Check for mislabeled clips — one poisons the boundary more than ten help it.
- Check the per-source table first; a domain gap masquerades as class confusion.

Run `--margin` as an ablation if you want the ArcFace comparison, but note its
logits are uncalibrated, so the per-class thresholds mean less on that path.

---

## 4. Plug the trained model in (no code changes)

Edit `configs/default.yaml`:
```yaml
detect:
  action_model: "models/action_gru.pt"   # <- your trained weights
  action_min_conf: 0.60                   # fallback only; checkpoint thresholds win
```
Restart HawkEye. On startup you'll see
`Stage A action source: TRAINED model (...)` instead of `heuristics`, and a log
line listing the per-class thresholds and temperature loaded from the checkpoint.
The classifier now drives violence/fall detection; everything downstream (Stage B
reasoning, identity, dispatch) is unchanged.

The `.pt` file is gitignored on purpose — share it directly, commit only code.

### 4.1 Re-tune the config from the run's metrics

Most thresholds are now **fitted during training and stored in the checkpoint**
(`temperature`, per-class `thresholds` at `--target-precision`), so there is far
less to hand-tune than before. `action_min_conf` is only a fallback for classes
the checkpoint has no threshold for.

What still deserves a human:

1. **Verify before trusting.** The trained model must beat the baseline:
   ```bash
   python eval/evaluate.py compare --data data/datasets/actions --model models/action_gru.pt
   ```
   If it doesn't, something is wrong in the pipeline, not just under-trained.
   To measure real generalisation, point `--data` at a dataset the model has
   never seen rather than the training directory.
2. **`--target-precision`** — retrain with a different value to move every class's
   operating point at once. Higher = fewer false alarms, more misses.
3. **`action_hysteresis_window` / `action_hysteresis_hits`** — raise `hits` if you
   still see brief flickers on the live feed; lower it if genuine short events
   (a single hard shove) are being missed.
4. **`weapon_conf` / `weapon_labels`** — see Section 5.

---

## 5. Train the weapon detector (knife / gun / etc.)

Weapon detection is a separate YOLO object detector, disabled until you provide
one. `scripts/train_weapon.py` fine-tunes YOLO11 and uses the same
`device_profile.py` (a safe
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
  weapon_labels: []         # empty = the built-in list (guns AND knives)
```
`hawkeye/detect/weapons.py` filters detections to `weapon_labels` and raises a
high-priority `WEAPON` anomaly that routes to `police`. Matching is token-aware,
so `handgun`, `machine_gun` and `pocket-knife` all match without a label like
`begun` sneaking through.

> **Set `weapon_labels` from your own run's per-class mAP.** After training, the
> script prints mAP50/mAP50-95 (and writes PR curves under `runs/detect/…`). Only
> narrow the list to exclude classes that genuinely failed to learn (mAP50 below
> ~0.5). Do **not** carry over another run's exclusions: a previous model scored
> ~0% mAP50 on knives, but that was a fact about that dataset and that run, not
> about knives. Judge your model on your numbers. Pick `weapon_conf` from the PR
> curve's knee.

---

## 6. Where each piece lives (quick map)

```
hawkeye/detect/skeleton_features.py  features + standardize() — train, eval AND
                                     infer all share this ← edit here, only here
hawkeye/detect/action_model.py       ActionNet (multi-scale → BiGRU → pooling)
hawkeye/detect/action_classifier.py  live inference wrapper (window, thresholds,
                                     hysteresis, physical guards)
hawkeye/detect/weapons.py            weapon YOLO wrapper + label matching
hawkeye/train/device_profile.py      MPS/CPU device + hyper-param profile
scripts/train_action.py              action training (grouped split, calibration)
scripts/train_weapon.py              weapon-detector training (+ --auto-download)
scripts/setup_datasets.py            dataset download/organize (kaggle|local)
scripts/_kaggle_utils.py             shared Kaggle download helper
scripts/train_all.sh                 one-command end-to-end pipeline
eval/evaluate.py                     fps / test-action / rwf2000 / compare
tests/test_action_pipeline.py        self-checks for the bugs in Section 1a
tests/test_pipeline_shutdown.py      self-checks for shutdown ordering
configs/default.yaml                 detect.action_model / detect.weapon_model
```

Future upgrade (optional, for the thesis): swap `ActionNet` for a true **ST-GCN**
(graph conv over the skeleton) — it plugs into the same checkpoint interface.
