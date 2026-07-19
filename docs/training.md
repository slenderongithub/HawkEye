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
If this prints `CUDA available: False`, the training script will still run (it
auto-falls back to CPU) but will be much slower — fix the CUDA/torch install
before training on real data.

### 0.3 Get the datasets

See **Section 2** below for exact sources — download them into
`data/datasets/actions/<label>/` (this folder is gitignored; datasets never go
through git).

### 0.4 Train, then commit + push your CODE changes

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

Download separately (they're gitignored) into `data/datasets/actions/<label>/`:

| Class folder | Suggested source |
|---|---|
| `violence/` | **RWF-2000** (2,000 real surveillance fight/non-fight clips), Hockey/Movies fight sets |
| `fall/`     | **UR Fall Detection**, **Le2i** fall datasets |
| `normal/`   | non-violent clips from RWF-2000 + ordinary walking/standing clips |

Final layout the trainer expects:
```
data/datasets/actions/
├── violence/  *.mp4
├── fall/      *.mp4
└── normal/    *.mp4
```
Folder names become the class labels. `nonviolence`, `fight`, `falling`, etc.
are auto-mapped to `normal`/`violence`/`fall` (see `_ALIASES` in
`action_classifier.py`). Add more classes (e.g. `theft/`) just by adding folders.

**Balance matters:** aim for a similar number of clips per class. If `fall` has
far fewer, the class-weighting helps, but more real clips helps more.

---

## 3. Train the action classifier

On the RTX 3050 (CUDA auto-detected):
```bash
source .venv/bin/activate            # or the CUDA env
python scripts/train_action.py --data data/datasets/actions --epochs 30
```

What happens:
1. Extracts YOLO-pose features from every clip (cached to
   `data/datasets/action_features.npz` — re-runs skip this and are fast).
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

---

## 5. Train the weapon detector (knife / gun / etc.)

Weapon detection is a separate object detector, disabled until you provide one.

1. **Get a weapon dataset** in YOLO format — e.g. open gun/knife detection sets
   on Roboflow/Kaggle (images + YOLO `.txt` labels).
2. **Fine-tune YOLO11** (ultralytics):
   ```bash
   yolo detect train model=yolo11n.pt data=weapons.yaml epochs=80 imgsz=640
   ```
   `weapons.yaml` lists your class names (e.g. `["knife","pistol","rifle"]`) and
   train/val image paths.
3. **Plug it in** — set in the config:
   ```yaml
   detect:
     weapon_model: "models/weapons.pt"
     weapon_conf: 0.4
   ```
   Restart. `hawkeye/detect/weapons.py` filters detections to weapon-like class
   names (`WEAPON_KEYWORDS`) and raises a high-priority `WEAPON` anomaly, which
   routes to `police` via the decision engine. Add class names to
   `WEAPON_KEYWORDS` if your dataset uses different words (e.g. `"machete"`).

Keep `weapon_conf` moderately high (0.4–0.6) — weapon false positives are costly.

---

## 6. Where each piece lives (quick map)

```
hawkeye/detect/skeleton_features.py  features (train + infer share this) ← edit here
hawkeye/detect/action_model.py       ActionNet + ArcMargin head
hawkeye/detect/action_classifier.py  live inference wrapper (rolling window)
hawkeye/detect/weapons.py            weapon YOLO wrapper
scripts/train_action.py              the training pipeline
configs/default.yaml                 detect.action_model / detect.weapon_model
```

Future upgrade (optional, for the thesis): swap `ActionNet` for a true **ST-GCN**
(graph conv over the skeleton) — it plugs into the same checkpoint interface.
