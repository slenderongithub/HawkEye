# HawkEye — Setup & Two-Machine Workflow

## Machines

| Machine | Role |
|---|---|
| **MacBook Air M2 (8 GB)** | All development. Runs the full pipeline (Stage A on CoreML/MPS, Stage B via cloud or Moondream2, dashboard). |
| **RTX 3050 laptop (4 GB VRAM, CUDA)** | GPU station: model training + optional faster local-VLM inference. |

## macOS (development)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py         # optional
python -m hawkeye.run                      # http://127.0.0.1:8000
```

macOS camera permission: grant Terminal/iTerm access under *System Settings →
Privacy & Security → Camera* the first time.

## RTX 3050 laptop (GPU)

```bash
# 1) CUDA PyTorch first (match your CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2) the rest:
pip install -r requirements-cuda.txt
# 3) (optional) local VLM offline mode:
pip install transformers accelerate einops pillow
python -m hawkeye.run --backend moondream
```

## Sync workflow (permanent): git via GitHub, `dev` branch

Chosen because a git remote is **asynchronous** — the two laptops never need to
be on the same Wi-Fi at the same time.

```bash
# Mac (dev):
git add -A && git commit -m "wip" && git push origin dev
# 3050 (run):
git pull origin dev && python -m hawkeye.run --backend moondream
```

- Keep rapid test commits on `dev`; **squash-merge `dev` → `main` at milestones**.
- **Datasets, model weights, and the venv do NOT travel through git** (gitignored /
  too big) — set them up once on the 3050; they persist there.

## Reasoner backends

| Backend | Needs | Notes |
|---|---|---|
| `mock` | nothing | Templated explanation from Stage A. CI/offline default. |
| `claude` | `anthropic` + `ANTHROPIC_API_KEY` | Best explanations (the USP). Tiny cost (triggered only). |
| `moondream` | `transformers` (+ weights on first run) | Fully offline; fits 4 GB VRAM / 8 GB RAM. |

`backend: auto` picks Claude if a key is present, else mock.

## Camera options

- Webcam: `--source 0`
- Video file (loops — great for dataset demos): `--source clip.mp4`
- Phone as IP camera (install an IP-webcam app): `--source http://<phone-ip>:8080/video`
