# HawkEye — Setup

## macOS (development + training)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py         # optional
python -m hawkeye.run                      # http://127.0.0.1:8000
```

macOS camera permission: grant Terminal/iTerm access under *System Settings →
Privacy & Security → Camera* the first time.

Optional local VLM offline mode:
```bash
pip install transformers accelerate einops pillow
python -m hawkeye.run --backend moondream
```

- **Datasets and model weights do NOT travel through git** (gitignored / too big).

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
