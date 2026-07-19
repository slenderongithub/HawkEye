# HawkEye — Indoor Surveillance & Attendance

Real-time system that watches a camera feed and does **two jobs from one local face database**:

1. **Surveillance** — **detects** anomalies (violence, falls/medical, weapons, intrusion, accidents, fire — and novel "any bad thing" events), **explains what happened in plain language**, **identifies who** was involved, and **decides the response** (police / ambulance / fire).
2. **Attendance** — continuously recognizes enrolled members on camera and **logs who's present** (one record per person per day).

Everything is shown on a live web dashboard. *"HawkEye doesn't just watch for danger — it knows who's in the building."*

Phase 1 (this build) runs on a **hand-moved / fixed camera**. The drone (a human-piloted LiteWing + WiFi camera) is a Phase 2 plug-in; production hardening is Phase 3. See the plan file for the full roadmap.

## Why it's different (USP)

1. **Explain, don't just detect** — a triggered vision-language layer writes a natural-language incident report (basis: LAVAD, CVPR 2024; Holmes-VAD).
2. **Open-vocabulary** — the VLM can flag anomalies it was never trained on ("a tree fell", "an abandoned bag"), directly answering the "any other bad thing" requirement.
3. **Fast/slow cascade** — a lightweight always-on detector runs real-time; the heavy VLM fires **only on suspicion**, which also **kills false alarms**.
4. **Anomaly → Identity → Action, one loop** — names *who* and decides *what to do*, not just "anomaly: yes".
5. **Camera-agnostic** — the same pipeline runs on webcam → phone → drone (Phase 2 swaps only the video source).
6. **Gated, auditable dispatch** — every automated action is severity/confidence-gated, cooldown-protected, and logged.
7. **Dual utility from one face DB** — the same enrolled-member database powers both perpetrator identification *and* passive attendance logging; enroll once, get both.

## Architecture

```
camera ─▶ Stage A (real-time)          ─▶ Stage B (triggered VLM)  ─▶ Identity  ─▶ Decision ─▶ Dashboard
         pose·fight/fall·weapon·motion    explain·severity·action     face match    gated dispatch
```

- **Stage A** (`hawkeye/detect/`): YOLO11 pose + keypoint heuristics (fight/fall) + motion anomaly + optional weapon YOLO → fused anomaly event.
- **Stage B** (`hawkeye/reason/`): pluggable reasoner — **Claude** (cloud, best quality) · **Moondream2** (local offline) · **mock** (no deps).
- **Identity** (`hawkeye/identity/`): InsightFace (RetinaFace + ArcFace) + cosine match vs a local ~20-person DB. Used both for perpetrator ID on incidents and for attendance.
- **Attendance** (`hawkeye/attendance/`): a slow-cadence worker (off the real-time path) that recognizes all enrolled members in view and logs presence — one record per person per day.
- **Decision** (`hawkeye/decision/`): config-driven rules + gating + cooldown + dispatchers.
- **Server** (`server/`): FastAPI + WebSocket live feed + REST + dashboard (Incidents / Attendance / People tabs).

## Setup

**macOS (dev — CPU/MPS):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py      # optional: warm the weights
```

**RTX 3050 laptop (GPU station — training / faster local VLM):** install a CUDA build of PyTorch, then `pip install -r requirements-cuda.txt`. Sync code Mac→3050 over git (`dev` branch); datasets/weights are set up once on the 3050 (they're gitignored).

The core runs even if optional groups fail: no `insightface` → identity disables itself; no `ANTHROPIC_API_KEY`/`anthropic` → Stage B falls back to the mock reasoner.

## Run

```bash
python -m hawkeye.run                    # webcam 0
python -m hawkeye.run --source clip.mp4  # a video file (looped)
python -m hawkeye.run --backend mock     # force offline mock reasoner
# cloud reasoning:
export ANTHROPIC_API_KEY=sk-...          # then run without --backend (auto picks Claude)
```
Open **http://127.0.0.1:8000**. Move the camera around, stage a fall / mock scuffle / show an object → watch the incident cards appear with explanation, face match, and action.

## Enroll faces (local identity DB)

```bash
python scripts/enroll_person.py --name "Asha Rao" --images data/faces/asha \
    --meta '{"role":"staff","id_no":"EMP-014"}'
```
Or use the **People** tab in the dashboard. (Local, consented dataset only — no Aadhaar/government-ID; that's future work pending official API access. See `docs/ethics.md`.)

## Configure

Everything is in `configs/default.yaml` — trigger threshold, clip length, cooldowns, reasoner backend, identity match threshold, and the anomaly→action dispatch rules. Override with `--config myconfig.yaml`.

## Evaluate

```bash
python eval/evaluate.py fps --source clip.mp4   # Stage A latency / FPS
```
Dataset metrics (UCF-Crime AUC, RWF-2000 accuracy, false-alarm-rate A vs A+B) are scaffolded in `eval/` and `scripts/train_action.py`.

## Project layout

```
hawkeye/   ingest · detect (Stage A) · reason (Stage B) · identity · decision · events · pipeline
server/    FastAPI app + dashboard
scripts/   enroll · download_models · train_action (ST-GCN scaffold)
eval/      metrics + FPS benchmark
docs/      architecture · ethics · setup
```

## Roadmap

- **Phase 1 (now):** model + app core on a manual camera. ✅ scaffolded end-to-end.
- **Phase 2:** autonomous drone — Tello `DroneSource` into `hawkeye/ingest/`, SLAM + coverage patrol (AirSim/Gazebo first), closed-loop "investigate".
- **Phase 3:** TensorRT/quantization, Docker, auth, DPDP-aligned encryption/audit, real dispatch integrations.
