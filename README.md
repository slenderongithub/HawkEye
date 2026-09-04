# HawkEye — Indoor Surveillance & Attendance

Real-time system that watches a camera feed and does **two jobs from one local face database**:

1. **Surveillance** — **detects** anomalies (violence, falls/medical, weapons, intrusion, accidents, fire — and novel "any bad thing" events), **explains what happened in plain language**, **identifies who** was involved, and **decides the response** (police / ambulance / fire).
2. **Attendance** — continuously recognizes enrolled members on camera and **logs who's present** (one record per person per day).

Everything is shown on a live web app. *"HawkEye doesn't just watch for danger — it knows who's in the building."*

Phase 1 (this build) runs on a **hand-moved / fixed camera**. The drone (a human-piloted LiteWing + WiFi camera) is a Phase 2 plug-in; production hardening is Phase 3.

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
camera ─▶ Stage A (real-time)          ─▶ Stage B (triggered VLM)  ─▶ Identity  ─▶ Decision ─▶ App
         pose·action model·weapon·motion   explain·severity·action     face match    gated dispatch
```

- **Stage A** (`hawkeye/detect/`): YOLO11 pose + a trained skeleton-sequence action classifier (violence/fall/normal — falls back to zero-training heuristics if no weights are configured) + motion anomaly + a fine-tuned weapon YOLO → fused anomaly event.
- **Stage B** (`hawkeye/reason/`): pluggable reasoner — **Claude** (cloud, best quality) · **Moondream2** (local offline) · **mock** (no deps).
- **Identity** (`hawkeye/identity/`): InsightFace (RetinaFace + ArcFace) + cosine match vs a local ~20-person DB. Used both for perpetrator ID on incidents and for attendance.
- **Attendance** (`hawkeye/attendance/`): a slow-cadence worker (off the real-time path) that recognizes all enrolled members in view and logs presence — one record per person per day.
- **Decision** (`hawkeye/decision/`): config-driven rules + gating + cooldown + dispatchers.
- **Server + App** (`server/`, `app/`): one FastAPI process serves the REST API, the live WebSocket feed, *and* the operator app (`app/dist`, an Expo build) — one address for everything. See [`app/README.md`](app/README.md) for the app itself, and Section "Run" below for how the two fit together.

## Setup

**macOS (primary target — CPU/MPS, including model training):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py      # optional: warm the pretrained pose/identity weights
```

The core runs even if optional groups fail: no `insightface` → identity disables itself; no `ANTHROPIC_API_KEY`/`anthropic` → Stage B falls back to the mock reasoner. With no trained action/weapon weights configured, Stage A falls back to the zero-training heuristics — see [`docs/training.md`](docs/training.md) to train real ones.

## Run (camera + app, together)

The camera pipeline and the operator app are **one process, one address** — there is nothing to run "at the same time" separately. Build the app once, then start HawkEye:

```bash
# 1) Build the app (only needed once, and again whenever app/ changes)
cd app && npm install && npm run build:web && cd ..

# 2) Start HawkEye — this boots the camera pipeline AND serves the app + API + WebSocket
python -m hawkeye.run                    # webcam 0
python -m hawkeye.run --source clip.mp4  # a video file (looped) instead of a live camera
python -m hawkeye.run --backend mock     # force the offline mock reasoner
# cloud reasoning:
export ANTHROPIC_API_KEY=sk-...          # then run without --backend (auto picks Claude)
```

Open **http://127.0.0.1:8000** — that single URL is the whole app (Live / Incidents / Attendance / People / Settings). Move the camera around, stage a fall / mock scuffle / show an object → watch incident cards appear with explanation, face match, and action.

If you open that URL before step 1, HawkEye still runs (API + camera are live) and the page tells you to build the app rather than 404ing.

**Iterating on the app UI?** Run the Expo dev server instead of the static build, alongside the backend:
```bash
# terminal 1 — backend + camera
python -m hawkeye.run
# terminal 2 — app with hot reload, talking to the backend above
cd app && npm start   # press w for web
```

**A physical phone on the same Wi-Fi:** start the backend bound to your LAN IP (`python -m hawkeye.run --host 0.0.0.0`), then set the server address in the app's Settings tab. See [`app/README.md`](app/README.md).

## Enroll faces (local identity DB)

```bash
python scripts/enroll_person.py --name "Asha Rao" --images data/faces/asha \
    --meta '{"role":"staff","id_no":"EMP-014"}'
```
Or use the **People** tab in the app. (Local, consented dataset only — no Aadhaar/government-ID; that's future work pending official API access. See `docs/ethics.md`.)

## Configure

Everything is in `configs/default.yaml` — trigger threshold, clip length, cooldowns, reasoner backend, identity match threshold, action/weapon model paths, and the anomaly→action dispatch rules. Override with `--config myconfig.yaml`.

## Train / retrain the detection models

Stage A's action classifier (violence/fall) and weapon detector are trained models, not hardcoded rules — see [`docs/training.md`](docs/training.md) for the full guide (primary target: Apple Silicon/MPS). Quick path once datasets are staged:

```bash
python tests/test_action_pipeline.py     # pipeline self-check, seconds
bash scripts/train_all.sh                # download data -> train action -> train weapon -> eval
```

## Evaluate

```bash
python eval/evaluate.py fps --source clip.mp4                                          # Stage A latency / FPS
python eval/evaluate.py compare --data data/datasets/actions --model models/action_gru.pt  # trained model vs heuristic baseline
```

## Project layout

```
hawkeye/   ingest · detect (Stage A) · reason (Stage B) · identity · decision · events · attendance · pipeline
server/    FastAPI app (API + WebSocket + serves app/dist)
app/       operator app (Expo/React Native — Live · Incidents · Attendance · People · Settings)
scripts/   enroll · download_models · train_action · train_weapon · setup_datasets · train_all.sh
eval/      metrics + FPS benchmark
tests/     pipeline + shutdown self-checks
docs/      architecture · ethics · setup · training
```

## Roadmap

- **Phase 1 (now):** model + app core on a manual camera, trained action/weapon models, unified app. ✅
- **Phase 2:** autonomous drone — Tello `DroneSource` into `hawkeye/ingest/`, SLAM + coverage patrol (AirSim/Gazebo first), closed-loop "investigate".
- **Phase 3:** TensorRT/quantization, Docker, auth, DPDP-aligned encryption/audit, real dispatch integrations.
