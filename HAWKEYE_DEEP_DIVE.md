# HawkEye: Complete Architecture & Technical Deep Dive

---

## 1. Executive Summary: What is HawkEye?

**HawkEye** is an intelligent indoor surveillance and continuous passive attendance edge platform. It addresses the core failure modes of conventional computer vision security systems:

> *"Traditional cameras detect motion or bounding boxes without understanding context. HawkEye watches for danger, understands what happened, names who was involved, decides what to do, and continuously tracks who is in the building—all from a single camera feed and a shared, consented local biometric database."*

### The Dual-Mission Model
From a single video stream and one local face database, HawkEye performs two simultaneous jobs:
1. **Surveillance (Threat Detection & Incident Response)**: Detects violence, falls/medical collapses, brandished weapons, intrusion, accidents, fire, and open-vocabulary anomalies. When an incident occurs, it triggers an AI vision-language model to write a plain-English explanation, matches the subject against enrolled members, and dispatches automated response protocols (police, ambulance, fire) with strict gating and cooldowns.
2. **Passive Continuous Attendance**: Off the hot detection path, it runs a background recognition worker every ~3 seconds that passively scans people in the frame, logs attendance (one record per person per day), and pushes live updates to the dashboard without requiring people to stop and badge in.

---

## 2. Core Architecture: The Fast/Slow Cascade & Multi-Thread Pipeline

HawkEye is designed around a **Fast/Slow cascade** (inspired by research like *LAVAD, CVPR 2024* and *Holmes-VAD*):
- Running heavy Vision-Language Models (VLMs) on every video frame is computationally impossible at 20–30 FPS on edge devices.
- Traditional lightweight heuristics or bounding-box detectors produce high rates of false alarms.
- **HawkEye's Solution**: A lightweight Stage A detector runs continuously in real-time. When it detects suspicious motion, pose changes, weapons, or actions crossing a threshold, it snapshots a 3-second buffer of video frames and hands it off to an asynchronous Stage B VLM. The VLM acts as an **intelligent semantic filter** that verifies the event, dismisses false alarms, and generates natural-language incident summaries.

```
                  ┌───────────────────────────────────────────────────────────┐
                  │                 CAPTURE THREAD (Real-time ~20 FPS)        │
Video Source ───▶ │  Frame Ingest ──▶ Stage A: YOLO11 Pose + ActionNet (BiGRU)│
(Webcam/IP/File)  │                   + Weapons YOLO + Motion Anomaly         │
                  │  Annotate Frame ──▶ StreamHub (MJPEG /ws @ 15 FPS)        │
                  └───────────────┬───────────────────────────────────────────┘
                                  │ (On Trigger: snapshot ring buffer)
                                  ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │            WORKER THREADS (Stage B & Gated Dispatch)       │
                  │  1. Stage B: VLM Reasoner (Claude Vision / Moondream2)    │
                  │     └─ If False Alarm / Normal ──▶ Dismiss (No noise)     │
                  │     └─ If Confirmed Incident:                             │
                  │  2. Identity: InsightFace (RetinaFace + ArcFace 512-d)    │
                  │  3. Decision Engine: Severity/Confidence Gates + Cooldown │
                  │  4. Save Evidence (Thumbnails, Crops) + SQLite Store      │
                  │  5. Dispatch (Webhook / Logs) + Push WebSocket Incident   │
                  └───────────────────────────────────────────────────────────┘

                  ┌───────────────────────────────────────────────────────────┐
                  │          ATTENDANCE THREAD (Slow Cadence, every 3s)       │
                  │  IdentityService.identify_all(latest_raw_frame)           │
                  │  AttendanceStore.mark_seen(person_id, name) (1 row/day)   │
                  │  StreamHub.push_attendance() ──▶ WebSocket Toast / Stats  │
                  └───────────────────────────────────────────────────────────┘
```

### Multi-Threading Design
1. **Capture Loop Never Blocks**: The camera feed never stutters, drops frames, or lags behind real-time.
2. **Worker Pool (2 Workers)**: Heavy cloud API calls (Claude) or local VLM inferences take 500ms–2s; bounded queues drop redundant triggers if overloaded rather than buffering stale frames.
3. **Dedicated Attendance Loop**: Runs every 3 seconds completely off the detection path. Both workers and attendance share the InsightFace model safely through an internal inference mutex lock (`_infer_lock`).

---

## 3. Technology Stack: Every Tech Used & What It's For

### 1. Vision, Machine Learning & Neural Networks
| Technology | Version / Package | Purpose in HawkEye |
|---|---|---|
| **YOLO11 Pose** (`ultralytics`) | `ultralytics>=8.3.0` | Real-time human keypoint estimation (`yolo11n-pose.pt`). Extracts 17 COCO body keypoints per person to track skeletons. |
| **YOLO11 Object Detection** | `ultralytics>=8.3.0` | Custom fine-tuned weapon detector (`weapons.pt`) trained to detect handguns, rifles, and knives. |
| **PyTorch & Torchvision** | `torch>=2.4` (or `torch>=2.7` cu128 for RTX 50 Blackwell) | Powers custom neural network training and inference on CPU, Apple Silicon (MPS), or NVIDIA CUDA. |
| **Custom ActionNet** | PyTorch (`nn.Module`) | 1D Multi-Scale Conv + BiGRU + Attention/Mean/Max temporal pooling network classifying skeleton sequences into Violence, Fall, or Normal. |
| **InsightFace** | `insightface>=0.7.3` (`buffalo_l`) | State-of-the-art face detection and biometric embeddings. Uses **RetinaFace** for face localization and **ArcFace** for 512-dimensional normalized face embeddings. |
| **ONNX Runtime / ONNX Runtime GPU** | `onnxruntime>=1.17` | High-performance inference engine running the InsightFace models locally. |
| **Anthropic Python SDK** | `anthropic>=0.40` | Cloud VLM reasoning backend (`claude-opus-4-8` / `claude-sonnet-5`). Receives sampled video clip frames and returns structured JSON schemas describing incidents. |
| **Moondream2** | `transformers`, `accelerate`, `einops` | Optional offline, local 2-Billion parameter Vision-Language Model (`vikhyatk/moondream2`) running directly on device memory. |
| **OpenCV** | `opencv-python>=4.9` | Video ingest, camera frame capture, frame scaling, visual BGR annotation (color-coded skeleton limb drawing), and JPEG streaming encoding. |

### 2. Backend & Server Infrastructure
| Technology | Version | Purpose in HawkEye |
|---|---|---|
| **FastAPI** | `fastapi>=0.110` | High-performance async ASGI web framework providing REST endpoints, SPA routing, and WebSockets. |
| **Uvicorn** | `uvicorn[standard]>=0.29` | ASGI web server hosting the FastAPI application. |
| **WebSockets** | Native FastAPI / Starlette | Streams binary MJPEG video frames (~15 FPS) and real-time JSON events (`incident`, `attendance`, `status`) over a single connection. |
| **Pydantic** | `pydantic>=2.6` | Strict typing and serialization schemas for events, incidents, reasoning results, and attendance records. |
| **SQLite3** | Python standard library | Zero-dependency transactional embedded database storing incidents, attendance history, person profiles, and biometric embeddings (`data/hawkeye.db`). |
| **PyYAML** | `pyyaml>=6.0` | Parses the central `configs/default.yaml` configuration file. |
| **Kaggle API** | `kaggle>=1.6` | Automated dataset downloading and preparation for training action and weapon models. |

### 3. Frontend & Mobile Operator App (`app/`)
| Technology | Version | Purpose in HawkEye |
|---|---|---|
| **React Native** | `0.76.5` | Cross-platform UI framework compiling to iOS, Android, and Web from a single codebase. |
| **Expo** | `~52.0.0` | Mobile application SDK and toolchain. |
| **Expo Router** | `~4.0.11` | File-based routing system (`app/(tabs)/live.tsx`, `incidents.tsx`, `attendance.tsx`, `people.tsx`, `drone.tsx`, `settings.tsx`). |
| **React Native Web** | `~0.19.13` | Compiles the Expo app into a single-page web application (`app/dist/`) served directly by FastAPI at `http://localhost:8000`. |
| **TanStack React Query** | `^5.62.7` | Client-side async state caching, auto-refetching, and optimistic updates across all tabs. |
| **Zustand** | `^5.0.2` | Ultra-lightweight global client state management for WebSocket frame streaming, live connection states, and toast queues. |
| **Zod** | `^3.24.1` | TypeScript runtime validation mirroring the Python backend Pydantic models for type safety. |
| **React Native Reanimated** | `~3.16.1` | Hardware-accelerated 60 FPS spring animations, entry transitions, and micro-interactions. |
| **Expo Camera & Image Picker** | `~16.0.x` | Native camera capture and gallery selection used on the People tab for face enrollment. |
| **Expo Secure Store** | `~14.0.0` | Encrypted persistent storage for backend server URLs and operator preferences. |

---

## 4. Deep Dive into System Subsystems

### 4.1 Ingest Layer (`hawkeye/ingest/`)
Abstracted behind `VideoSource`, allowing any video stream without altering downstream detection:
- **Webcams**: Direct hardware indices (`"0"`, `"1"` via V4L2/AVFoundation).
- **IP Cameras & Mobile Feeds**: RTSP/HTTP streams (e.g. `http://phone-ip:8080/video`).
- **Video Files**: Looped video files (e.g. `clip.mp4`) for benchmarking and dataset demos.
- **Drone Feeds (Phase 2)**: `DroneSource` is prepared to wrap `djitellopy` to pipe drone WiFi camera streams.
- **Decoupled Frame Draining**: The ingest thread continuously drains the OpenCV driver buffer; the detection thread grabs the latest frame to prevent latency build-up.

---

### 4.2 Stage A: Real-Time Detection & Kinematics (`hawkeye/detect/`)
Stage A processes incoming frames at ~20 FPS. It fuses four independent detectors into a single `StageAEvent`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE A PIPELINE TICK                           │
│                                                                        │
│  1. PoseDetector (YOLO11n-pose) ──▶ Extracted Person Skeletons         │
│  2. MotionAnomaly               ──▶ Scene Frame Differencing Score     │
│  3. ActionClassifier            ──▶ ActionNet (Violence/Fall/Normal)   │
│  4. WeaponDetector              ──▶ YOLO11 Weapons (Gun/Knife)         │
│  5. Safety Net (Open-vocab)     ──▶ High motion + people = UNKNOWN     │
│                                                                        │
│  FUSED SCORE = max(action_score, weapon_score, motion_score * weight)  │
│  If FUSED SCORE >= trigger_threshold (0.60) & Cooldown expired:        │
│     TRIGGER STAGE B WORKER                                             │
└────────────────────────────────────────────────────────────────────────┘
```

#### The 188-Dimensional Skeleton Feature Representation (`skeleton_features.py`)
To distinguish human activities, HawkEye constructs a specialized 188-dimensional feature vector:
1. **Base Features (112 dimensions)**:
   - `A_KP` (34 dims): Actor A's 17 keypoints $(x, y)$, normalized by torso scale and centered on the hip.
   - `A_KPC` (17 dims): Actor A's keypoint confidence scores (enables occlusion awareness so the network doesn't mistake missing limbs for deformed poses).
   - `A_POSE` (9 dims): Body posture scalars (aspect ratio, verticality, torso angle, limb spread).
   - `B_KP` (34 dims): Nearest interacting person (Actor B) in Actor A's normalized coordinate frame.
   - `B_POSE` (4 dims): Actor B presence and posture indicators.
   - `INTER` (10 dims): A $\leftrightarrow$ B interaction geometry (wrist-to-head reach distances, wrist-to-torso proximity, vertical dominance).
   - `SCENE` (4 dims): Total person count, scene motion energy, crowd proximity.
2. **Temporal Dynamic Features (76 dimensions)**:
   - `A_VEL` & `B_VEL` (68 dims): Per-keypoint velocity vectors over time.
   - `DYN` (8 dims): Hip vertical velocity and acceleration, torso angular rate, kinetic energy, closing speed, stillness metric, and jerk.

#### Solving the Violence vs. Fall Confusion
A classic failure of action detection is confusing violence and falls—both involve rapid limb movements. HawkEye resolves this through three principles:
- **Kinematic Asymmetry**: A fall is a *single body* undergoing *downward ballistic acceleration followed by stillness*. Violence is a *two-body interaction* with *sustained oscillating mutual motion* where wrists close on the other person's head/torso.
- **Actor-B Dropout Data Augmentation**: Prevents the model from taking a shortcut of assuming "two people standing close = fight."
- **Multi-Scale Convolutional Architecture (`ActionNet`)**:
  - Parallel 1D temporal convolutions with kernel sizes 3, 5, and 9 frames see both short ballistic events (falls ~0.5s) and multi-second sustained events (fights).
  - A 2-layer Bidirectional GRU (BiGRU) learns the chronological sequence of movement.
  - Hybrid pooling concatenates Attention pooling (identifies the decisive impact frame), Mean pooling (measures sustained aggression), and Max pooling (captures peak ballistic impact).
  - Temperature-scaled calibration outputs calibrated probabilities rather than overconfident saturated logits.

---

### 4.3 Stage B: Vision-Language Model Reasoning (`hawkeye/reason/`)
When Stage A triggers, a 3-second buffer of sampled frames is evaluated by a pluggable VLM:
- **Cloud Backend (Claude Vision)**:
  - Default: `claude-opus-4-8` (or `claude-sonnet-5`).
  - Strict JSON Schema enforcing structured outputs:
    ```json
    {
      "confirmed": true,
      "description": "A person in a black jacket pushed another individual to the floor near the entrance.",
      "anomaly_type": "violence",
      "severity": "high",
      "recommended_action": "police",
      "confidence": 0.94
    }
    ```
  - Eliminates false alarms: If someone tripped gently or was just tying their shoe, Claude sets `confirmed: false`, terminating the incident silently.
- **Local Offline Backend (Moondream2)**:
  - Uses `vikhyatk/moondream2` via Hugging Face `transformers`.
  - Fits in ~2 GB RAM/VRAM alongside Stage A.
  - Prompts the model to describe abnormalities and maps keywords to incident types.
- **Mock Backend**:
  - Deterministic fallback requiring zero external dependencies or API keys.

---

### 4.4 Biometric Identity & Passive Attendance (`hawkeye/identity/` & `hawkeye/attendance/`)

#### Biometric Engine (`hawkeye/identity/`)
- **Detector**: RetinaFace detects faces even under extreme angles or low resolution.
- **Recognizer**: ArcFace generates an $L_2$-normalized 512-dimensional feature embedding for each face.
- **Matching**: Uses Cosine Similarity against enrolled vectors in `hawkeye.db`. A similarity $\ge 0.42$ denotes a verified identity match.
- **Two Modalities**:
  - `identify(frame)`: Finds the primary subject face in the most recent clip frame and crops it as evidence.
  - `identify_all(frame)`: Detects and matches *every* face visible in the room concurrently for attendance tracking.

#### Continuous Passive Attendance (`hawkeye/attendance/`)
- Runs continuously on a slow background timer (every 3.0 seconds).
- Reuses the InsightFace instance thread-safely via its internal lock.
- Upserts into the SQLite table `attendance`:
  - Fields: `person_id`, `name`, `date`, `first_seen`, `last_seen`, `sightings`.
  - Primary key: `(person_id, date)`.
  - When a member is sighted for the first time on a given day, an event is broadcast over WebSocket, incrementing the live attendance count and displaying an on-screen toast.

---

### 4.5 Gated Decision Engine & Dispatch (`hawkeye/decision/`)
Prevents false or spam dispatches using safety gating:
1. **Confirmation Gate**: Must be verified as `confirmed: true` by Stage B.
2. **Confidence Floor**: Stage B confidence must be $\ge 0.50$.
3. **Severity Floor**: Severity must meet or exceed `dispatch_min_severity` (defaults to `"medium"`).
4. **Action Cooldown**: A per-action-type timer (default: 20 seconds) prevents repeated automated calls for the same event.
5. **Config-Driven Rule Mapping**:
   - `violence`, `robbery`, `theft`, `weapon`, `intrusion` $\rightarrow$ **Police**
   - `fall`, `medical`, `accident` $\rightarrow$ **Ambulance**
   - `fire` $\rightarrow$ **Fire Department**
6. **Dispatchers**: Dispatches via local audit logs and configurable HTTP Webhooks.

---

### 4.6 Server, Streaming & WebSocket Protocol (`server/app.py`)
FastAPI hosts the API, video stream, and frontend at a single address (`http://127.0.0.1:8000`):
- **WebSocket (`/ws`) Protocol**:
  - **Binary Messages**: Raw JPEG-compressed video frames (~15 FPS) rendered with Stage A skeleton bounding boxes.
  - **Text Messages**: JSON events multiplexed over the same socket:
    - `type: "incident"`: Dispatched when an anomaly is confirmed.
    - `type: "attendance"`: Dispatched when a member is marked present.
    - `type: "status"`: Broadcasts FPS, VLM backend, active device, and enrolled member counts.
- **REST Endpoints**:
  - `GET /api/status` & `GET /api/health`: System health and hardware status.
  - `GET /api/incidents`: Fetch paginated incident history.
  - `GET /api/incidents.csv`: Export audit logs as CSV.
  - `POST /api/incidents/{id}/status`: Mark incidents as reviewed or dismissed.
  - `GET /api/attendance?date=YYYY-MM-DD`: Retrieve daily attendance logs.
  - `GET /api/people` & `POST /api/people`: Enrolled roster list and multipart photo enrollment.
  - `DELETE /api/people/{id}`: Unenroll a member and purge biometric embeddings.
  - `GET /api/evidence/{path}`: Fetch incident thumbnails, full-clip videos, and face crops.
  - `GET /`: Serves the static compiled Expo SPA (`app/dist/index.html`).

---

### 4.7 Frontend Companion Console (`app/`)
Built with React Native and Expo, styled in a dark "command center" design system (`#0B0D10` base with `#3EA6FF` electric blue accents and semantic status indicators):

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HAWKEYE OPERATOR SUITE                          │
├───────────────┬────────────────────────────────────────────────────────┤
│ Live          │ Full-screen live video feed, real-time FPS/status      │
│               │ pills, and instantaneous incident alerts.              │
├───────────────┼────────────────────────────────────────────────────────┤
│ Incidents     │ Incident feed with severity filtering (Low/Med/High/   │
│               │ Critical), VLM descriptions, face matches, and export. │
├───────────────┼────────────────────────────────────────────────────────┤
│ Attendance    │ Date stepper, present count statistics, and member     │
│               │ check-in timelines with first/last seen timestamps.    │
├───────────────┼────────────────────────────────────────────────────────┤
│ People        │ Face database roster management, multi-photo camera    │
│               │ enrollment, role tagging, and profile deletion.        │
├───────────────┼────────────────────────────────────────────────────────┤
│ Drone (Prev.) │ Contract-first preview UI of autonomous patrol fleet   │
│               │ telemetry, waypoint routes, and locked manual flight.  │
├───────────────┼────────────────────────────────────────────────────────┤
│ Settings      │ Server connection management, health probes, and VLM.  │
└───────────────┴────────────────────────────────────────────────────────┘
```

---

## 5. Model Training, Datasets & Benchmarking (`scripts/`, `eval/`)

HawkEye includes a complete self-contained pipeline for training and evaluating models locally (targeted at Apple Silicon MPS and NVIDIA CUDA):

### Training Pipelines
1. **`scripts/setup_datasets.py`**:
   - Integrates with the Kaggle API to automatically download and structure:
     - Violence: **RWF-2000** (Fight split) and **Hockey Fights**.
     - Normal / Daily Living: **RWF-2000** (NonFight split) and **UCF101** daily living activities (walking, mopping, typing) as hard negatives.
     - Falls: Kaggle fall detection mirrors, **UR Fall Detection**, and **Le2i**.
2. **`scripts/train_action.py`**:
   - Pre-computes and caches skeleton features to disk (`action_features.npz`).
   - Implements **grouped splitting by source clip** (`group_split`) so overlapping temporal windows from the same video never leak between train, validation, and test splits.
   - Trains `ActionNet`, fits temperature calibration, determines per-class operating thresholds at 90% target precision, and outputs `models/action_gru.pt`.
3. **`scripts/train_weapon.py`**:
   - Fine-tunes YOLO11 on firearm and knife datasets, converting Pascal-VOC XML annotations to YOLO format automatically.
4. **`eval/evaluate.py`**:
   - Benchmarks live Stage A FPS and pipeline latency.
   - Generates full Confusion Matrices, Precision, Recall, and Macro-F1 scores.
   - Measures the critical cross-confusion metrics: `violence → fall` and `fall → violence`.
   - Directly compares trained models against the heuristic baseline (`compare` mode).

---

## 6. Ethics, Privacy & Legal Framework (DPDP Act, 2023)

Biometric surveillance and automated dispatch demand strict privacy standards:
- **Consented Local Identity Database Only**: HawkEye enforces a local, opt-in database of enrolled staff or family members.
- **Strictly No Aadhaar or Government ID Scraping**: No public Aadhaar face-matching API exists, and unconsented public matching violates India’s *Digital Personal Data Protection Act, 2023 (DPDP)* and the *Puttaswamy* privacy judgment.
- **Evidence Minimization**: Video clips and face crops are saved only when an anomaly is verified and confirmed.
- **Human-in-the-Loop Safeguards**: Dispatches are gated by confidence and cooldowns; operators can review and dismiss incidents directly from the dashboard.
- **Configurable Data Retention**: Automated daily retention cycles (`_retention_loop`) purge incidents and raw media older than the configured threshold (e.g. 30 days) to prevent perpetual storage.

---

## 7. Project Directory Structure Map

```
HawkEye/
├── README.md                 # Primary system guide and quickstart
├── HAWKEYE_DEEP_DIVE.md      # This comprehensive technical reference document
├── configs/
│   └── default.yaml          # Master configuration (thresholds, models, rules)
├── hawkeye/
│   ├── run.py                # Command-line entrypoint
│   ├── pipeline.py           # Core 3-thread orchestrator (capture, worker, attendance)
│   ├── hub.py                # StreamHub: thread-safe ring buffer and WebSocket bridge
│   ├── config.py             # Pydantic configuration loader
│   ├── detect/               # Stage A fast detectors
│   │   ├── stage_a.py        # Stage A coordinator & visual BGR annotator
│   │   ├── pose.py           # YOLO11 keypoint detector wrapper
│   │   ├── action_model.py   # ActionNet architecture (1D CNN + BiGRU + pooling)
│   │   ├── action_classifier.py # Live inference wrapper with hysteresis
│   │   ├── skeleton_features.py # 188-D feature extraction & shared standardizer
│   │   ├── weapons.py        # Fine-tuned YOLO weapon detection
│   │   ├── scene_anomaly.py  # Background motion anomaly calculator
│   │   └── heuristics.py     # Zero-training geometric fallback rules
│   ├── reason/               # Stage B VLM reasoners
│   │   ├── base.py           # Reasoner interface & frame sampling
│   │   ├── claude_reasoner.py# Cloud Claude vision backend (structured outputs)
│   │   ├── moondream_reasoner.py # Local 2B VLM offline backend
│   │   └── mock_reasoner.py  # Zero-dependency testing reasoner
│   ├── identity/             # Biometrics & face recognition
│   │   ├── recognizer.py     # InsightFace (RetinaFace + ArcFace) wrapper
│   │   ├── service.py        # Enrollment & multi-face matching service
│   │   └── store.py          # SQLite embedding store
│   ├── attendance/           # Passive attendance subsystem
│   │   └── store.py          # SQLite attendance logging (1 record/person/day)
│   ├── decision/             # Gated decision & dispatch
│   │   ├── engine.py         # Severity/confidence gates & cooldown enforcement
│   │   └── dispatch.py       # Log and Webhook dispatch handlers
│   ├── events/               # Schemas & persistence
│   │   ├── schema.py         # Core Pydantic event models (Incident, StageAEvent)
│   │   ├── store.py          # SQLite incident audit trail
│   │   └── evidence.py       # Media evidence file storage & thumbnailing
│   ├── ingest/               # Video input abstractions (webcam, IP, file, drone)
│   └── train/                # Device profiling (CUDA, MPS, CPU)
├── server/
│   └── app.py                # FastAPI app (REST API, WebSocket, static SPA server)
├── app/                      # React Native / Expo operator companion console
│   ├── app/                  # Expo Router navigation routes
│   │   ├── (tabs)/           # live, incidents, attendance, people, drone
│   │   ├── incident/[id].tsx # Incident detail view
│   │   └── settings.tsx      # Server configuration & health probes
│   └── src/                  # Theme tokens, Zustand stores, WebSocket client
├── scripts/                  # Training & maintenance utilities
│   ├── train_action.py       # Train ActionNet classifier
│   ├── train_weapon.py       # Fine-tune YOLO weapon detector
│   ├── setup_datasets.py     # Automated Kaggle dataset downloader
│   ├── enroll_person.py      # CLI face enrollment script
│   └── train_all.sh          # One-command full training pipeline
├── eval/
│   └── evaluate.py           # FPS benchmarks, confusion matrices, and model comparison
└── tests/                    # Pipeline self-checks
    ├── test_action_pipeline.py # Verifies feature consistency and absence of data leakage
    └── test_pipeline_shutdown.py # Verifies thread joining and clean database closing
```

---

## 8. Summary of Unique Selling Points (USPs)

1. **Explainable AI, Not Black-Box Bounding Boxes**: Instead of a generic bounding box with "Anomaly: 0.82", HawkEye outputs an English incident summary: *"A person in a black jacket pushed another individual to the floor near the entrance."*
2. **Open-Vocabulary Anomaly Recognition**: Because Stage B is an LLM/VLM, it flags novel situations that were never in the training set (e.g., someone falling unconscious, smoke drifting, or an abandoned suspicious package).
3. **Severe False-Alarm Reduction**: The two-stage cascade ensures that cheap motion triggers are vetted by the semantic VLM before any alarm sounds or dispatch occurs.
4. **Unified Dual-Utility Architecture**: Enrolling a face once powers both perpetrator identification on emergencies and passive attendance tracking on normal days.
5. **Zero-Lag Real-Time Video**: Decoupled multi-threading ensures that computationally heavy reasoning runs completely out of band, keeping the live camera feed running at a smooth 15–20 FPS.
