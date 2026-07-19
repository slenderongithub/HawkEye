# HawkEye — Architecture

## Data flow

```
VideoSource ──frames──▶ Pipeline (capture thread)
                         │  Stage A: pose + heuristics + motion + weapons → StageAEvent
                         │  annotate → StreamHub.set_frame → WebSocket → dashboard
                         │  on trigger: snapshot clip buffer → trigger queue
                         ▼
                       Pipeline (worker thread)
                         │  Stage B reason(event, frames) → Reasoning
                         │  if confirmed:
                         │     Identity.identify(frame) → FaceMatch (+ crop)
                         │     Decision.decide(reasoning, face) → ActionRecord (gated)
                         │     Evidence.save(...) ; IncidentStore.save(...)
                         │     StreamHub.push_incident → WebSocket → dashboard
```

A **third thread** runs attendance on a slow timer, fully independent of anomaly
triggers:

```
                       Pipeline (attendance thread, every ~3s)
                         │  Identity.identify_all(latest frame) → [FaceMatch, ...]
                         │  AttendanceStore.mark_seen(person) (one row/person/day)
                         │  StreamHub.push_attendance → WebSocket toast + present count
```

Three threads keep the video smooth: the **capture** loop never blocks on the
heavy VLM; incident triggers are handed to a **worker** via a bounded queue
(drops if busy, so real-time is never sacrificed); and **attendance** scans at a
low cadence off the hot path. All identity calls (worker + attendance) share the
recognizer safely via its internal inference lock.

## Modules

| Package | Responsibility | Key types |
|---|---|---|
| `hawkeye/ingest` | pluggable video sources | `VideoSource`, `open_source`, `DroneSource` (P2) |
| `hawkeye/detect` | Stage A real-time detection | `StageA`, `PoseDetector`, `heuristics`, `MotionAnomaly` |
| `hawkeye/reason` | Stage B reasoning (adapter) | `Reasoner`, `ClaudeReasoner`, `MoondreamReasoner`, `MockReasoner` |
| `hawkeye/identity` | face enroll + match (perpetrator ID + attendance) | `IdentityService` (`identify` / `identify_all`), `IdentityRecognizer`, `EmbeddingStore` |
| `hawkeye/attendance` | continuous presence logging | `AttendanceStore` |
| `hawkeye/decision` | gated dispatch | `DecisionEngine`, dispatchers |
| `hawkeye/events` | schema, persistence, evidence | `Incident`, `IncidentStore`, `EvidenceManager`, `AttendanceRecord` |
| `hawkeye/pipeline.py` | orchestration | `Pipeline` |
| `hawkeye/hub.py` | thread↔async bridge | `StreamHub` |
| `server/` | dashboard + API | FastAPI app |

## Extension points

- **Trained action model:** `scripts/train_action.py` produces a temporal
  classifier from YOLO-pose sequences. Add a trained candidate source alongside
  `heuristics.analyze()` in `hawkeye/detect/stage_a.py` — same
  `List[(AnomalyType, score, reason)]` contract.
- **Scene anomaly:** swap `MotionAnomaly` for a weakly-supervised MIL scorer
  (UCF-Crime) behind the same `.update(frame) -> float` interface.
- **Phase 2 drone:** implement `DroneSource.open()/read()` (djitellopy) — nothing
  downstream changes. Autonomy code lives in `drone/` + `sim/`.
- **New reasoner backend:** implement `Reasoner.reason()` and register it in
  `hawkeye/reason/factory.py`.

## Design choices

- **Fast/slow cascade** — real-time coverage on modest hardware; VLM only on
  suspicion (efficiency + false-alarm suppression).
- **Camera decoupling** — the intelligence is independent of the frame source,
  which makes the drone a plug-in and the model the reusable core.
- **Graceful degradation** — optional deps (insightface, anthropic, transformers)
  are lazy-imported; the pipeline runs without them.
