"""FastAPI application: serves the dashboard, streams frames + incidents over a
WebSocket, and exposes REST endpoints for incidents, people, and evidence.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from hawkeye.pipeline import Pipeline
from hawkeye.runtime import get_config, get_hub, setup_logging

log = logging.getLogger("hawkeye.server")

# The UI is the Expo app's web build (`cd app && npm run build:web`), served by
# this same process: one address for the dashboard, the REST API and the
# WebSocket, instead of a separate dev server on another port. There is no
# second hand-written dashboard to drift from it.
APP_DIST = Path(__file__).resolve().parent.parent / "app" / "dist"

_BUILD_MISSING_HTML = """<!doctype html>
<meta charset="utf-8"><title>HawkEye — UI not built</title>
<style>body{font:17px/1.5 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;
background:#1d1d1f;color:#f5f5f7;margin:0;display:grid;place-items:center;height:100vh}
main{max-width:34rem;padding:2rem}code{background:#272729;padding:.15em .4em;
border-radius:5px;font-size:.9em}a{color:#2997ff}</style>
<main><h1>UI not built yet</h1>
<p>The API and WebSocket are running, but the operator app hasn't been built.
Build it once, then reload:</p>
<p><code>cd app &amp;&amp; npm install &amp;&amp; npm run build:web</code></p>
<p>Expected output at <code>app/dist/index.html</code>.
API health: <a href="/api/health">/api/health</a></p></main>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    cfg = get_config()
    hub = get_hub()
    pipeline = Pipeline(cfg, hub)
    app.state.cfg = cfg
    app.state.hub = hub
    app.state.pipeline = pipeline
    pipeline.start()   # spawns its own threads, returns immediately
    try:
        yield
    finally:
        pipeline.stop()


app = FastAPI(title="HawkEye", lifespan=lifespan)

# Dev workflow (`cd app && npm start`) and a phone on the LAN both hit this API
# from a different origin than the served app; there's no auth/cookies to
# protect here (Phase 1, trusted LAN — see README roadmap), so open CORS is
# just unblocking a documented workflow, not a new exposure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    # No caching on the HTML shell: it is actively developed, and a stale cached
    # copy after a rebuild produces exactly the "still broken" confusion a hard
    # refresh would otherwise silently paper over. Hashed assets under /_expo/
    # are immutable and cached normally by StaticFiles.
    return _index_response()


@app.get("/api/status")
async def status():
    return app.state.hub.get_status()


@app.get("/api/health")
async def health():
    """Liveness/readiness probe: reports whether the core components actually
    came up (Stage A weights loaded, DB + evidence dir writable). Returns 503
    when a critical component is down so a container orchestrator can react."""
    import os

    pipeline = app.state.pipeline
    cfg = app.state.cfg

    stage_a_loaded = getattr(pipeline.stage_a.pose, "_model", None) is not None
    evidence_dir = Path(cfg.storage.evidence_dir)
    evidence_ok = evidence_dir.is_dir() and os.access(evidence_dir, os.W_OK)
    db_parent = Path(cfg.storage.db_path).parent
    db_ok = db_parent.is_dir() and os.access(db_parent, os.W_OK)

    components = {
        "running": pipeline._running,
        "stage_a_loaded": stage_a_loaded,
        "evidence_writable": evidence_ok,
        "db_writable": db_ok,
        "identity": pipeline.identity.available,
        "reasoner": pipeline.reasoner.name,
    }
    # Identity/reasoner degrade gracefully by design, so they don't fail health.
    healthy = stage_a_loaded and evidence_ok and db_ok and pipeline._running
    body = {"status": "ok" if healthy else "degraded", "components": components}
    return JSONResponse(body, status_code=200 if healthy else 503)


@app.get("/api/incidents")
async def incidents(limit: int = 50):
    items = app.state.pipeline.store.recent(limit)
    out = []
    for inc in items:
        d = json.loads(inc.model_dump_json())
        d["headline"] = inc.headline()
        out.append(d)
    return out


@app.post("/api/incidents/{incident_id}/status")
async def set_incident_status(incident_id: str, status: str = Form(...)):
    from hawkeye.events.schema import IncidentStatus

    try:
        IncidentStatus(status)
    except ValueError:
        raise HTTPException(400, f"Invalid status '{status}'. "
                                 f"Must be one of: {[s.value for s in IncidentStatus]}")
    ok = app.state.pipeline.store.set_status(incident_id, status)
    if not ok:
        raise HTTPException(404, "Incident not found")
    return {"ok": True}


@app.get("/api/incidents.csv")
async def incidents_csv(limit: int = 1000):
    import csv
    import io

    items = app.state.pipeline.store.recent(limit)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "ts", "anomaly_type", "status", "severity", "description",
                     "matched_person", "action", "dispatched"])
    for inc in items:
        writer.writerow([
            inc.id, inc.ts, inc.anomaly_type.value, inc.status.value,
            inc.reasoning.severity.value if inc.reasoning else "",
            inc.reasoning.description if inc.reasoning else "",
            inc.face_match.name if inc.face_match and inc.face_match.matched else "",
            inc.action.action.value if inc.action else "",
            inc.action.dispatched if inc.action else False,
        ])
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=hawkeye_incidents.csv"})


@app.get("/api/attendance")
async def attendance(date: str | None = None):
    records = app.state.pipeline.attendance.on_date(date)
    return [json.loads(r.model_dump_json()) for r in records]


@app.get("/api/people")
async def people():
    return [json.loads(p.model_dump_json()) for p in app.state.pipeline.identity.list_people()]


@app.post("/api/people")
async def enroll(name: str = Form(...), meta: str = Form("{}"),
                 files: List[UploadFile] = File(...)):
    identity = app.state.pipeline.identity
    if not identity.available:
        raise HTTPException(400, "Identity module unavailable (not installed, still "
                                 "loading, or failed to load — check server logs).")

    raw_files = [await f.read() for f in files]   # async I/O — fine on the event loop

    def _decode_and_enroll():
        # CPU-bound (image decode + face embedding): run off the event loop so
        # the live video feed / other requests aren't frozen while this runs,
        # and so it queues safely behind any concurrent pipeline trigger via
        # the recognizer's internal lock instead of racing it.
        import cv2

        images = []
        for data in raw_files:
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        if not images:
            raise ValueError("No valid images uploaded.")
        try:
            meta_dict = json.loads(meta) if meta else {}
        except json.JSONDecodeError:
            meta_dict = {}
        return identity.enroll_person(name, meta_dict, images)

    try:
        person = await run_in_threadpool(_decode_and_enroll)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))

    app.state.hub.set_status(enrolled=identity.enrolled)
    return json.loads(person.model_dump_json())


@app.delete("/api/people/{person_id}")
async def delete_person(person_id: str):
    app.state.pipeline.identity.delete_person(person_id)
    app.state.hub.set_status(enrolled=app.state.pipeline.identity.enrolled)
    return {"ok": True}


@app.get("/api/evidence/{path:path}")
async def evidence(path: str):
    root = Path(app.state.cfg.storage.evidence_dir).resolve()
    target = (root / path).resolve()
    # is_relative_to (vs a string startswith) correctly rejects sibling dirs
    # like ".../evidence-other/..." and, because both paths are resolved,
    # symlinks that escape the evidence root.
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(target)


def _index_response() -> Response:
    """The app shell, or build instructions if the UI hasn't been built."""
    index = APP_DIST / "index.html"
    if not index.is_file():
        return HTMLResponse(_BUILD_MISSING_HTML, status_code=503,
                            headers={"Cache-Control": "no-store"})
    return FileResponse(index, headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub = app.state.hub
    last_fid = -1
    # Replay recent incidents on connect so a fresh page isn't empty.
    version, _ = hub.incidents_since(0)
    att_version, _ = hub.attendance_since(0)   # only toast NEW marks (no replay flood)
    status_version, status = hub.status_since(-1)   # -1: always send the initial status
    for inc in hub.recent_incidents(20)[::-1]:
        await websocket.send_text(json.dumps({"type": "incident", "data": inc}))
    await websocket.send_text(json.dumps({"type": "status", "data": status}))
    try:
        while True:
            fid, jpeg = hub.get_frame()
            if jpeg is not None and fid != last_fid:
                last_fid = fid
                await websocket.send_bytes(jpeg)
            version, new = hub.incidents_since(version)
            for inc in new:
                await websocket.send_text(json.dumps({"type": "incident", "data": inc}))
            att_version, att_new = hub.attendance_since(att_version)
            for rec in att_new:
                await websocket.send_text(json.dumps({"type": "attendance", "data": rec}))
            status_version, status = hub.status_since(status_version)
            if status is not None:
                await websocket.send_text(json.dumps({"type": "status", "data": status}))
            await asyncio.sleep(1 / 15)
    except (WebSocketDisconnect, RuntimeError):
        return


# ── Static UI ────────────────────────────────────────────────────────────────
# Mounted LAST so nothing above can be shadowed: /api/*, /ws and / are already
# bound by the time these catch-alls are registered.
if APP_DIST.is_dir():
    for sub in ("_expo", "assets"):
        d = APP_DIST / sub
        if d.is_dir():
            app.mount(f"/{sub}", StaticFiles(directory=d), name=f"static-{sub}")


@app.get("/{path:path}")
async def spa_fallback(path: str):
    """Serve the app shell for client-side routes (expo-router is a SPA).

    A deep link like /incident/abc123 or a refresh on /settings has no file on
    disk; the router resolves it in the browser. Real files under the build root
    are still served directly, and unknown /api/* paths must 404 rather than
    return HTML — a JSON client receiving an HTML shell is a far more confusing
    failure than an honest 404.
    """
    if path.startswith("api/") or path == "ws":
        raise HTTPException(404, "Not found")
    candidate = (APP_DIST / path).resolve()
    if APP_DIST.is_dir() and candidate.is_relative_to(APP_DIST.resolve()) and candidate.is_file():
        return FileResponse(candidate)
    return _index_response()
