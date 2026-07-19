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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from hawkeye.pipeline import Pipeline
from hawkeye.runtime import get_config, get_hub, setup_logging

log = logging.getLogger("hawkeye.server")
STATIC_DIR = Path(__file__).resolve().parent / "static"


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


@app.get("/")
async def index():
    # No caching: this dashboard is actively developed, and a stale cached
    # copy after a CSS/JS fix produces exactly the "still broken" confusion
    # a hard refresh would otherwise silently paper over.
    return FileResponse(STATIC_DIR / "index.html",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/status")
async def status():
    return app.state.hub.get_status()


@app.get("/api/incidents")
async def incidents(limit: int = 50):
    items = app.state.pipeline.store.recent(limit)
    out = []
    for inc in items:
        d = json.loads(inc.model_dump_json())
        d["headline"] = inc.headline()
        out.append(d)
    return out


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
    if not str(target).startswith(str(root)) or not target.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(target)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub = app.state.hub
    last_fid = -1
    # Replay recent incidents on connect so a fresh page isn't empty.
    version, recent = hub.incidents_since(0)
    for inc in hub.recent_incidents(20)[::-1]:
        await websocket.send_text(json.dumps({"type": "incident", "data": inc}))
    try:
        while True:
            fid, jpeg = hub.get_frame()
            if jpeg is not None and fid != last_fid:
                last_fid = fid
                await websocket.send_bytes(jpeg)
            version, new = hub.incidents_since(version)
            for inc in new:
                await websocket.send_text(json.dumps({"type": "incident", "data": inc}))
            await websocket.send_text(json.dumps({"type": "status", "data": hub.get_status()}))
            await asyncio.sleep(1 / 15)
    except (WebSocketDisconnect, RuntimeError):
        return
