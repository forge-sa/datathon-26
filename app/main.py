"""FastAPI app: jury API (/api/v1, X-API-Key) and a simple test UI (/, /ui/api, no key)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import API_KEY, STATIC_DIR, load_settings
from .jobs import JobStore, public_view
from .vlm import ollama_ready

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_UPLOAD = 300 * 1024 * 1024
ALLOWED = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

app = FastAPI(title="ForteBank cash-discipline video audit")
store = JobStore()


def require_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


async def _submit(file: UploadFile) -> dict:
    name = file.filename or "video.mov"
    if not any(name.lower().endswith(ext) for ext in ALLOWED):
        raise HTTPException(status_code=400, detail="expected a .mov video in field 'file'")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="file too large")
    return {"job_id": store.submit(name, data), "status": "processing"}


def _job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# -- jury API --------------------------------------------------------------------------
api = APIRouter(prefix="/api/v1", dependencies=[Depends(require_key)])


@api.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    return await _submit(file)


@api.get("/jobs/{job_id}")
def get_job(job_id: str):
    return public_view(_job(job_id))


# -- UI API (local testing; protect or remove before exposing publicly) ------------------
ui = APIRouter(prefix="/ui/api")


@ui.post("/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await _submit(file)


@ui.get("/jobs/{job_id}")
def ui_job(job_id: str):
    return public_view(_job(job_id))


@ui.get("/jobs/{job_id}/debug")
def ui_debug(job_id: str):
    d = store.debug(job_id)
    if d is None:
        raise HTTPException(status_code=404, detail="no debug data yet")
    return d


@ui.get("/jobs/{job_id}/video")
def ui_video(job_id: str):
    p = store.video_path(job_id)
    if p is None:
        raise HTTPException(status_code=404)
    return FileResponse(p, media_type="video/quicktime" if p.suffix == ".mov" else "video/mp4")


@app.get("/health")
def health():
    s = load_settings()
    ok, msg = ollama_ready(s["vlm"]["url"], s["vlm"]["model"])
    return {"ok": ok, "vlm": msg, "model": s["vlm"]["model"], "active_jobs": store.active()}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.include_router(api)
app.include_router(ui)
