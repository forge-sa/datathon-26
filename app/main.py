"""FastAPI app: jury API (/api/v1, X-API-Key) and a simple test UI (/, /ui/api, no key)."""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import API_KEY, STATIC_DIR, load_regulations, load_settings, load_taxonomy
from .jobs import JobStore, public_view
from .vlm import ollama_ready

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class _NoPolling(logging.Filter):
    """Drop the access-log lines of status polling (UI every 1.5 s, jury clients) and health checks;
    jobs.py logs one line when a job is received and one when it is done."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        polls = ('"GET /ui/api/jobs/', '"GET /api/v1/jobs/', '"GET /health', '"GET /api/v1/health')
        return not (any(p in msg for p in polls) and '" 200' in msg)


logging.getLogger("uvicorn.access").addFilter(_NoPolling())
logging.getLogger("httpx").setLevel(logging.WARNING)   # one line per VLM call is too much
if os.environ.get("LOG_FILE"):   # set by run.ps1: the console log is also written to logs/server_*.log
    _fh = logging.FileHandler(os.environ["LOG_FILE"], encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _fh.addFilter(_NoPolling())
    for _name in ("", "uvicorn", "uvicorn.access"):
        logging.getLogger(_name).addHandler(_fh)

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


@api.get("/health")
def api_health():
    return health()


# -- UI API (local testing; protect or remove before exposing publicly) ------------------
ui = APIRouter(prefix="/ui/api")


@ui.post("/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await _submit(file)


@ui.get("/jobs/{job_id}")
def ui_job(job_id: str):
    return public_view(_job(job_id))


@ui.get("/jobs")
def ui_jobs():
    return store.recent()


@ui.get("/catalog")
def ui_catalog():
    """Violation types the service detects, with their regulation clauses (UI reference tab)."""
    regs = load_regulations()
    types = []
    for t in load_taxonomy().values():
        clauses = []
        for cid in t.get("clauses", []):
            cl = regs["clauses"].get(cid)
            if cl:
                src = regs["sources"].get(cl["source"], {})
                clauses.append({"act": src.get("title", cl["source"]), "point": cl.get("point", ""),
                                "quote": cl.get("quote", ""), "url": src.get("url", "")})
        types.append({"id": t["id"], "name": t["name_ru"], "summary": t.get("summary_ru", ""),
                      "why": t.get("why_ru", ""), "clauses": clauses})
    sources = [{"title": s["title"], "url": s.get("url", ""), "version": s.get("version", "")}
               for s in regs["sources"].values()]
    return {"types": types, "sources": sources}


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
