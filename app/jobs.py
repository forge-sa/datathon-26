"""In-process job queue. One uvicorn process (Windows has no fork); jobs run in threads,
results are mirrored to jobs/<id>/result.json so they survive a restart."""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import JOBS_DIR
from .pipeline import analyze

log = logging.getLogger("jobs")


class JobStore:
    def __init__(self, root: Path = JOBS_DIR, workers: int = 6):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")

    def active(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["status"] == "processing")

    def submit(self, filename: str, data: bytes) -> str:
        job_id = uuid.uuid4().hex[:8]
        d = self.root / job_id
        d.mkdir(parents=True)
        suffix = Path(filename).suffix.lower() or ".mov"
        path = d / f"input{suffix}"
        path.write_bytes(data)
        job = {"job_id": job_id, "status": "processing", "filename": filename, "submitted_at": time.time(),
               "progress": "В очереди"}
        with self.lock:
            self.jobs[job_id] = job
        self._save(job_id)
        self.pool.submit(self._run, job_id, path)
        return job_id

    def _run(self, job_id: str, path: Path) -> None:
        job = self.jobs[job_id]

        def progress(msg: str):
            job["progress"] = msg

        try:
            res = analyze(path, started_at=job["submitted_at"], active_jobs=self.active(), progress=progress)
            debug = res.pop("debug")
            (self.root / job_id / "debug.json").write_text(
                json.dumps(debug, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            job.update(res, status="done", progress="Готово",
                       processing_sec=round(time.time() - job["submitted_at"], 1))
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            job.update(status="failed", error=str(e), progress="Ошибка",
                       traceback=traceback.format_exc(limit=5))
        self._save(job_id)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            p = self.root / job_id / "result.json"
            if p.exists():
                job = json.loads(p.read_text(encoding="utf-8"))
                if job.get("status") == "processing":   # server restarted mid-job
                    job["status"] = "failed"
                    job["error"] = "server restarted during processing"
        return job

    def debug(self, job_id: str) -> dict | None:
        p = self.root / job_id / "debug.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def video_path(self, job_id: str) -> Path | None:
        d = self.root / job_id
        files = list(d.glob("input.*")) if d.exists() else []
        return files[0] if files else None

    def _save(self, job_id: str) -> None:
        job = self.jobs[job_id]
        (self.root / job_id / "result.json").write_text(
            json.dumps(job, ensure_ascii=False, default=str), encoding="utf-8")


def public_view(job: dict) -> dict:
    """Exact API schema (plus a few harmless extra fields)."""
    out = {"job_id": job["job_id"], "status": job["status"]}
    if job["status"] == "done":
        out["verdict"] = job["verdict"]
        out["violations"] = job["violations"]
        out["processing_sec"] = job.get("processing_sec")
    elif job["status"] == "failed":
        out["error"] = job.get("error", "")
    else:
        out["progress"] = job.get("progress", "")
    return out
