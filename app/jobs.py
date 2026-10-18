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

from .config import JOBS_DIR, load_settings
from .pipeline import analyze

log = logging.getLogger("jobs")


class JobStore:
    def __init__(self, root: Path = JOBS_DIR, workers: int = 6):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        threading.Thread(target=self._watchdog, daemon=True, name="job-watchdog").start()

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
        log.info("job %s: received %s (%.1f MB)", job_id, filename, len(data) / 1048576)
        self.pool.submit(self._run, job_id, path)
        return job_id

    def _run(self, job_id: str, path: Path) -> None:
        job = self.jobs[job_id]

        def progress(msg: str):
            job["progress"] = msg

        def partial(res: dict):
            job["partial"] = res

        try:
            res = analyze(path, started_at=job["submitted_at"], active_jobs=self.active(), progress=progress,
                          on_partial=partial)
            debug = res.pop("debug")
            (self.root / job_id / "debug.json").write_text(
                json.dumps(debug, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            with self.lock:
                if job["status"] == "done":     # the watchdog already answered; keep that answer stable
                    job["late_result"] = res
                else:
                    job.update(res, status="done", progress="Готово",
                               processing_sec=round(time.time() - job["submitted_at"], 1))
            log.info("job %s: done in %.0f s, tier %s, verdict %s%s", job_id, time.time() - job["submitted_at"],
                     debug.get("tier"), res["verdict"],
                     "".join(f"; {v['type_id']}@{v['timecode_sec']}s" for v in res["violations"]))
            for w in debug.get("warnings", []):
                log.warning("job %s: %s", job_id, w)
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job_id)
            with self.lock:
                if job["status"] != "done":
                    job.update(status="failed", error=str(e), progress="Ошибка",
                               traceback=traceback.format_exc(limit=5))
        job.pop("partial", None)
        self._save(job_id)

    def _watchdog(self) -> None:
        """The jury counts a video as unprocessed after 5 minutes. If a job is still running near that
        limit (GPU contention, a hung VLM call), answer with the provisional result of the main detector."""
        while True:
            time.sleep(2.0)
            try:
                limit = float(load_settings()["budget"].get("hard_limit_sec", 280))
            except Exception:  # noqa: BLE001
                limit = 280.0
            now = time.time()
            with self.lock:
                late = [j for j in self.jobs.values()
                        if j["status"] == "processing" and now - j["submitted_at"] > limit and j.get("partial")]
                for job in late:
                    job.update(job["partial"], status="done", progress="Готово (сокращённый анализ)",
                               processing_sec=round(now - job["submitted_at"], 1), partial_result=True)
                    log.warning("job %s: time limit, returned the provisional result", job["job_id"])
            for job in late:
                self._save(job["job_id"])

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
        job = {k: v for k, v in self.jobs[job_id].items() if k != "partial"}
        (self.root / job_id / "result.json").write_text(
            json.dumps(job, ensure_ascii=False, default=str), encoding="utf-8")

    def recent(self, n: int = 20) -> list[dict]:
        """Latest jobs (this process and earlier runs on disk), newest first, without frames."""
        rows = []
        for d in self.root.iterdir() if self.root.exists() else []:
            job = self.get(d.name) if (d / "result.json").exists() else None
            if job:
                rows.append({"job_id": job["job_id"], "filename": job.get("filename", ""),
                             "status": job["status"], "verdict": job.get("verdict"),
                             "n_violations": len(job.get("violations") or []),
                             "submitted_at": job.get("submitted_at", 0)})
        rows.sort(key=lambda r: -r["submitted_at"])
        return rows[:n]


def public_view(job: dict) -> dict:
    """Exact API schema (plus a few harmless extra fields)."""
    out = {"job_id": job["job_id"], "status": job["status"]}
    if job["status"] == "done":
        out["verdict"] = job["verdict"]
        out["violations"] = job["violations"]
        out["processing_sec"] = job.get("processing_sec")
        if job.get("partial_result"):
            out["partial_result"] = True
    elif job["status"] == "failed":
        out["error"] = job.get("error", "")
    else:
        out["progress"] = job.get("progress", "")
    return out
