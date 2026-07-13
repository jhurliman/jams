"""In-memory registry for async analysis jobs.

jams is a single-process service, so a process-local dict is the whole story: a job is
created by ``POST /v1/analyze`` with ``async=true``, mutated by the worker thread as
stages start/finish (stages may run concurrently — see analyze_track), and read by
``GET /v1/jobs/{id}``. Finished jobs are kept for ``ttl_seconds`` so late pollers still
see the result, then purged lazily on the next registry access.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

__all__ = ["Job", "JobRegistry", "get_registry"]

_TTL_SECONDS = 3600.0


@dataclass
class Job:
    id: str
    status: str = "running"  # running | done | error
    stages_running: list[str] = field(default_factory=list)
    stages_done: list[str] = field(default_factory=list)
    # stage -> [started_at, finished_at|None]; source for per-stage timing readouts
    stage_times: dict[str, list[float | None]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None
    error_stage: str | None = None

    def to_public(self) -> dict:
        out: dict = {
            "job_id": self.id,
            "status": self.status,
            "stages_running": list(self.stages_running),
            "stages_done": list(self.stages_done),
            "started_at": self.started_at,
        }
        if self.status == "done":
            out["result"] = self.result
        if self.status == "error":
            out["error"] = self.error
            out["error_stage"] = self.error_stage
        return out


class JobRegistry:
    def __init__(self, ttl_seconds: float = _TTL_SECONDS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def _purge_locked(self) -> None:
        now = time.time()
        dead = [
            jid for jid, j in self._jobs.items()
            if j.finished_at is not None and now - j.finished_at > self._ttl
        ]
        for jid in dead:
            del self._jobs[jid]

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._purge_locked()
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._purge_locked()
            return self._jobs.get(job_id)

    def start_stage(self, job: Job, stage: str) -> None:
        with self._lock:
            if stage not in job.stages_running:
                job.stages_running.append(stage)
            job.stage_times[stage] = [time.time(), None]

    def end_stage(self, job: Job, stage: str) -> None:
        with self._lock:
            if stage in job.stages_running:
                job.stages_running.remove(stage)
            if stage not in job.stages_done:
                job.stages_done.append(stage)
            if stage in job.stage_times:
                job.stage_times[stage][1] = time.time()

    def finish(self, job: Job, result: dict) -> None:
        with self._lock:
            for s in job.stages_running:
                if s not in job.stages_done:
                    job.stages_done.append(s)
            job.stages_running.clear()
            job.status = "done"
            job.result = result
            job.finished_at = time.time()

    def fail(self, job: Job, error: str, stage: str | None = None) -> None:
        with self._lock:
            job.status = "error"
            job.error = error
            job.error_stage = stage if stage is not None else next(iter(job.stages_running), None)
            job.stages_running.clear()
            job.finished_at = time.time()


_registry: JobRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> JobRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = JobRegistry()
        return _registry
