"""Async analysis jobs: registry semantics + the /v1/analyze async=true flow.

The analysis itself is mocked — these tests cover job lifecycle, concurrent-stage
bookkeeping, error propagation, and endpoint contracts, not the models.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

import jams.api.routes as routes
from jams.api.app import app
from jams.api.jobs import JobRegistry

client = TestClient(app)


# ------------------------------------------------------------------- registry unit
def test_registry_concurrent_stage_lifecycle():
    reg = JobRegistry()
    job = reg.create()
    reg.start_stage(job, "key")
    reg.start_stage(job, "tempo")
    assert set(job.stages_running) == {"key", "tempo"}
    reg.end_stage(job, "tempo")
    assert job.stages_running == ["key"]
    assert job.stages_done == ["tempo"]
    reg.start_stage(job, "structure")
    reg.finish(job, {"duration_sec": 1.0})
    pub = job.to_public()
    assert pub["status"] == "done"
    assert pub["stages_running"] == []
    # unfinished-but-running stages are folded into done on finish
    assert set(pub["stages_done"]) == {"key", "tempo", "structure"}
    assert pub["result"] == {"duration_sec": 1.0}


def test_registry_fail_records_stage():
    reg = JobRegistry()
    job = reg.create()
    reg.start_stage(job, "structure")
    reg.fail(job, "boom")
    pub = job.to_public()
    assert pub["status"] == "error"
    assert pub["error"] == "boom"
    assert pub["error_stage"] == "structure"


def test_registry_ttl_purges_finished_jobs():
    reg = JobRegistry(ttl_seconds=0.01)
    job = reg.create()
    reg.finish(job, {})
    time.sleep(0.05)
    assert reg.get(job.id) is None
    # running jobs are never purged
    running = reg.create()
    time.sleep(0.05)
    assert reg.get(running.id) is running


# --------------------------------------------------------------- endpoint contract
def _poll_done(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_async_analyze_reports_stages_and_result(cmajor_wav, monkeypatch):
    def fake_analyze(path, *, on_stage=None, **kwargs):
        for stage in ("key", "tempo"):
            on_stage(stage, "start")
            on_stage(stage, "done")
        return {"duration_sec": 2.5, "key": None, "tempo": None}

    monkeypatch.setattr(routes, "analyze_track", fake_analyze)
    with open(cmajor_wav, "rb") as fh:
        r = client.post(
            "/v1/analyze",
            files={"file": ("cmajor.wav", fh, "audio/wav")},
            data={"async": "true"},
        )
    assert r.status_code == 202, r.text
    body = _poll_done(r.json()["job_id"])
    assert body["status"] == "done"
    assert body["stages_done"] == ["key", "tempo"]
    assert body["result"]["duration_sec"] == 2.5
    assert body["result"]["filename"] == "cmajor.wav"


def test_async_analyze_propagates_errors(cmajor_wav, monkeypatch):
    def fake_analyze(path, *, on_stage=None, **kwargs):
        on_stage("structure", "start")
        raise RuntimeError("structure backend not configured")

    monkeypatch.setattr(routes, "analyze_track", fake_analyze)
    with open(cmajor_wav, "rb") as fh:
        r = client.post(
            "/v1/analyze",
            files={"file": ("cmajor.wav", fh, "audio/wav")},
            data={"async": "true"},
        )
    assert r.status_code == 202
    body = _poll_done(r.json()["job_id"])
    assert body["status"] == "error"
    assert "structure backend" in body["error"]
    assert body["error_stage"] == "structure"


def test_unknown_job_is_404():
    assert client.get("/v1/jobs/nope").status_code == 404


def test_sync_analyze_unchanged(cmajor_wav, monkeypatch):
    """No async field -> the blocking contract (result body, no job envelope)."""
    monkeypatch.setattr(
        routes, "analyze_track",
        lambda path, **kw: {"duration_sec": 1.0, "key": None, "tempo": None},
    )
    with open(cmajor_wav, "rb") as fh:
        r = client.post("/v1/analyze", files={"file": ("cmajor.wav", fh, "audio/wav")})
    assert r.status_code == 200, r.text
    assert "job_id" not in r.json()
    assert r.json()["duration_sec"] == 1.0
