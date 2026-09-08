"""Worker startup must ignore the caller's incompatible Python selection."""

import ast
import json
import shutil

import pytest

from jams.analysis import stems as S
from jams.config import Settings


@pytest.mark.parametrize("script", [S._STEMS_WORKER, S._DRUM_WORKER, S._YOURMT3_WORKER])
def test_worker_python_satisfies_script(script):
    from packaging.specifiers import SpecifierSet

    line = next(line for line in script.read_text().splitlines()
                if line.startswith("# requires-python = "))
    requirement = ast.literal_eval(line.split("=", 1)[1].strip())
    assert S._WORKER_PYTHON in SpecifierSet(requirement)


def test_worker_ignores_project_and_environment_python(monkeypatch, tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is needed for the worker startup integration test")
    # No model dependencies: exercise the real uv interpreter selection and JSONL pipe.
    (tmp_path / ".python-version").write_text("3.14\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "incompatible-host"\nversion = "0.0.0"\n'
        'requires-python = ">=3.14"\n'
    )
    script = tmp_path / "worker.py"
    script.write_text(
        '# /// script\n# requires-python = ">=3.10,<3.12"\n# dependencies = []\n# ///\n'
        'import json, sys\n'
        'for line in sys.stdin:\n'
        '    print(json.dumps({"ok": True, "result": {"python": list(sys.version_info[:2])}}),'
        ' flush=True)\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_PYTHON", "3.14")
    monkeypatch.setattr(S, "get_settings", lambda: Settings(stems_uv=uv))
    worker = S._Worker(script, "probe")
    worker._spawn()
    try:
        stdout, stderr = worker._proc.communicate(json.dumps({}) + "\n", timeout=60)
        assert worker._proc.returncode == 0, stderr
        assert json.loads(stdout)["result"]["python"] == [3, 11]
    finally:
        if worker._proc.poll() is None:
            worker._proc.kill()
            worker._proc.wait()


@pytest.mark.parametrize("failure", ["eof", "broken-pipe"])
def test_failed_retry_explains_worker_python(monkeypatch, failure):
    worker = S._Worker(S._STEMS_WORKER, "stems")
    monkeypatch.setattr(worker, "_ensure_alive", lambda: None)
    monkeypatch.setattr(worker, "_spawn", lambda: None)

    def round_trip(request):
        if failure == "broken-pipe":
            raise BrokenPipeError
        return ""

    monkeypatch.setattr(worker, "_round_trip", round_trip)
    with pytest.raises(RuntimeError, match=r"Python 3\.11.*resolution errors"):
        worker.analyze({})
