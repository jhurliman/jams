#!/usr/bin/env python
"""wandb-offline -> MLflow sync daemon (runs on aleph0 under ~/mlflow_venv).

Scans each ~/all-in-one/wandb/offline-run-* .wandb file with wandb's DataStore
scanner, extracts the FULL metric history + config, and mirrors them into the
MLflow server (experiment "raveform-structure"). Incremental and idempotent:
per-run last-synced step lives in /mnt/d/jams/mlflow/sync_state.json; MLflow
runs are found/created by the wandb run id tag, so restarts never duplicate.

Runs launched after the direct-MLflow patch (marker ~/all-in-one/.mlflow_direct_enabled)
are skipped -- they log to MLflow themselves; syncing them too would duplicate.

Also logs, for runs whose training process is still alive: GPU util/mem
(system/*) at sync time and the tail-refreshed train_foldN.log as an artifact.

Usage: wandb2mlflow.py [--once]
"""
import argparse
import contextlib
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
WANDB_DIR = Path.home() / "all-in-one" / "wandb"
STATE_PATH = Path("/mnt/d/jams/mlflow/sync_state.json")
MARKER = Path.home() / "all-in-one" / ".mlflow_direct_enabled"
NVSMI = "/usr/lib/wsl/lib/nvidia-smi"
EXPERIMENT = "raveform-structure"
INTERVAL = 300
KEY_RE = re.compile(r"[^A-Za-z0-9_\-./ ]")


def scan_wandb_file(path):
    """Return (wandb_run_id, config dict, history rows). Tolerates live tails."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    ds = DataStore()
    ds.open_for_scan(str(path))
    run_id, config, history = None, {}, []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break  # truncated tail of a file being written right now
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        which = rec.WhichOneof("record_type")
        if which == "run":
            run_id = rec.run.run_id or run_id
            _merge_config(config, rec.run.config.update)
        elif which == "config":
            _merge_config(config, rec.config.update)
        elif which == "history":
            row = {}
            for item in rec.history.item:
                # wandb >=0.28 puts the metric name in nested_key, not key
                k = item.key or ".".join(item.nested_key)
                if not k:
                    continue
                with contextlib.suppress(Exception):
                    row[k] = json.loads(item.value_json)
            if row:
                history.append(row)
    return run_id, config, history


def _merge_config(config, items):
    for item in items:
        try:
            v = json.loads(item.value_json)
        except Exception:
            continue
        if isinstance(v, dict) and set(v) <= {"value", "desc"}:
            v = v.get("value")
        config[item.key] = v


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = str(v)[:500]
    return out


def find_fold(config):
    flat = flatten(config)
    for k, v in flat.items():
        if k == "fold" or k.endswith(".fold"):
            return str(v)
    return "unknown"


def gpu_stats():
    try:
        out = subprocess.run(
            [NVSMI, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().split(",")
        return float(out[0]), float(out[1])
    except Exception:
        return None, None


def proc_alive(fold):
    r = subprocess.run(["pgrep", "-f", f"fold={fold}"], capture_output=True)
    return r.returncode == 0


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"runs": {}}


def save_state(state):
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE_PATH)


def dir_started_after_marker(run_dir):
    if not MARKER.exists():
        return False
    m = re.search(r"offline-run-(\d{8}_\d{6})-", run_dir.name)
    if not m:
        return False
    started = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
    return started > MARKER.stat().st_mtime


def ensure_run(client, exp_id, wandb_id, fold, params):
    hits = client.search_runs(
        [exp_id], filter_string=f"tags.wandb_run_id = '{wandb_id}'")
    for h in hits:
        return h.info.run_id, False
    # A direct-logged run for this fold that is currently RUNNING means the
    # trainer logs itself; don't shadow it (belt over the marker suspenders).
    direct = client.search_runs(
        [exp_id],
        filter_string=f"tags.source = 'direct' and tags.fold = '{fold}'",
        run_view_type=1)
    for d in direct:
        if d.info.status == "RUNNING":
            return None, False
    run = client.create_run(
        exp_id,
        run_name=f"fold-{fold}",
        tags={"wandb_run_id": wandb_id, "fold": fold,
              "host": "aleph0", "source": "wandb-sync"},
    )
    rid = run.info.run_id
    if params:
        for i in range(0, len(params), 100):
            client.log_batch(rid, params=[Param(k, v) for k, v in params[i:i + 100]])
    return rid, True


def sync_once(client, exp_id, state):
    for run_dir in sorted(WANDB_DIR.glob("offline-run-*")):
        wandb_files = list(run_dir.glob("*.wandb"))
        if not wandb_files:
            continue
        if dir_started_after_marker(run_dir):
            continue  # direct-logging era: the trainer logs to MLflow itself
        st = state["runs"].setdefault(run_dir.name, {"last_step": -1, "points": 0})
        wandb_id, config, history = scan_wandb_file(wandb_files[0])
        if wandb_id is None:
            continue
        if not history and "mlflow_run_id" not in st:
            continue  # crashed-at-startup attempts never become MLflow runs
        fold = find_fold(config)
        if "mlflow_run_id" not in st:
            params = sorted(flatten(config).items())
            rid, _created = ensure_run(client, exp_id, wandb_id, fold, params)
            if rid is None:
                state["runs"][run_dir.name]["skip"] = "direct-run-exists"
                continue
            st["mlflow_run_id"], st["wandb_run_id"], st["fold"] = rid, wandb_id, fold
        rid = st["mlflow_run_id"]

        new_metrics, max_step = [], st["last_step"]
        for row in history:
            step = row.get("_step")
            if step is None or step <= st["last_step"]:
                continue
            ts = int(float(row.get("_timestamp", time.time())) * 1000)
            for k, v in row.items():
                if k.startswith("_") or not isinstance(v, (int, float)):
                    continue
                new_metrics.append(Metric(KEY_RE.sub("_", k), float(v), ts, int(step)))
            max_step = max(max_step, int(step))
        for i in range(0, len(new_metrics), 900):
            client.log_batch(rid, metrics=new_metrics[i:i + 900])
        st["last_step"] = max_step
        st["points"] += len(new_metrics)

        alive = proc_alive(st.get("fold", fold))
        if alive:
            util, mem = gpu_stats()
            ts = int(time.time() * 1000)
            sysm = [Metric("system/gpu_util", util, ts, max(max_step, 0)),
                    Metric("system/gpu_mem_mib", mem, ts, max(max_step, 0))] \
                if util is not None else []
            if sysm:
                client.log_batch(rid, metrics=sysm)
            log_path = Path.home() / f"train_fold{st.get('fold', fold)}.log"
            if log_path.exists():
                try:
                    client.log_artifact(rid, str(log_path))
                except Exception as exc:
                    print(f"[warn] artifact upload failed: {exc}", flush=True)
        elif not st.get("terminated"):
            client.set_terminated(rid)
            st["terminated"] = True
        print(f"[{time.strftime('%H:%M:%S')}] {run_dir.name} fold={st.get('fold', fold)} "
              f"+{len(new_metrics)} pts (total {st['points']}, step {st['last_step']}, "
              f"alive={alive})", flush=True)
    save_state(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    mlflow.set_tracking_uri(TRACKING)
    client = MlflowClient(TRACKING)
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)
    state = load_state()
    while True:
        try:
            sync_once(client, exp_id, state)
        except Exception as exc:
            print(f"[error] sync pass failed: {type(exc).__name__}: {exc}", flush=True)
        if args.once:
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
