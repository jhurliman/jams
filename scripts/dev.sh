#!/usr/bin/env bash
# One-command local dev environment:
#   - jams analysis API        (FastAPI,  http://localhost:8000)
#   - annotator API            (Hono,     http://localhost:8787)
#   - annotator frontend       (Vite,     http://localhost:5173)
# Installs Python + webapp deps on first run, then starts everything and
# opens the annotator in your browser. Ctrl-C stops all three.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null 2>&1 || {
  echo "error: uv is required (https://docs.astral.sh/uv/getting-started/installation/)" >&2
  exit 1
}
command -v npm >/dev/null 2>&1 || {
  echo "error: node/npm >= 20 is required (https://nodejs.org)" >&2
  exit 1
}

# Pre-flight: fail with a clear message if a port is already taken (e.g. another
# dev.sh, or `npm run dev` started by hand) rather than half-starting.
for spec in "8000 jams API" "8787 annotator API" "5173 annotator frontend"; do
  port=${spec%% *}; name=${spec#* }
  if lsof -nP -i ":$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "error: port $port ($name) is already in use — is the stack already running?" >&2
    echo "       stop it first, or open http://localhost:5173 if that's your annotator." >&2
    exit 1
  fi
done

echo "==> Installing Python deps (uv sync)"
uv sync

echo "==> Installing webapp deps (npm install)"
(cd webapp && npm install --no-audit --no-fund)

# Kill the whole process group on exit so no server is left behind.
trap 'kill 0' EXIT INT TERM

echo "==> Starting jams API on :8000"
uv run jams &

echo "==> Starting annotator (API :8787 + frontend :5173)"
(cd webapp && npm run dev) &

sleep 3
URL=http://localhost:5173
if command -v open >/dev/null 2>&1; then open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
else echo "==> Open $URL in your browser"
fi

wait
