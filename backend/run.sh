#!/usr/bin/env bash
# Reality Debugger backend - one-step start for macOS / Linux.
# Creates the virtual environment on first run, installs dependencies, starts uvicorn.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if [ ! -d venv ]; then
  echo "Creating virtual environment (venv)..."
  "$PYTHON" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --disable-pip-version-check -q -r requirements.txt
echo "Starting backend on http://localhost:${PORT:-8000} (API docs: /api/docs)"
exec uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@"
