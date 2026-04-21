#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
PIP_CACHE_DIR="$ROOT_DIR/.pip-cache"
PYTHON_BIN="${PFE_PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in /opt/homebrew/bin/python3.11 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "error: Python 3.11 was not found. Set PFE_PYTHON_BIN to a Python 3.11 executable." >&2
  exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"

if command -v uv >/dev/null 2>&1; then
  (
    cd "$ROOT_DIR"
    UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install --python "$VENV_DIR/bin/python" -e '.[dev,training]'
  )
else
  PIP_CACHE_DIR="$PIP_CACHE_DIR" "$VENV_DIR/bin/pip" install --upgrade pip
  (
    cd "$ROOT_DIR"
    PIP_CACHE_DIR="$PIP_CACHE_DIR" "$VENV_DIR/bin/pip" install -e '.[dev,training]'
  )
fi

echo
echo "Environment ready."
echo "Activate with: source .venv/bin/activate"
echo "Run tests with: .venv/bin/python -m unittest discover -s tests -v"
