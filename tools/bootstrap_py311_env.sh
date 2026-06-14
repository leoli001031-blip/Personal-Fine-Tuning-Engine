#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
UV_CACHE_DIR="$ROOT_DIR/.uv-cache"
PIP_CACHE_DIR="$ROOT_DIR/.pip-cache"
PYTHON_BIN="${PFE_PYTHON_BIN:-}"
PFE_BOOTSTRAP_EXTRAS="${PFE_BOOTSTRAP_EXTRAS:-dev}"

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

if [[ -n "$PFE_BOOTSTRAP_EXTRAS" ]]; then
  INSTALL_SPEC=".[${PFE_BOOTSTRAP_EXTRAS}]"
else
  INSTALL_SPEC="."
fi

echo "Installing package spec: $INSTALL_SPEC"

if command -v uv >/dev/null 2>&1; then
  (
    cd "$ROOT_DIR"
    UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install --python "$VENV_DIR/bin/python" -e "$INSTALL_SPEC"
  )
else
  PIP_CACHE_DIR="$PIP_CACHE_DIR" "$VENV_DIR/bin/pip" install --upgrade pip
  (
    cd "$ROOT_DIR"
    PIP_CACHE_DIR="$PIP_CACHE_DIR" "$VENV_DIR/bin/pip" install -e "$INSTALL_SPEC"
  )
fi

echo
echo "Environment ready."
echo "Activate with: source .venv/bin/activate"
echo "Run fast tests with: make test"
echo "Install heavier extras with: PFE_BOOTSTRAP_EXTRAS=dev,training tools/bootstrap_py311_env.sh"
