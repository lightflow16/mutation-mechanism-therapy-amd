#!/usr/bin/env bash
# Isolated Boltz venv (numpy<2) — required for full rescue stack on Colab + AMD.
# Colab often fails `python3 -m venv` on ensurepip; use --without-pip + get-pip.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/external/boltz_venv"
BOLTZ_BIN="$VENV/bin/boltz"

if [ -x "$BOLTZ_BIN" ]; then
  echo "boltz venv exists: $BOLTZ_BIN"
  echo "export BOLTZ_BIN=$BOLTZ_BIN"
  exit 0
fi

# Remove broken partial venv from a prior failed ensurepip (common on Colab).
if [ -d "$VENV" ]; then
  echo "Removing incomplete boltz venv at $VENV"
  rm -rf "$VENV"
fi

_create_venv() {
  echo "Creating isolated venv at $VENV"
  if python3 -m venv "$VENV" --without-pip; then
    echo "Bootstrapping pip (Colab-safe)..."
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python3" -
    return 0
  fi
  echo "venv --without-pip failed; trying virtualenv..."
  if ! command -v virtualenv >/dev/null 2>&1; then
    python3 -m pip install -q virtualenv
  fi
  virtualenv -p python3 "$VENV"
}

_create_venv
"$VENV/bin/python3" -m pip install -U pip wheel
"$VENV/bin/python3" -m pip install 'numpy<2' boltz

if [ ! -x "$BOLTZ_BIN" ]; then
  echo "ERROR: boltz install finished but $BOLTZ_BIN is missing" >&2
  exit 1
fi

echo "installed boltz -> $BOLTZ_BIN"
echo "export BOLTZ_BIN=$BOLTZ_BIN"
