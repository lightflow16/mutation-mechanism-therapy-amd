#!/usr/bin/env bash
# Isolated Boltz venv (numpy<2) — required for full rescue stack on Colab + AMD.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/external/boltz_venv"

if [ -x "$VENV/bin/boltz" ]; then
  echo "boltz venv exists: $VENV/bin/boltz"
else
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install 'numpy<2' boltz
  echo "installed boltz -> $VENV/bin/boltz"
fi

echo "export BOLTZ_BIN=$VENV/bin/boltz"
