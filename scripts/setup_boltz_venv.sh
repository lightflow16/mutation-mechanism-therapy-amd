#!/usr/bin/env bash
# Isolated Boltz venv (numpy<2) - call via BOLTZ_BIN from main env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/external/boltz_venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install boltz
echo "export BOLTZ_BIN=$VENV/bin/boltz"
echo "Use: boltz predict ... --no_kernels"
