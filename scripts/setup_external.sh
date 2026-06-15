#!/usr/bin/env bash
# Clone external dependencies into build_work/external/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/external"
cd "$ROOT/external"

clone() {
  local url="$1" dir="$2"
  if [ ! -d "$dir/.git" ]; then
    git clone --depth 1 "$url" "$dir"
  else
    echo "exists: $dir"
  fi
}

clone https://github.com/lightflow16/sde_project_bMAS.git sde_project_bMAS
clone https://github.com/lightflow16/mini_protein_pipeline_6a95.git mini_protein_pipeline_6a95
clone https://github.com/Kuhlman-Lab/ThermoMPNN.git ThermoMPNN
clone https://github.com/dauparas/ProteinMPNN.git ProteinMPNN

# Patch ThermoMPNN local path for notebook
ROOT="$ROOT" python3 - <<'PY'
import os, yaml
from pathlib import Path
root = Path(os.environ["ROOT"])
p = root / "external" / "ThermoMPNN" / "local.yaml"
if p.exists():
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg.setdefault("platform", {})["thermompnn_dir"] = str((root / "external" / "ThermoMPNN").resolve())
    p.write_text(yaml.dump(cfg, default_flow_style=False))
    print("patched", p)
PY

echo "Done."
