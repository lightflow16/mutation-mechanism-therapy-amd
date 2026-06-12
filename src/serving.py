"""vLLM serving helpers."""
from __future__ import annotations

import subprocess
import urllib.request


def health_check(base_url: str = "http://localhost:8000/v1") -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_all_endpoints(endpoints: dict) -> dict[str, bool]:
    return {name: health_check(ep["base_url"]) for name, ep in endpoints.items()}


def amd_smi_snapshot() -> str:
    try:
        return subprocess.run(
            ["amd-smi", "monitor", "-putm"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception as e:
        return str(e)
