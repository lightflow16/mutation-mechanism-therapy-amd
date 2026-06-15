"""vLLM serving helpers and GPU verification."""
from __future__ import annotations

import subprocess
import urllib.request

from src.config import is_rocm


def health_check(base_url: str = "http://localhost:8000/v1") -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_all_endpoints(endpoints: dict) -> dict[str, bool]:
    return {name: health_check(ep["base_url"]) for name, ep in endpoints.items()}


def wait_for_endpoints(
    endpoints: dict,
    *,
    timeout_s: int = 900,
    poll_s: float = 10.0,
) -> dict[str, bool]:
    """Poll until all endpoints respond or timeout. First model download can take 10+ min."""
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = check_all_endpoints(endpoints)
        if all(status.values()):
            return status
        time.sleep(poll_s)
    return check_all_endpoints(endpoints)


def vllm_import_ok() -> bool:
    if is_rocm():
        return False
    try:
        import vllm._C  # noqa: F401
        return True
    except Exception:
        return False


def verify_gpu_torch() -> dict:
    """Real GPU test — is_available() alone can lie after a broken pip install."""
    out: dict = {"torch": None, "available": False, "ok": False, "error": None, "device": None}
    try:
        import torch

        out["torch"] = torch.__version__
        out["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["device"] = torch.cuda.get_device_name(0)
        x = torch.zeros(1, device="cuda")
        out["ok"] = True
        out["tensor"] = str(x.device)
    except Exception as e:
        out["error"] = str(e)
    return out


def platform_summary() -> dict:
    gpu = verify_gpu_torch()
    return {
        "rocm": is_rocm(),
        "vllm_import_ok": vllm_import_ok(),
        "gpu": gpu,
        "vllm_supported": not is_rocm(),
    }


def amd_smi_snapshot() -> str:
    try:
        return subprocess.run(
            ["amd-smi", "monitor", "-putm"], capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception as e:
        return str(e)
