"""Platform detection and run manifest for Colab (CUDA) vs AMD (ROCm) comparison."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from src.config import ROOT, is_rocm, load_config


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def detect_platform() -> dict[str, Any]:
    """Return stable platform_id used for metrics bundles and cross-platform diff."""
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        torch_ver = torch.__version__
    except Exception:
        cuda = False
        torch_ver = "unknown"

    if is_rocm():
        platform_id = "amd_rocm"
        vendor = "AMD"
    elif cuda and Path("/content").exists():
        platform_id = "colab_cuda"
        vendor = "NVIDIA"
    elif cuda:
        platform_id = "cuda"
        vendor = "NVIDIA"
    else:
        platform_id = "local_cpu"
        vendor = "CPU"

    return {
        "platform_id": platform_id,
        "vendor": vendor,
        "gpu_name": _gpu_name() if cuda else "none",
        "torch_version": torch_ver,
        "llm_backend": os.environ.get("LLM_BACKEND", "auto"),
        "metrics_dir": os.environ.get("METRICS_DIR", ""),
        "hostname": os.environ.get("HOSTNAME", ""),
        "git_commit": _git_commit(),
    }


def write_run_manifest(extra: dict[str, Any] | None = None) -> Path:
    """Write run_manifest.json into METRICS_DIR at session start."""
    from src.config import metrics_dir

    md = metrics_dir()
    md.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **detect_platform(),
        "config": load_config().get("pipeline", {}),
    }
    if extra:
        manifest.update(extra)
    path = md / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
