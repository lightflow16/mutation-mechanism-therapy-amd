"""Load targets.yaml and runtime paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "targets.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_TARGETS
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_target(cfg: dict, gene: str, mutation: str | None = None) -> dict[str, Any]:
    for t in cfg.get("targets", []):
        if t["gene"].upper() == gene.upper():
            if mutation is None or t["mutation"].upper() == mutation.upper():
                return t
    raise KeyError(f"Target not found: {gene} {mutation or ''}")


def is_rocm() -> bool:
    try:
        import torch
        return "rocm" in torch.__version__.lower()
    except Exception:
        return False


def use_int8() -> bool:
    """True when int8 quantization should be applied to LLMs.

    Enabled when USE_INT8=1 env var is set OR pipeline.use_int8 is true in config,
    AND a CUDA GPU is present, AND NOT ROCm (bitsandbytes is CUDA-only).
    MI300X has 192 GiB HBM — quantization is unnecessary and unsupported there.
    """
    if is_rocm():
        return False
    if os.environ.get("USE_INT8") == "1":
        return True
    try:
        return bool(load_config().get("pipeline", {}).get("use_int8", False))
    except Exception:
        return False


def _resolve_path(preferred: str | None, local_name: str) -> Path:
    """Use notebook paths on AMD; fall back to repo-local dirs for offline dev."""
    if preferred:
        p = Path(preferred)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            pass
    local = ROOT / local_name
    local.mkdir(parents=True, exist_ok=True)
    return local


def configure_paths(cfg: dict | None = None) -> Path:
    """Set HF_HOME + METRICS_DIR for AMD /workspace/shared, Colab, or repo-local."""
    cfg = cfg or load_config()
    ws = Path("/workspace/shared")
    if ws.parent.exists():
        hf = ws / "hf_cache"
        met = ws / "metrics"
    elif Path("/content").exists():
        hf = ROOT / "shared" / "hf_cache"
        met = ROOT / "metrics" / "colab"
    else:
        hf = ROOT / "shared" / "hf_cache"
        met = ROOT / "metrics" / "local"
    hf.mkdir(parents=True, exist_ok=True)
    met.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf))
    os.environ.setdefault("METRICS_DIR", str(met))
    if is_rocm():
        os.environ.setdefault("LLM_BACKEND", "transformers")
        # Reduce VRAM fragmentation on MI300X by enabling expandable memory segments.
        os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    return ROOT


def shared_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return _resolve_path(cfg.get("paths", {}).get("shared"), "shared")


def metrics_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    env = os.environ.get("METRICS_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return _resolve_path(cfg.get("paths", {}).get("metrics"), "metrics")


def setup_env(cfg: dict | None = None) -> None:
    """Set HF_HOME, METRICS_DIR, and platform defaults from config."""
    configure_paths(cfg)
    cfg = cfg or load_config()
    paths = cfg.get("paths", {})
    hf = paths.get("hf_cache")
    if hf:
        hf_path = Path(hf)
        try:
            hf_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(hf_path))
        except OSError:
            local_hf = ROOT / "shared" / "hf_cache"
            local_hf.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(local_hf))
    os.environ.setdefault("METRICS_DIR", str(metrics_dir(cfg)))
    boltz = ROOT / "external" / "boltz_venv" / "bin" / "boltz"
    if boltz.is_file():
        os.environ.setdefault("BOLTZ_BIN", str(boltz))


def reload_src_modules() -> None:
    """Drop cached src imports so git pull changes apply (required on Colab notebooks)."""
    import sys

    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
