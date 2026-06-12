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
    """Set HF_HOME and METRICS_DIR from config."""
    cfg = cfg or load_config()
    paths = cfg.get("paths", {})
    hf = paths.get("hf_cache")
    if hf:
        hf_path = Path(hf)
        try:
            hf_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(hf_path))
        except OSError:
            local_hf = ROOT / "hf_cache"
            local_hf.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(local_hf))
    os.environ.setdefault("METRICS_DIR", str(metrics_dir(cfg)))
