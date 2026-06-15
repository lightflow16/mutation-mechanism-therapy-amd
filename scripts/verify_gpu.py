#!/usr/bin/env python3
"""Quick GPU + platform sanity check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import configure_paths, is_rocm
from src.llm_client import use_vllm
from src.serving import platform_summary, verify_gpu_torch


def main() -> int:
    configure_paths()
    summary = platform_summary()
    summary["llm_backend"] = "vllm" if use_vllm() else "transformers"
    print(json.dumps(summary, indent=2))
    gpu = summary["gpu"]
    if not gpu.get("ok"):
        print("\nGPU NOT READY:", gpu.get("error"), file=sys.stderr)
        if is_rocm():
            print("If you ran 'pip install vllm', restart the session and install requirements.txt only.", file=sys.stderr)
        return 1
    print("\nGPU OK — ready for live inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
