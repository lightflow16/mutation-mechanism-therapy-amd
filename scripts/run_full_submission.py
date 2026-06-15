#!/usr/bin/env python3
"""Run all flows always: live matrix + eval + metrics bundle export."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import setup_env
from src.submission import run_full_submission


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full submission on Colab or AMD (all flows, all metrics).")
    ap.add_argument("--lora-path", default=None)
    ap.add_argument("--train-lora", action="store_true", help="Run LoRA SFT before live matrix")
    ap.add_argument("--skip-live", action="store_true", help="Eval/export only (traces must exist)")
    args = ap.parse_args()

    setup_env()
    run_full_submission(
        lora_path=args.lora_path,
        run_lora_train=args.train_lora,
        skip_live=args.skip_live,
    )


if __name__ == "__main__":
    main()
