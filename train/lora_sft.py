"""LoRA SFT on Qwen2.5-VL language backbone (vision frozen)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os

from src.config import load_config, setup_env, shared_dir
from src import metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lora_train.jsonl"


def _ensure_torchao_compat() -> None:
    """peft may import torchao; upgrade only when too old. Optional CUDA kernels are not used for bf16 LoRA."""
    import subprocess

    try:
        import importlib.metadata as im

        ver = im.version("torchao")
        major, minor = (int(x) for x in ver.split(".")[:2])
        if (major, minor) >= (0, 16):
            print(
                "Note: torchao may print 'Failed to load ...cutlass/mxfp8' on Colab — "
                "safe to ignore for bf16 LoRA (those kernels are not used)."
            )
            return
        print(f"Upgrading torchao {ver} -> >=0.16.0 (required by peft)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "torchao>=0.16.0"],
        )
        print(
            "Note: ignore torchao 'Failed to load ...cutlass/mxfp8' lines — bf16 LoRA does not need them."
        )
    except im.PackageNotFoundError:
        pass


def main():
    setup_env()
    _ensure_torchao_compat()
    cfg = load_config()
    shared = shared_dir(cfg)
    out_dir = shared / "lora_ckpts"
    final = shared / "lora_adapter_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    ds = load_dataset("json", data_files=str(DATA), split="train")

    with metrics.phase("lora_sft", model=model_id):
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        for n, p in model.named_parameters():
            if "visual" in n or "merger" in n:
                p.requires_grad = False
        lora = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)

        def fmt(row):
            msgs = row["messages"]
            text = proc.apply_chat_template(msgs, tokenize=False)
            return {"text": text}

        ds = ds.map(fmt)

        sft_cfg = SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1.5e-4,
            bf16=True,
            logging_steps=5,
            save_strategy="steps",
            save_steps=25,
            save_total_limit=3,
            max_length=2048,
        )
        trainer = SFTTrainer(model=model, train_dataset=ds, args=sft_cfg)
        trainer.train()
        trainer.save_model(str(final))
        print(f"Saved adapter -> {final}")


if __name__ == "__main__":
    main()
