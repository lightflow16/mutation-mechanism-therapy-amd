"""LoRA SFT on Qwen2.5-VL language backbone (vision frozen)."""
from __future__ import annotations

import os
from pathlib import Path

from src.config import load_config, setup_env
from src import metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lora_train.jsonl"


def main():
    setup_env()
    cfg = load_config()
    paths = cfg.get("paths", {})
    out_dir = paths.get("lora_ckpts", "/workspace/shared/lora_ckpts")
    final = paths.get("lora_adapter_final", "/workspace/shared/lora_adapter_final")
    os.makedirs(out_dir, exist_ok=True)

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
            output_dir=out_dir,
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
        trainer.save_model(final)
        print(f"Saved adapter -> {final}")


if __name__ == "__main__":
    main()
