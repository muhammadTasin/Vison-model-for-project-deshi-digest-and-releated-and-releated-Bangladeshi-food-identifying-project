#!/usr/bin/env python3
"""Print or run the recorded Stage-4 MS-Swift QLoRA training command."""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("stage4-output"))
    parser.add_argument("--run", action="store_true", help="Execute instead of printing the command")
    args = parser.parse_args()
    for manifest in (args.train_manifest, args.val_manifest):
        if not manifest.is_file():
            parser.error(f"manifest not found: {manifest}")
    command = [
        "swift", "sft", "--model", "Qwen/Qwen3-VL-2B-Instruct", "--model_type", "qwen3_vl",
        "--template", "qwen3_vl", "--dataset", str(args.train_manifest), "--val_dataset", str(args.val_manifest),
        "--train_type", "lora", "--quant_bits", "4", "--bnb_4bit_quant_type", "nf4",
        "--bnb_4bit_use_double_quant", "true", "--lora_rank", "4", "--lora_alpha", "16",
        "--lora_dropout", "0.05", "--num_train_epochs", "1.5", "--learning_rate", "2e-5",
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "8",
        "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.05", "--weight_decay", "0.01",
        "--max_grad_norm", "1.0", "--max_pixels", "262144", "--max_length", "512",
        "--image_max_token_num", "128", "--gradient_checkpointing", "true", "--output_dir", str(args.output_dir),
    ]
    print(" ".join(command))
    if args.run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
