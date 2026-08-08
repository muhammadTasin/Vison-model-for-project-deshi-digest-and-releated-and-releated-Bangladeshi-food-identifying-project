#!/usr/bin/env python3
"""Print or run deterministic Stage-4 Qwen3-VL MS-Swift image inference."""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "stage4_27class")
    parser.add_argument("--run", action="store_true", help="Execute instead of printing the command")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (args.adapter / name).is_file():
            parser.error(f"missing adapter artifact: {args.adapter / name}")
    command = [
        "swift", "infer", "--model", args.model, "--adapters", str(args.adapter),
        "--model_type", "qwen3_vl", "--template", "qwen3_vl", "--attn_impl", "sdpa",
        "--max_batch_size", "1", "--temperature", "0", "--images", str(args.image),
        "--query", "Identify the Bangladeshi food. Reply with exactly one supported class label.",
    ]
    print(" ".join(command))
    if args.run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
