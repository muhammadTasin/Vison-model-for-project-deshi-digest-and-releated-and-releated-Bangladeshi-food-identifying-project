#!/usr/bin/env python3
"""Experimental Qwen3-VL + PEFT adapter inference CLI.

The exported checkpoint identifies Qwen3VLForConditionalGeneration, PEFT LoRA,
and qwen-vl-utils. This example has not been rerun in the original Kaggle
environment; validate dependency versions and outputs before application use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CANONICAL_CLASSES = (
    "bakorkhani",
    "bangladeshi_biriyani",
    "beguni",
    "chickpea_curry",
    "egg_omelette",
    "fuchka",
    "haleem",
    "hilsa_fish",
    "kacha_golla",
    "kala_bhuna",
    "kebab",
    "khichuri",
    "morog_polao",
    "nehari",
    "paratha",
    "potato_bhorta",
    "roshogolla",
    "roshmalai",
    "sweet_yogurt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experimental closed-set Deshi Digest inference."
    )
    parser.add_argument("image", type=Path, help="Path to one local image")
    parser.add_argument(
        "--adapter", type=Path, required=True, help="Extracted checkpoint-348 directory"
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3-VL-2B-Instruct", help="Base model ID or path"
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=Path(__file__).with_name("label_aliases.json"),
        help="Exact label alias map",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def load_aliases(path: Path) -> dict[str, str]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read alias map {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Alias map must be a JSON object")

    aliases: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Every alias and target must be a string")
        normalized_key = key.strip().lower().replace("-", " ").replace(" ", "_")
        if value not in CANONICAL_CLASSES:
            raise ValueError(f"Alias {key!r} targets unsupported class {value!r}")
        aliases[normalized_key] = value
    return aliases


def normalize_exact_label(raw: str, aliases: dict[str, str]) -> str | None:
    """Normalize a complete response; never search within arbitrary text."""
    candidate = raw.strip().lower().replace("-", " ").replace(" ", "_")
    if candidate in CANONICAL_CLASSES:
        return candidate
    return aliases.get(candidate)


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise ValueError(f"Image does not exist or is not a file: {args.image}")
    if args.max_new_tokens < 1 or args.max_new_tokens > 64:
        raise ValueError("--max-new-tokens must be between 1 and 64")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        required = args.adapter / name
        if not required.is_file():
            raise ValueError(f"Missing required adapter file: {required}")


def run_inference(args: argparse.Namespace) -> str:
    # Imports are delayed so `--help` and input validation work without the
    # heavyweight model stack installed.
    try:
        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependency. Install a platform-appropriate PyTorch "
            "build and then `pip install -r requirements.txt`."
        ) from exc

    class_list = ", ".join(CANONICAL_CLASSES)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a closed-set Bangladeshi food image classifier. "
                f"Choose exactly one food_id from this list: {class_list}. "
                "Return only the exact food_id. Do not return JSON, explanation, "
                "punctuation, or extra text."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(args.image.resolve())},
                {
                    "type": "text",
                    "text": "Classify the main visible food. Return only its exact food_id.",
                },
            ],
        },
    ]

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    model.eval()
    processor = AutoProcessor.from_pretrained(args.base_model)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def main() -> int:
    args = parse_args()
    try:
        validate_inputs(args)
        aliases = load_aliases(args.aliases)
        raw_response = run_inference(args)
        food_id = normalize_exact_label(raw_response, aliases)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if food_id is None:
        result: dict[str, Any] = {
            "status": "unknown",
            "food_id": None,
            "display_name": None,
            "confidence": "uncalibrated",
            "alternatives": [],
            "raw_response": raw_response,
        }
    else:
        result = {
            "status": "ok",
            "food_id": food_id,
            "display_name": food_id.replace("_", " ").title(),
            "confidence": "uncalibrated",
            "alternatives": [],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
