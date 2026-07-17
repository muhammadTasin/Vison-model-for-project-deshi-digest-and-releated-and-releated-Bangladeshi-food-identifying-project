#!/usr/bin/env python3
"""Validate the structure and labels of an MS-SWIFT image JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CANONICAL_CLASSES = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--aliases",
        type=Path,
        default=Path(__file__).parents[1] / "inference" / "label_aliases.json",
    )
    return parser.parse_args()


def token(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def load_aliases(path: Path) -> dict[str, str]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Alias map must be a JSON object")
    aliases: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Alias keys and values must be strings")
        canonical = token(value)
        if canonical not in CANONICAL_CLASSES:
            raise ValueError(f"Alias {key!r} targets unsupported class {value!r}")
        aliases[token(key)] = canonical
    return aliases


def validate_record(
    record: dict[str, Any], line: int, aliases: dict[str, str]
) -> str:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"Line {line}: images must be a non-empty array")
    for image in images:
        if not isinstance(image, (str, dict)):
            raise ValueError(f"Line {line}: each image must be a path or object")
        if isinstance(image, dict) and not isinstance(image.get("path"), str):
            raise ValueError(f"Line {line}: image object requires a string path")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Line {line}: messages must be a non-empty array")
    assistant = [
        item
        for item in messages
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    if len(assistant) != 1:
        raise ValueError(f"Line {line}: expected exactly one assistant message")
    content = assistant[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Line {line}: assistant content must be a non-empty label")
    normalized = aliases.get(token(content), token(content))
    if normalized not in CANONICAL_CLASSES:
        raise ValueError(f"Line {line}: unsupported assistant label {content!r}")
    return normalized


def main() -> int:
    args = parse_args()
    try:
        aliases = load_aliases(args.aliases)
        counts: Counter[str] = Counter()
        records = 0
        with args.manifest.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                payload: Any = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Line {line_number}: record must be an object")
                counts[validate_record(payload, line_number, aliases)] += 1
                records += 1
        if records == 0:
            raise ValueError("Manifest contains no records")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"records": records, "class_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
