#!/usr/bin/env python3
"""Check local image paths referenced by one or more JSONL manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument(
        "--root", type=Path, help="Resolve relative image paths against this directory"
    )
    parser.add_argument(
        "--check-decode", action="store_true", help="Decode each image with Pillow"
    )
    return parser.parse_args()


def image_paths(record: dict[str, Any], line: int) -> Iterable[str]:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"Line {line}: images must be a non-empty array")
    for image in images:
        if isinstance(image, str) and image.strip():
            yield image
        elif isinstance(image, dict) and isinstance(image.get("path"), str):
            yield image["path"]
        else:
            raise ValueError(f"Line {line}: invalid image path entry")


def verify_decode(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --check-decode") from exc
    with Image.open(path) as image:
        image.verify()


def main() -> int:
    args = parse_args()
    checked = 0
    failures: list[str] = []
    try:
        for manifest in args.manifests:
            root = args.root if args.root is not None else manifest.parent
            with manifest.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    payload: Any = json.loads(raw_line)
                    if not isinstance(payload, dict):
                        raise ValueError(f"{manifest}:{line_number}: record must be an object")
                    for raw_path in image_paths(payload, line_number):
                        path = Path(raw_path)
                        if not path.is_absolute():
                            path = root / path
                        checked += 1
                        if not path.is_file():
                            failures.append(f"missing: {path}")
                            continue
                        if args.check_decode:
                            try:
                                verify_decode(path)
                            except (OSError, RuntimeError) as exc:
                                failures.append(f"cannot decode {path}: {exc}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"checked": checked, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
