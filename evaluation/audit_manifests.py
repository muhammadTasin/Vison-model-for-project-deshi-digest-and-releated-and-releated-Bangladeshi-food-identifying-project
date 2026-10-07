#!/usr/bin/env python3
"""Audit train/validation manifests for split overlap without publishing any paths.

Reads two JSONL manifests (chat-format rows: ``messages[-1].content`` is the label and
``images[0]`` is a path or ``{"path": ...}``) and prints aggregate counts only.
Image references and file names are never written to the output.

Checks that need only the manifests:
  * exact image-reference overlap between train and validation
  * duplicate references inside each split, and references carrying conflicting labels
  * labels outside metadata/labels.json
  * PithaNet rows whose source folder name does not match the label
  * same (label, file name) in both splits, which can be a name collision or a real
    duplicate; it cannot be told apart without the image bytes

NOT checked (image bytes are not in the manifests): content-level or perceptual
near-duplicates, and label noise.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", type=Path, help="Training manifest (JSONL)")
    parser.add_argument("validation", type=Path, help="Validation manifest (JSONL)")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).parents[1] / "metadata" / "labels.json",
        help="labels.json with old_food_classes and pithanet_classes",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON summary here as well")
    return parser.parse_args()


def load(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image = row["images"][0] if isinstance(row["images"], list) else row["images"]
            ref = image if isinstance(image, str) else image["path"]
            label = row["messages"][-1]["content"].strip()
            if not ref or not label:
                raise ValueError(f"{path.name}:{number}: missing image or label")
            rows.append((ref, label))
    return rows


def name_key(ref: str, label: str) -> tuple[str, str]:
    return label, posixpath.basename(ref).lower()


def main() -> int:
    args = parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    old, pitha = set(labels["old_food_classes"]), set(labels["pithanet_classes"])
    allowed = old | pitha
    train, val = load(args.train), load(args.validation)

    train_refs, val_refs = {r for r, _ in train}, {r for r, _ in val}
    by_ref = defaultdict(set)
    for ref, label in train + val:
        by_ref[ref].add(label)
    folder_mismatch = sum(
        1
        for ref, label in train + val
        if label in pitha and posixpath.basename(posixpath.dirname(ref)).lower().replace(" ", "_") != label
    )
    collisions = {name_key(r, l) for r, l in train} & {name_key(r, l) for r, l in val}

    def group(rows):
        counts = Counter(label for _, label in rows)
        return {
            "rows": len(rows),
            "unique_image_refs": len({r for r, _ in rows}),
            "existing_food_rows": sum(c for l, c in counts.items() if l in old),
            "pithanet_rows": sum(c for l, c in counts.items() if l in pitha),
            "classes_present": len(counts),
        }

    summary = {
        "train": group(train),
        "validation": group(val),
        "exact_image_ref_overlap_train_vs_validation": len(train_refs & val_refs),
        "duplicate_refs_within_train": len(train) - len(train_refs),
        "duplicate_refs_within_validation": len(val) - len(val_refs),
        "refs_with_conflicting_labels": sum(1 for v in by_ref.values() if len(v) > 1),
        "labels_outside_label_set": sorted({l for _, l in train + val} - allowed),
        "pithanet_rows_with_folder_label_mismatch": folder_mismatch,
        "same_label_and_filename_in_both_splits": {
            "count": len(collisions),
            "existing_food": sum(1 for l, _ in collisions if l in old),
            "pithanet": sum(1 for l, _ in collisions if l in pitha),
            "note": "name collision or duplicate; undecidable without image bytes",
        },
        "not_checked": ["content or near-duplicate images", "label noise"],
    }
    text = json.dumps(summary, indent=2, sort_keys=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    hard_fail = (
        summary["exact_image_ref_overlap_train_vs_validation"]
        or summary["refs_with_conflicting_labels"]
        or summary["labels_outside_label_set"]
    )
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
