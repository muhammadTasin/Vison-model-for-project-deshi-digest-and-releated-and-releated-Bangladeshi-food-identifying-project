#!/usr/bin/env python3
"""Content-level train/validation overlap audit for the PithaNet classes.

Complements evaluation/audit_manifests.py, which only sees image references. This script
needs the actual image files. For the PithaNet rows of the Stage-4 manifests it computes

  * SHA-256 of every image: byte-identical images across train and validation
  * dHash and pHash (64-bit): near-duplicate pairs across train and validation at a
    Hamming distance threshold
  * for the (label, file name) pairs present in both splits: how many are
    byte-identical and how many have different content

Output is aggregate counts only. No file names, image paths or image copies are written.
Requires Pillow and numpy.

Example:
  python3 evaluation/audit_content.py train.jsonl validation.jsonl \\
      --image-root /path/that/contains/Crop_Resize --anchor Crop_Resize \\
      --output evaluation/reports/stage4_content_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HASH_BITS = 64
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Directory that contains the anchor folder (for example Crop_Resize)")
    parser.add_argument("--anchor", default="Crop_Resize",
                        help="Folder name after which manifest paths are re-rooted under --image-root")
    parser.add_argument("--labels", type=Path,
                        default=Path(__file__).parents[1] / "metadata" / "labels.json")
    parser.add_argument("--threshold", type=int, default=5, help="Max Hamming distance for near-duplicates")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image = row["images"][0] if isinstance(row["images"], list) else row["images"]
            rows.append((image if isinstance(image, str) else image["path"],
                         row["messages"][-1]["content"].strip()))
    return rows


def resolve(ref: str, anchor: str, root: Path) -> Path | None:
    marker = anchor + "/"
    if marker not in ref:
        return None
    return root / anchor / ref.split(marker, 1)[1]


def dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2)
    return m


_DCT = dct_matrix(32)


def to_int(bits: np.ndarray) -> np.uint64:
    value = 0
    for b in bits.flatten():
        value = (value << 1) | int(b)
    return np.uint64(value)


def dhash(img: Image.Image) -> np.uint64:
    small = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float64)
    return to_int(small[:, 1:] > small[:, :-1])


def phash(img: Image.Image) -> np.uint64:
    small = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_DCT @ small @ _DCT.T)[:8, :8]
    return to_int(low > np.median(low.flatten()[1:]))


def hamming(one: np.uint64, many: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(many, one)
    return _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def fingerprint(path: Path):
    data = path.read_bytes()
    with Image.open(path) as img:
        img.load()
        return hashlib.sha256(data).hexdigest(), dhash(img), phash(img)


def pair_stats(val_hashes, train_hashes, threshold, exclude_pairs=None):
    """Near-duplicate statistics for one hash type. Distances are validation x train."""
    tr = np.array(train_hashes, dtype=np.uint64)
    pairs = 0
    val_with_match = 0
    nearest = []
    kept = []
    for vi, h in enumerate(val_hashes):
        d = hamming(np.uint64(h), tr)
        hits = np.nonzero(d <= threshold)[0]
        if exclude_pairs is not None:
            hits = np.array([t for t in hits if (vi, int(t)) not in exclude_pairs], dtype=int)
        pairs += len(hits)
        val_with_match += int(len(hits) > 0)
        nearest.append(int(d.min()))
        kept.append(hits)
    buckets = Counter(
        "0" if n == 0 else "1-%d" % threshold if n <= threshold else "%d-10" % (threshold + 1) if n <= 10 else ">10"
        for n in nearest
    )
    return pairs, val_with_match, dict(buckets), kept


def main() -> int:
    args = parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    pitha = set(labels["pithanet_classes"])
    train = [(r, l) for r, l in load(args.train) if l in pitha]
    val = [(r, l) for r, l in load(args.validation) if l in pitha]

    def fingerprints(rows):
        out, missing, bad = [], 0, 0
        for ref, label in rows:
            path = resolve(ref, args.anchor, args.image_root)
            if path is None or not path.is_file():
                missing += 1
                out.append(None)
                continue
            try:
                out.append(fingerprint(path))
            except Exception:  # undecodable image
                bad += 1
                out.append(None)
        return out, missing, bad

    tr_fp, tr_missing, tr_bad = fingerprints(train)
    va_fp, va_missing, va_bad = fingerprints(val)
    problems = {"train_missing": tr_missing, "train_undecodable": tr_bad,
                "validation_missing": va_missing, "validation_undecodable": va_bad}
    if any(problems.values()):
        print(json.dumps({"error": "some images could not be hashed", **problems}, indent=2))
        return 2

    tr_sha = [f[0] for f in tr_fp]; va_sha = [f[0] for f in va_fp]
    tr_sha_set = set(tr_sha)
    tr_by_sha = defaultdict(list)
    for i, s in enumerate(tr_sha):
        tr_by_sha[s].append(i)
    exact_val_images = sum(1 for s in va_sha if s in tr_sha_set)
    exact_pairs = sum(len(tr_by_sha[s]) for s in va_sha if s in tr_sha_set)
    exact_pair_set = {(vi, ti) for vi, s in enumerate(va_sha) for ti in tr_by_sha.get(s, [])}
    cross_label_exact = sum(1 for vi, ti in exact_pair_set if val[vi][1] != train[ti][1])

    result = {
        "scope": "PithaNet classes only (8)",
        "images": {"train": len(train), "validation": len(val)},
        "images_hashed_ok": {"train": len(tr_fp), "validation": len(va_fp)},
        "sha256": {
            "validation_images_with_byte_identical_train_image": exact_val_images,
            "byte_identical_train_validation_pairs": exact_pairs,
            "byte_identical_pairs_with_different_labels": cross_label_exact,
            "duplicate_groups_within_train": sum(1 for v in tr_by_sha.values() if len(v) > 1),
            "duplicate_groups_within_validation": sum(1 for c in Counter(va_sha).values() if c > 1),
        },
        "near_duplicates": {"threshold_hamming_distance": args.threshold, "hash_bits": HASH_BITS},
    }
    for name, idx in (("dhash", 1), ("phash", 2)):
        v = [f[idx] for f in va_fp]; t = [f[idx] for f in tr_fp]
        pairs, val_with, buckets, kept = pair_stats(v, t, args.threshold)
        pairs_x, val_with_x, _, _ = pair_stats(v, t, args.threshold, exclude_pairs=exact_pair_set)
        same_label = sum(1 for vi, hits in enumerate(kept) for ti in hits if val[vi][1] == train[int(ti)][1])
        result["near_duplicates"][name] = {
            "pairs_within_threshold": pairs,
            "pairs_within_threshold_same_label": same_label,
            "validation_images_with_a_match": val_with,
            "pairs_within_threshold_excluding_byte_identical": pairs_x,
            "validation_images_with_a_match_excluding_byte_identical": val_with_x,
            "nearest_train_distance_per_validation_image": dict(sorted(buckets.items())),
        }

    # same label + same file name in both splits
    key = lambda ref, label: (label, posixpath.basename(ref).lower())
    tr_idx = {key(*train[i]): i for i in range(len(train))}
    va_idx = {key(*val[i]): i for i in range(len(val))}
    shared = sorted(set(tr_idx) & set(va_idx))
    identical = sum(1 for k in shared if tr_fp[tr_idx[k]][0] == va_fp[va_idx[k]][0])
    different = len(shared) - identical
    near = {}
    for name, idx in (("dhash", 1), ("phash", 2)):
        near[name] = sum(
            1 for k in shared
            if tr_fp[tr_idx[k]][0] != va_fp[va_idx[k]][0]
            and bin(int(tr_fp[tr_idx[k]][idx]) ^ int(va_fp[va_idx[k]][idx])).count("1") <= args.threshold
        )
    result["same_label_and_filename_pairs"] = {
        "count": len(shared),
        "byte_identical": identical,
        "different_content": different,
        "different_content_but_within_threshold": near,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
