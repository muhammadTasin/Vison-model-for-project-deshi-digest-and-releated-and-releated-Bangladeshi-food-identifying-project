#!/usr/bin/env python3
"""Score JSONL food predictions with exact, alias-aware label comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GROUND_TRUTH_FIELDS = ("ground_truth", "expected", "label", "labels")
PREDICTION_FIELDS = ("prediction", "predicted", "response", "food_id")


class InputError(ValueError):
    """Raised when prediction data cannot be scored safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Input JSONL predictions")
    parser.add_argument(
        "--aliases",
        type=Path,
        default=Path(__file__).parents[1] / "inference" / "label_aliases.json",
        help="JSON object mapping exact aliases to canonical labels",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/reports/generated")
    )
    return parser.parse_args()


def normalized_token(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def load_aliases(path: Path) -> dict[str, str]:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read alias map {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError("Alias map must be a JSON object")

    result: dict[str, str] = {}
    for raw_alias, raw_target in data.items():
        if not isinstance(raw_alias, str) or not isinstance(raw_target, str):
            raise InputError("Alias keys and values must be strings")
        alias = normalized_token(raw_alias)
        target = normalized_token(raw_target)
        previous = result.get(alias)
        if previous is not None and previous != target:
            raise InputError(f"Conflicting targets for alias {raw_alias!r}")
        result[alias] = target
    return result


def extract_field(record: dict[str, Any], fields: Iterable[str], line: int, role: str) -> str:
    for field in fields:
        if field not in record:
            continue
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise InputError(
                f"Line {line}: {role} field {field!r} must be a non-empty string"
            )
        return value
    choices = ", ".join(fields)
    raise InputError(f"Line {line}: missing {role}; expected one of: {choices}")


def normalize_label(raw: str, aliases: dict[str, str]) -> str:
    """Normalize a complete label or structured food_id, never a substring."""
    candidate = raw.strip()
    if candidate.startswith("{"):
        try:
            payload: Any = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise InputError(f"Invalid JSON prediction object: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("food_id"), str):
            raise InputError("JSON prediction must contain a string food_id")
        candidate = payload["food_id"]
    token = normalized_token(candidate)
    return aliases.get(token, token)


def read_records(path: Path, aliases: dict[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise InputError(f"Cannot open predictions {path}: {exc}") from exc

    with lines:
        for line_number, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            try:
                payload: Any = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise InputError(f"Line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise InputError(f"Line {line_number}: record must be a JSON object")
            expected_raw = extract_field(
                payload, GROUND_TRUTH_FIELDS, line_number, "ground truth"
            )
            predicted_raw = extract_field(
                payload, PREDICTION_FIELDS, line_number, "prediction"
            )
            expected = normalize_label(expected_raw, aliases)
            predicted = normalize_label(predicted_raw, aliases)
            records.append(
                {
                    "line": line_number,
                    "expected_raw": expected_raw,
                    "predicted_raw": predicted_raw,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": expected == predicted,
                }
            )
    if not records:
        raise InputError("Prediction file contains no non-empty records")
    return records


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    wrong: list[dict[str, Any]] = []

    for record in records:
        expected = record["expected"]
        predicted = record["predicted"]
        totals[expected] += 1
        if record["correct"]:
            correct[expected] += 1
        else:
            confusions[(expected, predicted)] += 1
            wrong.append(record)

    per_class = [
        {
            "class": label,
            "correct": correct[label],
            "total": totals[label],
            "accuracy": correct[label] / totals[label],
            "accuracy_percent": round(100 * correct[label] / totals[label], 2),
        }
        for label in sorted(totals)
    ]
    confusion_rows = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(
            confusions.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    total_correct = sum(correct.values())
    report = {
        "total": len(records),
        "correct": total_correct,
        "incorrect": len(records) - total_correct,
        "accuracy": total_correct / len(records),
        "accuracy_percent": round(100 * total_correct / len(records), 2),
        "classes": len(totals),
        "per_class": per_class,
        "confusion_pairs": confusion_rows,
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(
        output_dir / "per_class.csv",
        ["class", "correct", "total", "accuracy", "accuracy_percent"],
        per_class,
    )
    write_csv(
        output_dir / "wrong_predictions.csv",
        ["line", "expected_raw", "predicted_raw", "expected", "predicted", "correct"],
        wrong,
    )
    write_csv(
        output_dir / "confusion_pairs.csv",
        ["expected", "predicted", "count"],
        confusion_rows,
    )
    return report


def main() -> int:
    args = parse_args()
    try:
        aliases = load_aliases(args.aliases)
        records = read_records(args.predictions, aliases)
        report = score(records, args.output_dir)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
