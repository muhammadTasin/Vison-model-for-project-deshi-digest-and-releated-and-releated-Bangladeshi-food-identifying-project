#!/usr/bin/env python3
"""Render dependency-free SVG charts for the repository README.

The script reads the compact, versioned Stage-3 reports under
``evaluation/reports`` and writes deterministic SVG assets. It intentionally
uses only the Python standard library so maintainers can regenerate the charts
without a plotting stack.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evaluation" / "reports"
OUTPUT = ROOT / "assets" / "readme"

INK = "#172033"
MUTED = "#667085"
GRID = "#D9E0EA"
PANEL = "#F8FAFC"
BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
BLUE_OPEN = "#DBEAFE"
GOLD = "#D99A00"
GOLD_LIGHT = "#F7D774"
ORANGE = "#D65A31"
ORANGE_OPEN = "#FDE7DF"
WHITE = "#FFFFFF"

CLASSES = (
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

PRIMARY_WEAK = {"morog_polao", "roshmalai", "kacha_golla"}
SECONDARY_WEAK = {"bakorkhani", "fuchka", "nehari", "potato_bhorta"}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_document(width: int, height: int, title: str, description: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
  viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title>
  <desc id="desc">{esc(description)}</desc>
  <style>
    text {{ font-family: Inter, "Segoe UI", Arial, sans-serif; fill: {INK}; }}
    .title {{ font-size: 30px; font-weight: 700; }}
    .subtitle {{ font-size: 16px; fill: {MUTED}; }}
    .label {{ font-size: 15px; font-weight: 600; }}
    .small {{ font-size: 13px; fill: {MUTED}; }}
    .value {{ font-size: 16px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .mono {{ font-family: "SFMono-Regular", Consolas, monospace; font-variant-numeric: tabular-nums; }}
  </style>
  <rect width="100%" height="100%" fill="{WHITE}"/>
  {body}
</svg>
'''


def text(x: float, y: float, value: object, css: str = "", anchor: str = "start", **attrs: object) -> str:
    extra = " ".join(f'{key.replace("_", "-")}="{esc(val)}"' for key, val in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" text-anchor="{anchor}" {extra}>{esc(value)}</text>'


def rect(x: float, y: float, width: float, height: float, fill: str, **attrs: object) -> str:
    extra = " ".join(f'{key.replace("_", "-")}="{esc(val)}"' for key, val in attrs.items())
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" {extra}/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID, width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def load_reports() -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    report = json.loads(
        (REPORTS / "stage3_checkpoint348_report.json").read_text(encoding="utf-8")
    )
    with (REPORTS / "stage3_checkpoint348_per_class.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        per_class = list(csv.DictReader(handle))
    with (REPORTS / "stage3_checkpoint348_confusion_pairs.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        confusions = list(csv.DictReader(handle))

    for row in per_class:
        row["correct"] = int(row["correct"])
        row["total"] = int(row["total"])
        row["accuracy_percent"] = float(row["accuracy_percent"])
    for row in confusions:
        row["count"] = int(row["count"])
    return report, per_class, confusions


def validate(report: dict[str, object], per_class: list[dict[str, object]], confusions: list[dict[str, object]]) -> None:
    labels = {str(row["class"]) for row in per_class}
    if labels != set(CLASSES):
        raise ValueError("Per-class report does not match the 19 canonical classes")
    if sum(int(row["total"]) for row in per_class) != 95:
        raise ValueError("Per-class totals must sum to 95")
    if sum(int(row["correct"]) for row in per_class) != 83:
        raise ValueError("Per-class correct counts must sum to 83")
    if sum(int(row["count"]) for row in confusions) != 12:
        raise ValueError("Confusion counts must sum to 12")
    if int(report["total"]) != 95 or int(report["correct"]) != 83:
        raise ValueError("Overall report must record 83/95")


def render_stage_comparison(report: dict[str, object]) -> str:
    width, height = 1200, 560
    left, right, top = 220, 1080, 150
    plot_width = right - left
    stages = [
        ("Stage 1", 83.16, "79 / 95", BLUE_LIGHT),
        ("Stage 2", 82.11, "78 / 95", BLUE_OPEN),
        ("Stage 3", float(report["accuracy_percent"]), "83 / 95", GOLD),
    ]
    parts = [
        text(70, 62, "Training-stage benchmark comparison", "title"),
        text(70, 94, "Accuracy on the same fixed internal benchmark · n=95 images · 19 classes", "subtitle"),
    ]
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100
        parts.append(line(x, top - 18, x, 430, GRID))
        parts.append(text(x, 455, f"{tick}%", "small mono", "middle"))
    for index, (stage, accuracy, correct, color) in enumerate(stages):
        y = top + index * 100
        parts.append(text(left - 24, y + 31, stage, "label", "end"))
        parts.append(rect(left, y, plot_width, 48, PANEL, rx=8))
        parts.append(rect(left, y, plot_width * accuracy / 100, 48, color, rx=8))
        parts.append(text(left + 18, y + 31, correct, "value mono"))
        parts.append(text(right + 24, y + 31, f"{accuracy:.2f}%", "value mono"))
    parts.extend(
        [
            rect(70, 492, 12, 12, GOLD, rx=2),
            text(92, 503, "Stage 3 is +4.21 pp vs Stage 1 and +5.26 pp vs Stage 2", "small"),
            text(1130, 503, "Source: versioned evaluation reports", "small", "end"),
        ]
    )
    return svg_document(
        width,
        height,
        "Training-stage benchmark comparison",
        "Horizontal bars compare Stage 1 at 83.16 percent, Stage 2 at 82.11 percent, and Stage 3 at 87.37 percent on 95 images.",
        "\n".join(parts),
    )


def render_per_class(per_class: list[dict[str, object]]) -> str:
    width, height = 1200, 1120
    left, right, top = 310, 1030, 180
    plot_width = right - left
    rows = sorted(per_class, key=lambda row: (float(row["accuracy_percent"]), str(row["class"])))
    parts = [
        text(70, 62, "Stage-3 accuracy by food class", "title"),
        text(70, 94, "Normalized exact-label accuracy · 5 benchmark images per class · n=95 total", "subtitle"),
        rect(70, 124, 12, 12, ORANGE, rx=2),
        text(92, 135, "P · Primary Stage-4 target", "small"),
        rect(260, 124, 12, 12, GOLD_LIGHT, rx=2),
        text(282, 135, "S · Secondary review class", "small"),
        rect(490, 124, 12, 12, BLUE, rx=2),
        text(512, 135, "Other supported class", "small"),
    ]
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100
        parts.append(line(x, top - 12, x, 1040, GRID))
        parts.append(text(x, 1065, f"{tick}%", "small mono", "middle"))
    for index, row in enumerate(rows):
        label = str(row["class"])
        accuracy = float(row["accuracy_percent"])
        correct = int(row["correct"])
        total = int(row["total"])
        y = top + index * 44
        if label in PRIMARY_WEAK:
            color, tag = ORANGE, "P"
        elif label in SECONDARY_WEAK:
            color, tag = GOLD_LIGHT, "S"
        else:
            color, tag = BLUE, ""
        parts.append(text(left - 22, y + 23, label, "label mono", "end"))
        parts.append(rect(left, y, plot_width, 28, PANEL, rx=5))
        parts.append(rect(left, y, plot_width * accuracy / 100, 28, color, rx=5))
        suffix = f" · {tag}" if tag else ""
        parts.append(
            text(right + 18, y + 22, f"{correct}/{total} · {accuracy:.0f}%{suffix}", "value mono")
        )
    parts.append(text(1130, 1098, "Source: Stage-3 per-class report", "small", "end"))
    return svg_document(
        width,
        height,
        "Stage-3 accuracy by food class",
        "Horizontal bars rank 19 food classes by exact-label accuracy. Morog polao and roshmalai are lowest at 40 percent; kacha golla is 60 percent.",
        "\n".join(parts),
    )


def blend(open_hex: str, base_hex: str, fraction: float) -> str:
    def rgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))

    start, end = rgb(open_hex), rgb(base_hex)
    mixed = tuple(round(a + (b - a) * fraction) for a, b in zip(start, end))
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def render_confusion_matrix(
    per_class: list[dict[str, object]], confusions: list[dict[str, object]]
) -> str:
    width, height = 1320, 1260
    cell = 42
    grid_x, grid_y = 390, 170
    matrix = {(expected, predicted): 0 for expected in CLASSES for predicted in CLASSES}
    for row in per_class:
        label = str(row["class"])
        matrix[(label, label)] = int(row["correct"])
    for row in confusions:
        matrix[(str(row["expected"]), str(row["predicted"]))] = int(row["count"])

    parts = [
        text(70, 62, "Stage-3 confusion matrix", "title"),
        text(70, 94, "Rows are expected labels; columns are predictions · exact counts · n=95", "subtitle"),
        rect(890, 58, 16, 16, BLUE, rx=2),
        text(916, 72, "correct", "small"),
        rect(990, 58, 16, 16, ORANGE, rx=2),
        text(1016, 72, "error", "small"),
    ]
    for row_index, expected in enumerate(CLASSES):
        y = grid_y + row_index * cell
        parts.append(text(grid_x - 18, y + 27, expected, "small mono", "end"))
        for column_index, predicted in enumerate(CLASSES):
            x = grid_x + column_index * cell
            count = matrix[(expected, predicted)]
            diagonal = expected == predicted
            if count == 0:
                fill = PANEL
                label_color = MUTED
            elif diagonal:
                fill = blend(BLUE_OPEN, BLUE, count / 5)
                label_color = WHITE if count >= 4 else INK
            else:
                fill = blend(ORANGE_OPEN, ORANGE, min(1.0, count / 2))
                label_color = WHITE if count >= 2 else INK
            parts.append(rect(x, y, cell - 2, cell - 2, fill, rx=3))
            parts.append(
                f'<text x="{x + (cell - 2) / 2:.1f}" y="{y + 27:.1f}" '
                f'class="small mono" text-anchor="middle" style="fill:{label_color};font-weight:{700 if count else 400}">{count}</text>'
            )
    for column_index, predicted in enumerate(CLASSES):
        x = grid_x + column_index * cell + 25
        y = grid_y + len(CLASSES) * cell + 20
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="small mono" '
            f'text-anchor="end" transform="rotate(-55 {x:.1f} {y:.1f})">{esc(predicted)}</text>'
        )
    parts.extend(
        [
            text(95, grid_y + 380, "EXPECTED LABEL", "small", "middle", transform=f"rotate(-90 95 {grid_y + 380})"),
            text(grid_x + 390, 1215, "PREDICTED LABEL", "small", "middle"),
            text(1240, 1215, "83 correct · 12 incorrect · 87.37%", "value mono", "end"),
        ]
    )
    return svg_document(
        width,
        height,
        "Stage-3 confusion matrix",
        "A 19 by 19 confusion matrix. The diagonal contains 83 correct predictions and off-diagonal cells contain 12 errors, including two roshmalai predictions as roshogolla.",
        "\n".join(parts),
    )


def write_assets(charts: Iterable[tuple[str, str]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, content in charts:
        (OUTPUT / filename).write_text(content, encoding="utf-8")
        print(f"wrote {OUTPUT / filename}")


def main() -> int:
    report, per_class, confusions = load_reports()
    validate(report, per_class, confusions)
    write_assets(
        (
            ("stage_comparison.svg", render_stage_comparison(report)),
            ("per_class_accuracy.svg", render_per_class(per_class)),
            ("confusion_matrix.svg", render_confusion_matrix(per_class, confusions)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
