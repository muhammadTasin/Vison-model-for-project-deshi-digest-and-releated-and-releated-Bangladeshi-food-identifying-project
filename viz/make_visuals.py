#!/usr/bin/env python3
"""Build LinkedIn-ready result visuals for the Deshi Digest Vision Model.

Run from anywhere:   viz/.venv/bin/python viz/make_visuals.py

Reads ONLY:
  evaluation/reports/*, metadata/*, README.md, docs/*     (versioned in the repo)
  viz/data/stage4_predictions_pairs.csv                    (true,pred for the 996 validation images)
  viz/data/trainer_state_loss.csv                          (training log of checkpoint-645)
Writes PNG (dpi 220) + SVG into viz/out/.

A verification gate runs first. If the per-sample predictions do not reproduce the
published numbers, the script stops and prints the discrepancy instead of plotting.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "viz"
OUT = VIZ / "out"
REPORTS = ROOT / "evaluation" / "reports"
DPI = 220

# ---------- palette ----------
BG = "#0B0F0E"
PANEL = "#131A18"
PANEL_EDGE = "#22302C"
TEXT = "#F2F0E9"
MUTED = "#8E9692"
FAINT = "#46504C"
GREEN = "#006A4E"
GREEN_HI = "#2FBF8F"
RED = "#F42A41"
EXISTING = "#CFCBBF"

CMAP = LinearSegmentedColormap.from_list(
    "deshi",
    [(0.0, "#111917"), (0.15, "#0F4A39"), (0.4, GREEN), (0.7, GREEN_HI), (1.0, "#C9F5E4")],
)

FOOTER = "Held-out validation split, 996 images, 27 classes. Research prototype."
CREDIT = (
    "PithaNet dataset, courtesy of the authors at Daffodil International "
    "University (used with permission)"
)


# ---------- fonts ----------
def pick_font():
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Inter", "Noto Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "DejaVu Sans"


FONT = pick_font()
plt.rcParams.update(
    {
        "font.family": [FONT, "DejaVu Sans"],
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": PANEL_EDGE,
        "svg.fonttype": "path",
    }
)


def px(n):
    """pixels at DPI -> points"""
    return n * 72.0 / DPI


def pretty(c):
    return c.replace("_", " ")


def body_pt(w_in):
    """Font size (pt) that renders as 22 px when the PNG is shown 1080 px wide."""
    return 22.0 * w_in * 72.0 / 1080.0


def text_w(s, size, weight="normal"):
    """Width of s in the same unit as size (pt -> points, px -> pixels)."""
    fp = FontProperties(family=[FONT, "DejaVu Sans"], weight=weight)
    return TextPath((0, 0), s, size=size, prop=fp).get_extents().width


def wrap(s, size, max_w, weight="normal"):
    """Greedy word wrap by measured width. size and max_w share a unit."""
    lines, cur = [], ""
    for word in s.split():
        trial = word if not cur else cur + " " + word
        if text_w(trial, size, weight) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(a + (b - a) * t)


# ---------- data ----------
def read_text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def load_data():
    labels = json.load(open(ROOT / "metadata" / "labels.json"))
    old, pitha = labels["old_food_classes"], labels["pithanet_classes"]
    classes = old + pitha
    assert len(old) == 19 and len(pitha) == 8 and len(classes) == 27

    report = json.load(open(REPORTS / "stage4_checkpoint645_report.json"))
    summary = json.load(open(REPORTS / "stage4_summary.json"))
    comparison = json.load(open(REPORTS / "stage4_checkpoint_comparison.json"))
    conf_summary = json.load(open(REPORTS / "stage4_checkpoint645_confusion_summary.json"))
    train_cfg = json.load(open(ROOT / "metadata" / "training_config.json"))
    audit = json.load(open(REPORTS / "stage4_manifest_audit.json"))
    content = json.load(open(REPORTS / "stage4_content_audit.json"))
    per_class_csv = {
        r["class"]: (int(r["correct"]), int(r["total"]), float(r["accuracy"]))
        for r in csv.DictReader(open(REPORTS / "stage4_checkpoint645_per_class.csv"))
    }
    pairs = [
        (r["true"], r["pred"])
        for r in csv.DictReader(open(VIZ / "data" / "stage4_predictions_pairs.csv"))
    ]
    loss = [
        {k: float(v) if v != "" else None for k, v in r.items()}
        for r in csv.DictReader(open(VIZ / "data" / "trainer_state_loss.csv"))
    ]
    return dict(
        old=old, pitha=pitha, classes=classes, report=report, summary=summary,
        comparison=comparison, conf_summary=conf_summary, per_class_csv=per_class_csv,
        pairs=pairs, loss=loss, train_cfg=train_cfg, audit=audit, content=content,
    )


def confusion_counts(d):
    idx = {c: i for i, c in enumerate(d["classes"])}
    cm = np.zeros((27, 27), dtype=int)
    for t, p in d["pairs"]:
        cm[idx[t], idx[p]] += 1
    return cm


def class_metrics(cm, classes):
    """Per-class precision / recall / F1 / support from the confusion matrix."""
    rows = []
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append(dict(cls=c, precision=p, recall=r, f1=f, support=support, tp=tp))
    return rows


def pitha_matrix(cm):
    """8 pitha rows x (8 pitha columns + 'other food class')."""
    m = np.zeros((8, 9), dtype=int)
    m[:, :8] = cm[19:, 19:]
    m[:, 8] = cm[19:, :19].sum(1)
    return m


def top_confusions(cm, classes, k=3):
    """Top-k confusions by count; ties at the cut-off are all returned. (rank, count, i, j)"""
    items = []
    for i in range(27):
        for j in range(27):
            if i != j and cm[i, j] > 0:
                items.append((int(cm[i, j]), i, j))
    items.sort(key=lambda t: (-t[0], t[1], t[2]))
    cutoff = items[k - 1][0]
    ranked, rank, prev = [], 0, None
    for cnt, i, j in items:
        if cnt < cutoff:
            break
        if cnt != prev:
            rank = len(ranked) + 1
            prev = cnt
        ranked.append((rank, cnt, i, j))
    return ranked


def rank_label(rank, ranked):
    return f"{rank}{'=' if sum(1 for r in ranked if r[0] == rank) > 1 else ''}"


def verify(d):
    """Cross-check every published number against numbers recomputed from predictions."""
    classes = d["classes"]
    idx = {c: i for i, c in enumerate(classes)}
    problems = [f"unknown label in predictions: {t!r} -> {p!r}"
                for t, p in d["pairs"] if t not in idx or p not in idx]
    if problems:
        for p in problems:
            print("  ", p)
        sys.exit("Stopping: unknown labels in predictions.")
    cm = confusion_counts(d)
    n = int(cm.sum())
    correct = int(np.trace(cm))
    totals = cm.sum(1)
    diag = np.diag(cm)
    old_c = int(diag[:19].sum()); old_n = int(totals[:19].sum())
    pit_c = int(diag[19:].sum()); pit_n = int(totals[19:].sum())
    macro = float(np.mean(diag / totals) * 100)

    rep = d["report"]
    checks = [
        ("total images", n, rep["validation_total"]),
        ("overall correct", correct, rep["overall"]["correct"]),
        ("overall total", n, rep["overall"]["total"]),
        ("existing correct", old_c, rep["old_food"]["correct"]),
        ("existing total", old_n, rep["old_food"]["total"]),
        ("pithanet correct", pit_c, rep["pithanet"]["correct"]),
        ("pithanet total", pit_n, rep["pithanet"]["total"]),
        ("overall % (2dp)", round(correct / n * 100, 2), d["summary"]["overall_accuracy"]),
        ("macro % (2dp)", round(macro, 2), d["summary"]["macro_accuracy"]),
        ("ckpt-645 overall == summary", d["comparison"]["checkpoint-645"]["overall_accuracy"],
         d["summary"]["overall_accuracy"]),
        ("ckpt-645 macro == summary", d["comparison"]["checkpoint-645"]["macro_accuracy"],
         d["summary"]["macro_accuracy"]),
    ]
    for c in classes:
        rc, rt, _ = d["per_class_csv"][c]
        jc = rep["per_class"][c]
        checks.append((f"per-class {c} (preds vs csv)", (int(diag[idx[c]]), int(totals[idx[c]])), (rc, rt)))
        checks.append((f"per-class {c} (csv vs json)", (rc, rt), (jc["correct"], jc["total"])))
    for key, cnt in d["conf_summary"].items():
        a, b = [s.strip() for s in key.split("->")]
        checks.append((f"confusion {key}", int(cm[idx[a], idx[b]]), cnt))

    # --- round 2 checks ---
    metrics = class_metrics(cm, classes)
    for m in metrics:
        rc, rt, acc = d["per_class_csv"][m["cls"]]
        checks.append((f"recall {m['cls']} == csv accuracy", abs(m["recall"] * 100 - acc) < 1e-9, True))
        checks.append((f"support {m['cls']} == csv total", m["support"], rt))
    m8 = pitha_matrix(cm)
    for i, c in enumerate(d["pitha"]):
        checks.append((f"8x8 row sum {c} == support", int(m8[i].sum()), d["per_class_csv"][c][1]))
    checks.append(("8x8 diagonal sum == pithanet correct", int(np.trace(m8[:, :8])), rep["pithanet"]["correct"]))
    checks.append(("8x8 total == pithanet total", int(m8.sum()), rep["pithanet"]["total"]))
    ranked10 = top_confusions(cm, classes, 10)
    ranked_keys = {f"{classes[i]} -> {classes[j]}": cnt for _, cnt, i, j in ranked10}
    for key, cnt in d["conf_summary"].items():
        checks.append((f"top-10 contains {key}", ranked_keys.get(key), cnt))
    checks.append(("macro recall == macro accuracy (2dp)",
                   round(float(np.mean([m["recall"] for m in metrics])) * 100, 2),
                   d["summary"]["macro_accuracy"]))
    # numbers used by chart 09 must appear in the docs they are attributed to
    docs = {rel: read_text(rel) for rel in
            ("README.md", "docs/DATASETS.md", "docs/TRAINING_HISTORY.md", "docs/LIMITATIONS.md")}
    s = d["summary"]
    doc_facts = [
        ("docs/DATASETS.md", f"{s['pithanet_unique_training_samples']:,} clean unique PithaNet"),
        ("docs/DATASETS.md", f"{s['old_food_rehearsal_samples']:,}"),
        ("README.md", f"{s['training_samples']:,} samples"),
        ("docs/TRAINING_HISTORY.md", "rank 4, alpha 16, dropout 0.05"),
        ("docs/TRAINING_HISTORY.md", "1.5 epochs"),
        ("README.md", f"{rep['overall']['correct']}/{rep['overall']['total']}"),
        ("README.md", f"{rep['old_food']['correct']}/{rep['old_food']['total']}"),
        ("README.md", f"{rep['pithanet']['correct']}/{rep['pithanet']['total']}"),
        ("docs/LIMITATIONS.md", "label noise was not checked"),
        ("docs/LIMITATIONS.md", f"{d['content']['near_duplicates']['dhash']['validation_images_with_a_match']} PithaNet validation images"),
        ("docs/TRAINING_HISTORY.md", f"**{d['audit']['same_label_and_filename_in_both_splits']['count']}** (all PithaNet"),
        ("docs/TRAINING_HISTORY.md", f"**0 / {d['content']['same_label_and_filename_pairs']['count']}**"),
    ]
    for rel, needle in doc_facts:
        checks.append((f"{rel} contains '{needle}'", needle in docs[rel], True))
    au = d["audit"]
    checks += [
        ("audit train rows", au["train"]["rows"], s["training_samples"]),
        ("audit validation rows", au["validation"]["rows"], rep["validation_total"]),
        ("audit exact overlap == 0", au["exact_image_ref_overlap_train_vs_validation"], 0),
        ("audit conflicting labels == 0", au["refs_with_conflicting_labels"], 0),
        ("audit val pithanet rows", au["validation"]["pithanet_rows"], rep["pithanet"]["total"]),
        ("audit val existing rows", au["validation"]["existing_food_rows"], rep["old_food"]["total"]),
    ]
    cn = d["content"]
    checks += [
        ("content audit train images", cn["images"]["train"], s["pithanet_unique_training_samples"]),
        ("content audit validation images", cn["images"]["validation"], rep["pithanet"]["total"]),
        ("content audit pairs == manifest same-name pairs",
         cn["same_label_and_filename_pairs"]["count"], au["same_label_and_filename_in_both_splits"]["count"]),
        ("content audit same-name identical + different == count",
         cn["same_label_and_filename_pairs"]["byte_identical"] + cn["same_label_and_filename_pairs"]["different_content"],
         cn["same_label_and_filename_pairs"]["count"]),
    ]
    # linkedin_ready texts must carry the audit numbers and never overclaim
    nd = cn["near_duplicates"]["dhash"]["validation_images_with_a_match"]
    k_, n_ = rep["overall"]["correct"], rep["overall"]["total"]
    lo, hi = (k_ - nd) / (n_ - nd) * 100, k_ / (n_ - nd) * 100
    must = [f"{lo:.2f}%", f"{hi:.2f}%", f"{cn['images']['train']:,}", str(cn["images"]["validation"]),
            "existing-food", FOOTER]
    for name in ("post_en.txt", "SHOWCASE.md"):
        text = (ROOT / "linkedin_ready" / name).read_text(encoding="utf-8")
        for needle in must:
            checks.append((f"linkedin_ready/{name} contains '{needle}'", needle in text, True))
    for name in ("post.txt", "post_en.txt", "SHOWCASE.md"):
        text = (ROOT / "linkedin_ready" / name).read_text(encoding="utf-8")
        low = text.lower()
        for banned in ("leakage-free", "leakage free", "no leakage", "independent"):
            checks.append((f"linkedin_ready/{name} avoids '{banned}'", banned in low, False))
        for needle in (f"{lo:.2f}%", f"{hi:.2f}%", f"{cn['images']['train']:,}"):
            checks.append((f"linkedin_ready/{name} has number '{needle}'", needle in text, True))
    tc = d["train_cfg"]
    checks.append(("training_config rank/alpha/epochs/lr == docs",
                   (tc["lora_rank"], tc["lora_alpha"], tc["training"]["epochs"], tc["training"]["learning_rate"]),
                   (4, 16, 1.5, 2e-05)))

    print("Verification gate (published vs recomputed from 996 predictions)")
    bad = 0
    for name, got, want in checks:
        ok = got == want
        bad += not ok
        if not ok:
            print(f"  MISMATCH {name}: recomputed={got} published={want}")
    print(f"  {len(checks) - bad}/{len(checks)} checks match")
    extras = [(classes[i], classes[j], cnt) for _, cnt, i, j in ranked10
              if f"{classes[i]} -> {classes[j]}" not in d["conf_summary"]]
    if extras:
        print("  note: pairs tied with 10th place but not in the repo confusion summary:",
              ", ".join(f"{a} -> {b} ({c})" for a, b, c in extras))
    if bad:
        sys.exit("Stopping: predictions do not reproduce the published numbers. No visuals drawn.")
    return cm, diag, totals, macro


# ---------- figure helpers ----------
LH = 1.4  # line-height factor


def new_fig(w_in, h_in):
    fig = plt.figure(figsize=(w_in, h_in), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def canvas(fig):
    """Axes covering the whole figure with inch coordinates, y pointing down."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.patch.set_alpha(0)
    return ax


def header_lines(w_in, subtitle):
    fs = body_pt(w_in)
    lines = []
    for para in subtitle.split("\n"):
        lines += wrap(para, fs, (w_in - 1.0) * 72)
    return lines, fs


def header_h(w_in, subtitle):
    lines, fs = header_lines(w_in, subtitle)
    return 0.92 + len(lines) * fs * LH / 72 + 0.3


def header(fig, title, subtitle, margin=0.5):
    w, h = fig.get_size_inches()
    lines, fs = header_lines(w, subtitle)
    fig.text(margin / w, 1 - 0.40 / h, title, fontsize=21, fontweight="bold",
             color=TEXT, ha="left", va="top")
    for i, ln in enumerate(lines):
        fig.text(margin / w, 1 - (0.92 + i * fs * LH / 72) / h, ln, fontsize=fs, color=MUTED,
                 ha="left", va="top")


def footer_lines(w_in, source, extra=None):
    fs = body_pt(w_in)
    out = [FOOTER]
    if extra:
        out += wrap(extra, fs, (w_in - 1.0) * 72)
    out += wrap("Source: " + source, fs, (w_in - 1.0) * 72)
    return out, fs


def footer_h(w_in, source, extra=None):
    lines, fs = footer_lines(w_in, source, extra)
    return 0.3 + len(lines) * fs * LH / 72 + 0.15


def footer(fig, source, extra=None, margin=0.5):
    w, h = fig.get_size_inches()
    lines, fs = footer_lines(w, source, extra)
    lh = fs * LH / 72
    for i, ln in enumerate(lines):
        y = h - 0.3 - (len(lines) - 1 - i) * lh
        fig.text(margin / w, 1 - y / h, ln, fontsize=fs, color=MUTED, ha="left", va="center")


def save(fig, name, svg=True):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    if svg:
        fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("  wrote", f"viz/out/{name}.png", "+ svg" if svg else "")


# ---------- 01: full confusion heatmap (detail chart) ----------
def chart_confusion(d, cm, totals):
    classes = d["classes"]
    norm_cm = cm / totals[:, None]
    W = 8.2
    sub = ("Row-normalized confusion matrix, 27 classes. Each row is the true class; "
           "brighter = larger share of that class predicted as the column class.")
    src = "viz/data/stage4_predictions_pairs.csv (per-sample predictions), verified against evaluation/reports/*"
    extra = "Colour scale is square-root so rare errors stay visible. Detail chart: see 07 and 08 for counts."
    hh, fh = header_h(W, sub), footer_h(W, src, extra)
    size, left = 5.75, 2.05
    top_y = hh + 0.55
    xlab_end = top_y + size + 1.55
    call_h = 1.85
    call_y = xlab_end + 0.65
    H = call_y + call_h + 0.35 + fh
    fig = new_fig(W, H)
    header(fig, "Where the model gets confused", sub)
    ax = fig.add_axes([left / W, 1 - (top_y + size) / H, size / W, size / H])
    ax.imshow(norm_cm, cmap=CMAP, norm=PowerNorm(0.5, vmin=0, vmax=1), aspect="equal",
              interpolation="nearest")
    ax.set_xticks(range(27)); ax.set_yticks(range(27))
    names = [pretty(c) for c in classes]
    ax.set_xticklabels(names, rotation=90, fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    for i in range(27):
        col = GREEN_HI if i >= 19 else TEXT
        ax.get_xticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_color(col)
    ax.tick_params(length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(18.5, color=TEXT, lw=0.7, alpha=0.55)
    ax.axvline(18.5, color=TEXT, lw=0.7, alpha=0.55)
    fig.text((left + size / 2) / W, 1 - (xlab_end + 0.1) / H, "Predicted class",
             fontsize=body_pt(W), color=MUTED, ha="center", va="top")
    fig.text(0.45 / W, 1 - (top_y + size / 2) / H, "True class", fontsize=body_pt(W),
             color=MUTED, ha="center", va="center", rotation=90)
    ax.text(9, -1.2, "existing foods (19)", ha="center", va="bottom", fontsize=body_pt(W), color=TEXT)
    ax.text(23, -1.2, "PithaNet (8)", ha="center", va="bottom", fontsize=body_pt(W), color=GREEN_HI)

    ranked = top_confusions(cm, classes, 3)
    for rank, cnt, i, j in ranked:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=RED, lw=1.6))
        ax.text(j + 0.85, i, str(rank), color=RED, fontsize=10, fontweight="bold",
                ha="left", va="center")

    cv = canvas(fig)
    cx, cw = 0.5, W - 1.0
    cv.add_patch(FancyBboxPatch((cx, call_y), cw, call_h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=PANEL, ec=PANEL_EDGE, lw=1))
    cv.add_patch(Rectangle((cx, call_y), 0.07, call_h, fc=RED, ec="none"))
    fs = body_pt(W)
    cv.text(cx + 0.28, call_y + 0.2, "Most frequent mistakes (red boxes above)", fontsize=fs,
            color=MUTED, ha="left", va="top")
    for n, (rank, cnt, i, j) in enumerate(ranked):
        yy = call_y + 0.62 + n * fs * 1.55 / 72
        cv.text(cx + 0.28, yy, rank_label(rank, ranked), fontsize=fs, color=RED, fontweight="bold",
                ha="left", va="top")
        cv.text(cx + 0.72, yy, f"{pretty(classes[i])}  →  {pretty(classes[j])}", fontsize=fs,
                color=TEXT, ha="left", va="top")
        cv.text(cx + cw - 0.25, yy, f"{cnt} of {int(totals[i])} images", fontsize=fs, color=MUTED,
                ha="right", va="top")
    footer(fig, src, extra)
    save(fig, "01_confusion_heatmap")


# ---------- 02: per-class accuracy ----------
def chart_per_class(d, diag, totals):
    classes = d["classes"]
    acc = {c: diag[i] / totals[i] * 100 for i, c in enumerate(classes)}
    overall = d["summary"]["overall_accuracy"]
    macro_pub = d["summary"]["macro_accuracy"]
    xmin = math.floor((min(acc.values()) - 4) / 5) * 5
    W = 7.6
    sub = (f"Share of each class's validation images identified correctly. "
           f"Axis starts at {xmin}%; dots show position, not bar length.")
    src = "evaluation/reports/stage4_checkpoint645_per_class.csv, stage4_summary.json"
    extra = "Macro = unweighted mean of the 27 per-class accuracies; overall = 905/996."
    hh, fh = header_h(W, sub), footer_h(W, src, extra)
    row = 0.275
    plot_top = hh + 0.75
    plot_h = 27 * row
    leg_y = plot_top + plot_h + 0.75
    H = leg_y + 0.5 + fh
    fig = new_fig(W, H)
    header(fig, "Accuracy by class", sub)
    ax = fig.add_axes([2.15 / W, 1 - (plot_top + plot_h) / H, 3.85 / W, plot_h / H])
    order = sorted(classes, key=lambda c: (-acc[c], c))
    ypos = {c: 26 - i for i, c in enumerate(order)}
    fs = body_pt(W)
    for c in classes:
        col = GREEN_HI if c in d["pitha"] else EXISTING
        ax.hlines(ypos[c], xmin, acc[c], color=col, alpha=0.30, lw=2.2)
        ax.scatter(acc[c], ypos[c], s=75, color=col, zorder=3, edgecolor=BG, linewidth=0.8)
        k, n = int(diag[classes.index(c)]), int(totals[classes.index(c)])
        ax.text(101.2, ypos[c], f"{acc[c]:.1f}%", ha="left", va="center", fontsize=fs,
                color=col, fontweight="bold")
        ax.text(109.2, ypos[c], f"{k}/{n}", ha="left", va="center", fontsize=fs, color=MUTED)
    ax.axvline(overall, color=TEXT, lw=1.1, ls=(0, (4, 3)), alpha=0.9, zorder=1)
    ax.axvline(macro_pub, color=GREEN_HI, lw=1.1, ls=(0, (4, 3)), alpha=0.9, zorder=1)
    ax.text(overall + 0.3, 27.1, f"overall\n{overall:.2f}%", ha="left", va="bottom",
            fontsize=fs, color=TEXT, linespacing=1.15)
    ax.text(macro_pub - 0.3, 27.1, f"macro\n{macro_pub:.2f}%", ha="right", va="bottom",
            fontsize=fs, color=GREEN_HI, linespacing=1.15)
    ax.set_xlim(xmin, 100.8)
    ax.set_ylim(-0.7, 26.7)
    ax.set_yticks([ypos[c] for c in classes])
    ax.set_yticklabels([pretty(c) for c in classes], fontsize=fs)
    for t, c in zip(ax.get_yticklabels(), classes):
        t.set_color(GREEN_HI if c in d["pitha"] else TEXT)
    ticks = [t for t in range(60, 101, 10) if t >= xmin]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=fs)
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    cv = canvas(fig)
    cv.text(2.15, leg_y, "●", color=EXISTING, fontsize=fs + 1, ha="left", va="center")
    cv.text(2.4, leg_y, "existing foods (19)", color=TEXT, fontsize=fs, ha="left", va="center")
    cv.text(4.5, leg_y, "●", color=GREEN_HI, fontsize=fs + 1, ha="left", va="center")
    cv.text(4.75, leg_y, "PithaNet (8)", color=GREEN_HI, fontsize=fs, ha="left", va="center")
    footer(fig, src, extra)
    save(fig, "02_per_class_accuracy")


# ---------- 03: checkpoint selection ----------
def chart_checkpoints(d):
    comp = d["comparison"]
    cks = ["checkpoint-500", "checkpoint-600", "checkpoint-645"]
    ov = [comp[c]["overall_accuracy"] for c in cks]
    mc = [comp[c]["macro_accuracy"] for c in cks]
    pit = {comp[c]["pithanet_accuracy"] for c in cks}
    W = 7.4
    fs = body_pt(W)
    sub = ("Overall and macro accuracy on the same 996-image validation split. "
           "The y-axis is truncated to 88–91.5% to make small gaps visible.")
    src = "evaluation/reports/stage4_checkpoint_comparison.json"
    extra = "Differences are 0.3 points or less: a tie-break, not a breakthrough."
    note = ""
    if len(pit) == 1:
        note = ("PithaNet accuracy is %.2f%% at all three, so the gain comes from existing foods "
                "(%.2f%% → %.2f%%)." % (list(pit)[0], comp[cks[0]]["old_food_accuracy"],
                                            comp[cks[2]]["old_food_accuracy"]))
    note_lines = wrap(note, fs, (W - 1.0) * 72)
    hh, fh = header_h(W, sub), footer_h(W, src, extra)
    plot_top = hh + 0.3
    plot_h = 3.9
    note_y = plot_top + plot_h + 1.15
    H = note_y + len(note_lines) * fs * LH / 72 + 0.4 + fh
    fig = new_fig(W, H)
    header(fig, "Choosing the final checkpoint", sub)
    ax = fig.add_axes([1.0 / W, 1 - (plot_top + plot_h) / H, 5.4 / W, plot_h / H])
    x = np.arange(3)
    ax.axvspan(1.55, 2.45, color=GREEN, alpha=0.22, lw=0)
    for vals, col, dy in ((ov, GREEN_HI, 0.2), (mc, TEXT, -0.2)):
        ax.plot(x, vals, color=col, lw=1.8, alpha=0.8, zorder=2)
        ax.scatter(x, vals, s=[70, 70, 170], color=col, zorder=3, edgecolor=BG, linewidth=1)
        for xi, v in zip(x, vals):
            ax.text(xi, v + dy, f"{v:.2f}%", ha="center", va="center", fontsize=fs + 1,
                    color=col, fontweight="bold" if xi == 2 else "normal")
    ax.text(2.0, 91.45, "selected", ha="center", va="top", fontsize=fs + 0.5, color=GREEN_HI,
            fontweight="bold")
    ax.text(-0.9, ov[0], "overall", ha="left", va="center", fontsize=fs + 0.5, color=GREEN_HI)
    ax.text(-0.9, mc[0], "macro", ha="left", va="center", fontsize=fs + 0.5, color=TEXT)
    ax.set_xlim(-0.95, 2.5); ax.set_ylim(88.0, 91.5)
    ax.set_xticks(x); ax.set_xticklabels(["500", "600", "645"], fontsize=fs + 1)
    ax.set_yticks([88, 89, 90, 91]); ax.set_yticklabels(["88%", "89%", "90%", "91%"], fontsize=fs)
    ax.set_xlabel("checkpoint (training step)", fontsize=fs, labelpad=8)
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, ln in enumerate(note_lines):
        fig.text(0.5 / W, 1 - (note_y + i * fs * LH / 72) / H, ln, fontsize=fs, color=TEXT,
                 ha="left", va="top")
    footer(fig, src, extra)
    save(fig, "03_checkpoint_selection")


# ---------- 05: training curve ----------
def chart_training(d):
    rows = d["loss"]
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    acc = np.array([r["token_acc"] for r in rows]) * 100
    W = 7.4
    fs = body_pt(W)
    sub = "Logged during training on the training set (not validation metrics). Final step 645 is the released checkpoint."
    src = "checkpoint-645 trainer_state.json (training log, viz/data/trainer_state_loss.csv)"
    extra = "Train-time metrics only; validation accuracy is on the other charts."
    hh, fh = header_h(W, sub), footer_h(W, src, extra)
    p1 = hh + 0.55
    ph = 2.2
    p2 = p1 + ph + 0.95
    H = p2 + ph + 0.95 + fh
    fig = new_fig(W, H)
    header(fig, "Training curve", sub)
    for k, (series, ttl, col, fmt) in enumerate((
            (loss, "training loss", GREEN_HI, "{:.3f}"),
            (acc, "training token accuracy (%)", TEXT, "{:.1f}%"))):
        top = p1 if k == 0 else p2
        ax = fig.add_axes([1.0 / W, 1 - (top + ph) / H, 5.5 / W, ph / H])
        ax.plot(steps, series, color=col, lw=1.6, alpha=0.95)
        ax.scatter([steps[-1]], [series[-1]], s=55, color=col, zorder=3, edgecolor=BG)
        ax.text(0.98, 0.78 if k == 0 else 0.14, "final (step 645): " + fmt.format(series[-1]),
                ha="right", va="center", fontsize=fs, color=col, transform=ax.transAxes)
        for ck in (500, 600):
            ax.axvline(ck, color=MUTED, lw=0.6, ls=":", alpha=0.7)
        ax.set_xlim(0, 660)
        if k == 1:
            ax.set_ylim(85, 101)
            ax.set_yticks([85, 90, 95, 100])
        ax.set_title(ttl, loc="left", fontsize=fs, color=MUTED, pad=8)
        ax.tick_params(length=0, pad=5, labelsize=fs)
        for s in ax.spines.values():
            s.set_visible(False)
        if k == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("training step (dotted: checkpoints 500 and 600)", fontsize=fs, labelpad=8)
    footer(fig, src, extra)
    save(fig, "05_training_curve")


# ---------- 06: class metrics table ----------
def chart_metrics_table(d, cm):
    classes = d["classes"]
    metrics = class_metrics(cm, classes)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "class_metrics_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "group", "precision_pct", "recall_pct", "f1_pct", "support", "correct"])
        for m in metrics:
            w.writerow([m["cls"], "pithanet" if m["cls"] in d["pitha"] else "existing",
                        f"{m['precision'] * 100:.2f}", f"{m['recall'] * 100:.2f}",
                        f"{m['f1'] * 100:.2f}", m["support"], m["tp"]])
    print("  wrote viz/out/class_metrics_full.csv")

    weakest = sorted(metrics, key=lambda m: (m["f1"], m["cls"]))[:10]
    macro_p = float(np.mean([m["precision"] for m in metrics]))
    macro_r = float(np.mean([m["recall"] for m in metrics]))
    macro_f = float(np.mean([m["f1"] for m in metrics]))
    total = int(cm.sum()); correct = int(np.trace(cm))

    W = 7.8
    fs = body_pt(W) + 0.8
    sub = "The 10 classes with the lowest F1 on the validation split, with macro average and overall accuracy."
    src = "viz/data/stage4_predictions_pairs.csv; recall and support checked against evaluation/reports/stage4_checkpoint645_per_class.csv"
    extra = ("Precision = correct / times predicted. Recall = per-class accuracy. Macro = unweighted mean "
             "over all 27 classes. Full 27-row table: viz/out/class_metrics_full.csv.")
    hh, fh = header_h(W, sub), footer_h(W, src, extra)
    rh = 0.50
    top = hh + 0.25
    H = top + rh * (1 + 10 + 1 + 2) + 0.3 + fh
    fig = new_fig(W, H)
    header(fig, "Weakest classes: precision, recall, F1", sub)
    cv = canvas(fig)
    xs = {"cls": 0.5, "p": 4.55, "r": 5.65, "f": 6.6, "n": 7.3}
    y = top
    cv.text(xs["cls"], y + rh / 2, "Class", fontsize=fs, color=MUTED, ha="left", va="center")
    for key, lab in (("p", "Precision"), ("r", "Recall"), ("f", "F1"), ("n", "n")):
        cv.text(xs[key], y + rh / 2, lab, fontsize=fs, color=MUTED, ha="right", va="center")
    y += rh
    cv.plot([0.5, W - 0.5], [y, y], color=PANEL_EDGE, lw=1)

    def put(y0, name, p, r, f, n, col=TEXT, bold=False):
        wt = "bold" if bold else "normal"
        cv.text(xs["cls"], y0 + rh / 2, name, fontsize=fs, color=col, ha="left", va="center", fontweight=wt)
        for key, val in (("p", p), ("r", r), ("f", f), ("n", n)):
            cv.text(xs[key], y0 + rh / 2, val, fontsize=fs, ha="right", va="center", fontweight=wt,
                    color=TEXT if key != "n" else MUTED)

    for k, m in enumerate(weakest):
        if k % 2 == 0:
            cv.add_patch(Rectangle((0.4, y), W - 0.8, rh, fc=PANEL, ec="none"))
        col = GREEN_HI if m["cls"] in d["pitha"] else TEXT
        put(y, pretty(m["cls"]), f"{m['precision'] * 100:.1f}%", f"{m['recall'] * 100:.1f}%",
            f"{m['f1'] * 100:.1f}%", str(m["support"]), col)
        y += rh
    cv.plot([0.5, W - 0.5], [y + 0.04, y + 0.04], color=MUTED, lw=1)
    y += 0.1
    put(y, "Macro avg (27 classes)", f"{macro_p * 100:.1f}%", f"{macro_r * 100:.1f}%",
        f"{macro_f * 100:.1f}%", str(total), TEXT, True)
    y += rh
    put(y, "Overall accuracy", "", "", f"{correct / total * 100:.2f}%", str(total), TEXT, True)
    footer(fig, src, extra)
    save(fig, "06_class_metrics_table")


# ---------- 07: pitha 8x8 confusion ----------
def chart_pitha_matrix(d, cm):
    pith = d["pitha"]
    m = pitha_matrix(cm)
    W = 8.0
    fs = body_pt(W) + 0.6
    sub = ("Counts of validation images. Rows are the true pitha; columns are what the model predicted. "
           "“other food class” means it answered with one of the 19 non-pitha foods.")
    src = "viz/data/stage4_predictions_pairs.csv; row sums and diagonal (329) checked against evaluation/reports/"
    extra = "Each row sums to that class's number of validation images, shown in brackets."
    hh, fh = header_h(W, sub), footer_h(W, src, extra)
    left, cell = 2.35, 0.6
    top = hh + 0.55
    x_lab_h = 1.45
    H = top + 8 * cell + x_lab_h + 0.4 + fh
    fig = new_fig(W, H)
    header(fig, "Pitha confusion matrix", sub)
    cv = canvas(fig)
    maxoff = max(int(m[i, j]) for i in range(8) for j in range(9) if i != j) or 1
    for i in range(8):
        n = int(m[i].sum())
        cv.text(left - 0.15, top + i * cell + cell / 2, f"{pretty(pith[i])} ({n})", fontsize=fs,
                color=GREEN_HI, ha="right", va="center")
        for j in range(9):
            v = int(m[i, j])
            x0, y0 = left + j * cell, top + i * cell
            if j == i:
                fc, tc = mix(PANEL, GREEN_HI, 0.35 + 0.65 * v / n), BG
            elif v > 0:
                fc, tc = mix(PANEL, RED, 0.28 + 0.72 * v / maxoff), TEXT
            else:
                fc, tc = PANEL, FAINT
            cv.add_patch(Rectangle((x0 + 0.02, y0 + 0.02), cell - 0.04, cell - 0.04, fc=fc, ec="none"))
            cv.text(x0 + cell / 2, y0 + cell / 2, str(v), fontsize=fs + 1, color=tc, ha="center",
                    va="center", fontweight="bold" if v else "normal")
    cv.plot([left + 8 * cell, left + 8 * cell], [top - 0.05, top + 8 * cell + 0.05],
            color=MUTED, lw=1)
    labels = [pretty(c) for c in pith] + ["other food class"]
    for j, lab in enumerate(labels):
        cv.text(left + j * cell + cell / 2 + 0.08, top + 8 * cell + 0.12, lab, fontsize=fs,
                color=GREEN_HI if j < 8 else MUTED, ha="right", va="top", rotation=40,
                rotation_mode="anchor")
    cv.text(0.5, top - 0.2, "True class", fontsize=fs, color=MUTED, ha="left", va="bottom")
    cv.text(left + 4.5 * cell, top + 8 * cell + x_lab_h + 0.05, "Predicted class", fontsize=fs,
            color=MUTED, ha="center", va="top")
    footer(fig, src, extra)
    save(fig, "07_pitha_confusion_8x8")


# ---------- 08: top confusions ----------
def chart_top_confusions(d, cm, totals):
    classes = d["classes"]
    ranked = top_confusions(cm, classes, 10)
    in_summary = set(d["conf_summary"])
    W = 8.4
    fs = body_pt(W) + 0.8
    tied_extra = [r for r in ranked if f"{classes[r[2]]} -> {classes[r[3]]}" not in in_summary]
    sub = ("The most frequent wrong answers, as true class → predicted class, "
           "with how many of that class's validation images it affected.")
    src = "evaluation/reports/stage4_checkpoint645_confusion_summary.json (counts re-derived from predictions)"
    extra = ("All pairs tied with 10th place are shown"
             + ("; † = tied but not listed in the repo summary." if tied_extra else "."))
    hh, fh = header_h(W, sub), footer_h(W, src, extra)
    labels = [f"{classes[i]} → {classes[j]}"
              + (" †" if f"{classes[i]} -> {classes[j]}" not in in_summary else "")
              for _, _, i, j in ranked]
    lab_w = max(text_w(s, fs) for s in labels) / 72
    bar0 = 0.5 + lab_w + 0.25
    count_w = max(text_w(f"{cnt} of {int(totals[i])}", fs, "bold") for _, cnt, i, _ in ranked) / 72
    maxcnt = max(cnt for _, cnt, _, _ in ranked)
    scale = (W - 0.5 - bar0 - count_w - 0.2) / maxcnt
    rh = 0.62
    top = hh + 0.2
    H = top + rh * len(ranked) + 0.35 + fh
    fig = new_fig(W, H)
    header(fig, "Top confusions", sub)
    cv = canvas(fig)
    for k, ((rank, cnt, i, j), lab) in enumerate(zip(ranked, labels)):
        y = top + k * rh + rh / 2
        cv.text(bar0 - 0.2, y, lab, fontsize=fs, color=TEXT, ha="right", va="center")
        cv.add_patch(Rectangle((bar0, y - 0.17), cnt * scale, 0.34, fc=RED, ec="none", alpha=0.9))
        cv.text(bar0 + cnt * scale + 0.12, y, f"{cnt} of {int(totals[i])}", fontsize=fs, color=TEXT,
                ha="left", va="center", fontweight="bold")
    footer(fig, src, extra)
    save(fig, "08_top_confusions")


# ---------- 09: how it was built ----------
def chart_how_built(d):
    s, rep, cfg, comp = d["summary"], d["report"], d["train_cfg"], d["comparison"]["checkpoint-645"]
    rows = [
        ("Data",
         "Combined several Bangladeshi food datasets with PithaNet, used with permission.",
         f"{s['pithanet_unique_training_samples']:,} clean unique PithaNet training images"),
        ("Rehearsal",
         "Mixed in samples of the 19 existing food classes alongside the new pithas.",
         f"{s['old_food_rehearsal_samples']:,} rehearsal samples; {s['training_samples']:,} training samples in total"),
        ("Labels",
         "Expanded the closed label set from 19 to 27 classes by adding 8 PithaNet pithas.",
         f"{s['num_classes']} labels"),
        ("Fine-tuning",
         f"Qwen3-VL-2B-Instruct with LoRA (rank {cfg['lora_rank']}, alpha {cfg['lora_alpha']}, "
         f"dropout {cfg['lora_dropout']}) on 4-bit NF4 quantization, {cfg['training']['epochs']} epochs, "
         f"learning rate {format(cfg['training']['learning_rate'], '.0e').replace('e-0', 'e-')}, 2 × Tesla T4.",
         "A small adapter; the base model is not redistributed"),
        ("Selection",
         "Compared checkpoints 500, 600 and 645 on the same validation split.",
         f"checkpoint-645 best: {comp['overall_accuracy']:.2f}% overall, {comp['macro_accuracy']:.2f}% macro"),
        ("Evaluation",
         f"Scored the held-out validation split of {rep['validation_total']} images.",
         f"{rep['overall']['correct']}/{rep['overall']['total']} correct; foods "
         f"{rep['old_food']['correct']}/{rep['old_food']['total']}, pithas "
         f"{rep['pithanet']['correct']}/{rep['pithanet']['total']}"),
        ("Split check",
         "Hashed every PithaNet train and validation image (SHA-256, dHash, pHash) and compared the splits.",
         f"{d['content']['sha256']['validation_images_with_byte_identical_train_image']} byte-identical; "
         f"{d['content']['near_duplicates']['dhash']['validation_images_with_a_match']} validation images "
         f"have a near-duplicate in train"),
        ("Known limits",
         "The 19 existing-food classes were not content-audited, and label noise was not checked.",
         "Not evidence of performance on new cameras, regions or unseen foods"),
    ]
    W = 8.2
    fs = body_pt(W) + 0.4
    sub = "What was done and what it produced, using only facts recorded in the repository's docs and reports."
    src = ("README.md, docs/TRAINING_HISTORY.md, docs/DATASETS.md, docs/LIMITATIONS.md, "
           "metadata/training_config.json, evaluation/reports/stage4_* (incl. the two audit reports)")
    hh, fh = header_h(W, sub), footer_h(W, src)
    cx = [0.5, 2.05, 5.55]
    cw = [1.4, 3.35, W - 0.5 - 5.55]
    pad = 0.18
    lh = fs * 1.35 / 72
    wrapped = []
    for step, what, res in rows:
        a = wrap(what, fs, (cw[1] - 0.1) * 72)
        b = wrap(res, fs, (cw[2] - 0.05) * 72)
        wrapped.append((step, a, b))
    heights = [max(len(a), len(b), 1) * lh + 2 * pad for _, a, b in wrapped]
    top = hh + 0.2
    head_h = 0.55
    H = top + head_h + sum(heights) + 0.3 + fh
    fig = new_fig(W, H)
    header(fig, "How it was built", sub)
    cv = canvas(fig)
    for x, lab in zip(cx, ("Step", "What happened", "Result")):
        cv.text(x, top + head_h / 2, lab, fontsize=fs, color=MUTED, ha="left", va="center")
    y = top + head_h
    cv.plot([0.5, W - 0.5], [y, y], color=MUTED, lw=1)
    for k, ((step, a, b), h) in enumerate(zip(wrapped, heights)):
        if k % 2 == 0:
            cv.add_patch(Rectangle((0.4, y), W - 0.8, h, fc=PANEL, ec="none"))
        cv.text(cx[0], y + pad, step, fontsize=fs, color=GREEN_HI, fontweight="bold", ha="left", va="top")
        for n, ln in enumerate(a):
            cv.text(cx[1], y + pad + n * lh, ln, fontsize=fs, color=TEXT, ha="left", va="top")
        for n, ln in enumerate(b):
            cv.text(cx[2], y + pad + n * lh, ln, fontsize=fs, color=GREEN_HI if k in (4, 5) else TEXT,
                    ha="left", va="top", fontweight="bold" if k in (4, 5) else "normal")
        y += h
    footer(fig, src)
    save(fig, "09_how_it_was_built")


# ---------- hero cards ----------
def tile(ax, x, y, w, h, big, label, sub, big_px, label_px, sub_px):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=16",
                                fc=PANEL, ec=PANEL_EDGE, lw=1))
    ax.text(x + 26, y + 18, big, fontsize=px(big_px), fontweight="bold", color=GREEN_HI,
            ha="left", va="top")
    ax.text(x + 26, y + 18 + big_px * 1.22, label, fontsize=px(label_px), fontweight="bold",
            color=TEXT, ha="left", va="top")
    ax.text(x + 26, y + 18 + big_px * 1.22 + label_px * 1.4, sub, fontsize=px(sub_px),
            color=MUTED, ha="left", va="top")


def fails_panel(ax, x, y, w, ranked, classes, totals, title_px, row_px, count_px, row_h,
                two_line=False, pad=28, rank_off=70, gap=26):
    inner_top = y + pad
    ax.text(x + pad, inner_top, "Where it still fails", fontsize=px(title_px), fontweight="bold",
            color=TEXT, ha="left", va="top")
    ry = inner_top + title_px * 1.25 + gap
    for rank, cnt, i, j in ranked:
        lab = rank_label(rank, ranked)
        pair = f"{pretty(classes[i])} → {pretty(classes[j])}"
        count = f"{cnt} of {int(totals[i])} images"
        ax.text(x + pad, ry, lab, fontsize=px(row_px), fontweight="bold", color=RED, ha="left", va="top")
        if two_line:
            ax.text(x + pad + 54, ry, pair, fontsize=px(row_px), color=TEXT, ha="left", va="top")
            ax.text(x + pad + 54, ry + row_px * 1.3, count, fontsize=px(count_px), color=MUTED,
                    ha="left", va="top")
        else:
            ax.text(x + pad + rank_off, ry, pair, fontsize=px(row_px), color=TEXT, ha="left", va="top")
            ax.text(x + w - pad, ry + (row_px - count_px) * 0.4, count, fontsize=px(count_px),
                    color=MUTED, ha="right", va="top")
        ry += row_h


def hero(d, cm, totals, W, Hh, name):
    fig = plt.figure(figsize=(W / DPI, Hh / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(Hh, 0); ax.axis("off")
    classes = d["classes"]
    s = d["summary"]; rep = d["report"]
    ov, mc = s["overall_accuracy"], s["macro_accuracy"]
    ranked = top_confusions(cm, classes, 3)
    src = "Source: evaluation/reports/stage4_* (checkpoint-645)"

    if W == 1080:
        m = 64
        ax.text(m, 62, "QWEN3-VL-2B  ·  LoRA / QLoRA  ·  STAGE-4", fontsize=px(24),
                color=MUTED, ha="left", va="top")
        ax.text(m, 106, "Deshi Digest", fontsize=px(92), fontweight="bold", color=TEXT, ha="left", va="top")
        ax.text(m, 204, "Vision Model", fontsize=px(92), fontweight="bold", color=TEXT, ha="left", va="top")
        ax.text(m, 332, "27 Bangladeshi foods incl. 8 traditional pithas", fontsize=px(36),
                color=GREEN_HI, ha="left", va="top")
        tw, gap, ty, th = 296, 32, 414, 200
        tile(ax, m, ty, tw, th, f"{ov:.2f}%", "overall accuracy",
             f"{rep['overall']['correct']} of {rep['overall']['total']} correct", 70, 28, 22)
        tile(ax, m + tw + gap, ty, tw, th, f"{mc:.2f}%", "macro accuracy", "mean over 27 classes", 70, 28, 22)
        tile(ax, m + 2 * (tw + gap), ty, tw, th, "27", "classes", "19 foods + 8 pithas", 70, 28, 22)
        gy = ty + th + 34
        ax.text(m, gy, f"{rep['old_food']['accuracy']:.2f}%", fontsize=px(50), fontweight="bold",
                color=TEXT, ha="left", va="top")
        ax.text(m, gy + 62, f"existing foods, {rep['old_food']['correct']}/{rep['old_food']['total']}",
                fontsize=px(24), color=MUTED, ha="left", va="top")
        ax.text(m + 484, gy, f"{rep['pithanet']['accuracy']:.2f}%", fontsize=px(50), fontweight="bold",
                color=GREEN_HI, ha="left", va="top")
        ax.text(m + 484, gy + 62, f"pithas, {rep['pithanet']['correct']}/{rep['pithanet']['total']}",
                fontsize=px(24), color=MUTED, ha="left", va="top")
        py = gy + 124
        ph = 352
        ax.add_patch(FancyBboxPatch((m, py), W - 2 * m, ph, boxstyle="round,pad=0,rounding_size=16",
                                    fc=PANEL, ec=PANEL_EDGE, lw=1))
        ax.add_patch(Rectangle((m, py + 16), 6, ph - 32, fc=RED, ec="none"))
        fails_panel(ax, m, py, W - 2 * m, ranked, classes, totals, 34, 30, 26, 60)
        fy = py + ph + 34
        ax.text(m, fy, "\n".join(wrap(CREDIT, 22, 700)), fontsize=px(22), color=TEXT,
                ha="left", va="top", linespacing=1.35)
        ax.text(m, fy + 82, FOOTER, fontsize=px(22), color=MUTED, ha="left", va="top")
        ax.text(m, fy + 118, src, fontsize=px(22), color=MUTED, ha="left", va="top")
        print("    1080 hero: content ends near y =", fy + 140, "of", Hh)
    else:  # 1200 x 627 link preview
        m = 44
        ax.text(m, 20, "Deshi Digest Vision Model", fontsize=px(48), fontweight="bold",
                color=TEXT, ha="left", va="top")
        ax.text(m, 80, "27 Bangladeshi foods incl. 8 traditional pithas", fontsize=px(27),
                color=GREEN_HI, ha="left", va="top")
        tw, gap, ty, th = 360, 16, 118, 140
        tile(ax, m, ty, tw, th, f"{ov:.2f}%", "overall accuracy",
             f"{rep['overall']['correct']} of {rep['overall']['total']} correct", 44, 24, 22)
        tile(ax, m + tw + gap, ty, tw, th, f"{mc:.2f}%", "macro accuracy", "mean over 27 classes", 44, 24, 22)
        tile(ax, m + 2 * (tw + gap), ty, tw, th, "27", "classes", "19 foods + 8 pithas", 44, 24, 22)
        ly = ty + th + 18
        ax.text(m, ly, f"{rep['old_food']['accuracy']:.2f}%", fontsize=px(40), fontweight="bold",
                color=TEXT, ha="left", va="top")
        ax.text(m, ly + 46, f"existing foods, {rep['old_food']['correct']}/{rep['old_food']['total']}",
                fontsize=px(22), color=MUTED, ha="left", va="top")
        ax.text(m, ly + 92, f"{rep['pithanet']['accuracy']:.2f}%", fontsize=px(40), fontweight="bold",
                color=GREEN_HI, ha="left", va="top")
        ax.text(m, ly + 138, f"pithas, {rep['pithanet']['correct']}/{rep['pithanet']['total']}",
                fontsize=px(22), color=MUTED, ha="left", va="top")
        px0 = 470
        pw = W - m - px0
        ph = 200
        ax.add_patch(FancyBboxPatch((px0, ly - 4), pw, ph, boxstyle="round,pad=0,rounding_size=14",
                                    fc=PANEL, ec=PANEL_EDGE, lw=1))
        ax.add_patch(Rectangle((px0, ly + 10), 5, ph - 28, fc=RED, ec="none"))
        fails_panel(ax, px0, ly - 4, pw, ranked, classes, totals, 24, 22, 22, 34,
                    pad=18, rank_off=56, gap=14)
        fy = ly - 4 + ph + 14
        ax.plot([m, W - m], [fy, fy], color=PANEL_EDGE, lw=1)
        ax.text(m, fy + 14, CREDIT, fontsize=px(22), color=TEXT, ha="left", va="top")
        ax.text(m, fy + 50, FOOTER, fontsize=px(22), color=MUTED, ha="left", va="top")
        ax.text(m, fy + 86, src, fontsize=px(22), color=MUTED, ha="left", va="top")
        print("    1200 hero: content ends near y =", fy + 110, "of", Hh)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print("  wrote", f"viz/out/{name}.png")


def main():
    d = load_data()
    cm, diag, totals, macro = verify(d)
    print(f"Font: {FONT}")
    print("Drawing")
    chart_confusion(d, cm, totals)
    chart_per_class(d, diag, totals)
    chart_checkpoints(d)
    chart_training(d)
    chart_metrics_table(d, cm)
    chart_pitha_matrix(d, cm)
    chart_top_confusions(d, cm, totals)
    chart_how_built(d)
    hero(d, cm, totals, 1080, 1350, "00_hero_card_1080x1350")
    hero(d, cm, totals, 1200, 627, "00_hero_card_1200x627")
    print("Skipped: 04_pitha_gallery_grid (no permission-safe example images in examples/)")
    if not (ROOT / "demo_photos").is_dir():
        print("Skipped: 10_live_predictions (needs a demo_photos/ folder with your own photos)")


if __name__ == "__main__":
    main()
