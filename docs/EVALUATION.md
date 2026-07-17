# Evaluation

## Current protocol

The controlled internal benchmark has 19 classes, five images per class, and 95
images total. Each model response is normalized through an exact alias map and
then compared with the normalized ground-truth identifier. There is no
substring or semantic matching.

The Stage-3 result is:

- Correct: 83
- Incorrect: 12
- Accuracy: 87.37%
- Observed runtime: approximately 45.65 seconds for 95 samples
- Observed throughput: approximately 2.08 samples/second
- Original prediction path:
  `/kaggle/working/stage3_checkpoint348_predictions_resolved.jsonl`

Runtime and throughput are environment-specific observations, not guaranteed
deployment speed.

## Label normalization

The canonical public identifiers include `bangladeshi_biriyani` and
`roshogolla`. Exported benchmark files instead contain
`bangladeshi_biryani` and `roshgolla`. The scorer maps these exact aliases to
the public identifiers before comparison. It never searches for a known class
as a substring of arbitrary text.

## Results

The per-class and confusion reports are checked into
[`evaluation/reports`](../evaluation/reports/README.md). The lowest Stage-3
per-class results are `morog_polao` (2/5), `roshmalai` (2/5), and
`kacha_golla` (3/5). The most frequent confusion is `roshmalai` →
`roshogolla` (2); each other observed pair occurred once.

## Reproduce scoring

The standard-library scorer accepts the exported MS-SWIFT fields `labels` and
`response` as well as common alternatives such as `ground_truth` and
`prediction`:

```bash
python evaluation/evaluate_predictions.py \
  path/to/predictions.jsonl \
  --aliases inference/label_aliases.json \
  --output-dir evaluation/reports/local-run
```

It writes:

- `report.json`
- `per_class.csv`
- `wrong_predictions.csv`
- `confusion_pairs.csv`

Missing or null ground truth is a fatal input error. It is never silently
counted as an incorrect prediction. Invalid JSON, duplicate alias conflicts,
non-string labels, and missing predictions also fail with a line-specific
message.

## Wrong-prediction inspection

Review the source image, original label, normalized label, raw response,
normalized prediction, and split membership for every error. Check whether the
image is ambiguous, mislabeled, duplicated, mixed-plate, outside the class
definition, or contains textual leakage. Corrections to benchmark labels must
be versioned and must trigger rescoring of every stage.

## Limitations

- Five examples per class produce high uncertainty in per-class rates: one
  image changes a class result by 20 percentage points.
- The benchmark's source distribution and independence from training have not
  been fully audited.
- Balanced class counts do not reflect real application prevalence.
- Closed-set exact-label accuracy does not measure unknown or non-food rejection.
- Repeated checkpoint selection on one small benchmark can overfit model
  development decisions to the benchmark.
- No external real-world accuracy is currently verified.

This benchmark is too small to establish production-level generalization.

## Recommended future metrics

- Macro F1 and balanced accuracy
- Full confusion matrix with confidence intervals
- JSON/schema validity rate
- Unknown-detection precision and recall
- Non-food rejection accuracy
- Hallucination rate
- Mixed-plate component recall
- External real-world accuracy by capture condition
- Base-model versus adapter comparison
- Latency distributions and resource use by deployment hardware

Maintain the original 95-image set for historical comparison, but select final
deployment thresholds and model versions using a separately governed validation
set and confirm once on a held-out external test set.
