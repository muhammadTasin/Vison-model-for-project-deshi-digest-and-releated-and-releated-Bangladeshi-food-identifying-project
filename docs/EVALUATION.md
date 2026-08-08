# Evaluation

## Current Stage‑4 validation

Stage‑4 checkpoint‑645 was selected from checkpoints 500, 600, and 645 on a 996-image held-out validation split. It achieved 576/637 (90.42%) on existing foods, 329/359 (91.64%) on PithaNet foods, 905/996 (90.86%) overall, and 89.09% macro accuracy.

The aggregate reports in `evaluation/reports/stage4_*` are the source of truth. They omit raw predictions and image paths. Use `evaluation/evaluate_predictions.py` only with authorized local prediction data.

## Method and caveats

Predictions should be normalized by exact alias mapping and compared against versioned ground truth. Report overall and macro accuracy, per-class accuracy, and confusion summaries; inspect errors for label issues, duplicates, ambiguous images, and split leakage. The published result is specific to its held-out split and does not establish unknown/non-food performance, confidence calibration, or external generalization.

## Historical Stage‑3 benchmark

Stage‑3 `checkpoint-348` reached 83/95 (87.37%) on a separate fixed 19-class internal benchmark. Its reports remain under `evaluation/reports/stage3_*`; do not directly compare it to the Stage‑4 996-image validation score.
