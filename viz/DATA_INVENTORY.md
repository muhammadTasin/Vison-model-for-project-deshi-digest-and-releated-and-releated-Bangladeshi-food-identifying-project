# Data inventory for result visuals

Stage-4 `checkpoint-645`, Qwen3-VL-2B-Instruct + LoRA/QLoRA, 27 classes.
Held-out validation split, 996 images (637 existing foods, 359 PithaNet).

## In the repo (read directly by `make_visuals.py`)

| File | Contents |
|---|---|
| `evaluation/reports/stage4_checkpoint645_report.json` | overall 905/996 (90.86%), existing 576/637 (90.42%), PithaNet 329/359 (91.64%), per-class correct/total |
| `evaluation/reports/stage4_checkpoint645_per_class.csv` | same per-class numbers as CSV |
| `evaluation/reports/stage4_summary.json` | overall 90.86, macro 89.09, 6,872 train / 996 val, LoRA r=4, 2x T4 |
| `evaluation/reports/stage4_checkpoint_comparison.json` | checkpoints 500 / 600 / 645 (same split): overall 90.56 / 90.66 / 90.86, macro 88.89 / 88.89 / 89.09 |
| `evaluation/reports/stage4_checkpoint645_confusion_summary.json` | top-10 confusion pairs only (not a full matrix) |
| `metadata/labels.json` | 19 existing + 8 PithaNet class ids |
| `examples/` | three JSON API outputs only, no images, so no gallery |

Stage-3 reports (`stage3_checkpoint348_*`, 87.37%, 83/95) are on a different 95-image
internal benchmark and are intentionally NOT used in any visual.

## Not in the repo, found on this laptop (read-only sources)

Both live in local backup archives (zip files) of the Stage-4 training run:

| Source (file inside the archive) | Used for | Copied to |
|---|---|---|
| Final-model archive (`DESHI_DIGEST_STAGE4_BEST_90_86.zip`): `benchmark/gpu0_results.jsonl`, `benchmark/gpu1_results.jsonl` (498 + 498 rows) | per-sample true label + prediction for all 996 validation images | `viz/data/stage4_predictions_pairs.csv` (only `true,pred`; image references and prompts dropped) |
| Final-model archive: `benchmark/FINAL_ACCURACY_REPORT.json` | cross-check only | not copied |
| Full training backup archive (`DESHI_DIGEST_STAGE4_27CLASS_BACKUP.zip`): `manifests/stage4_val_27class.jsonl` | cross-check only (996 rows, 27 classes) | not copied |
| Full training backup archive: `training_run/.../checkpoint-645/trainer_state.json` | training-loss curve (130 logged steps, 1 to 645) | `viz/data/trainer_state_loss.csv` |

`trainer_state.json` contains training loss and token accuracy only. There is no
eval loss and no validation accuracy per step, so the curve is labelled "training loss".

## Round 2 additions (all computed from the files above)

- `06_class_metrics_table` and `class_metrics_full.csv`: precision / recall / F1 / support from the predictions.
- `07_pitha_confusion_8x8`: 8 pitha rows x (8 pitha + "other food class") columns.
- `08_top_confusions`: top 10 pairs plus any ties; counts equal `stage4_checkpoint645_confusion_summary.json`.
- `09_how_it_was_built`: every number is read from `stage4_summary.json`, `metadata/training_config.json`
  or the reports and is checked to appear in `README.md`, `docs/DATASETS.md`,
  `docs/TRAINING_HISTORY.md` or `docs/LIMITATIONS.md`.

## Deliberately omitted, not documented in the repo

Duplicate removal, label fixes, leakage removal and "frozen vision encoder" are not stated in the
README, MODEL_CARD or docs (the docs list leakage and label noise as remaining risks), so they
are not claimed in any visual.

## Cross-checks (recomputed, not trusted)

| Quantity | README / report | Recomputed from predictions | Match |
|---|---|---|---|
| Overall | 905/996, 90.86% | 905/996 | yes |
| Existing foods | 576/637 | 576/637 | yes |
| PithaNet | 329/359 | 329/359 | yes |
| Macro | 89.09% | 89.09% (89.0917) | yes |
| Top confusions | nakshi->jamai 8, morog_polao->biryani 6, jamai->patishapta 4, khichuri->biryani 4 | identical | yes |
| `FINAL_ACCURACY_REPORT.json` (zip) vs repo report | | identical except sanitized `checkpoint` path | yes |
| Validation manifest | 996 rows | 996 rows, 27 classes | yes |

`make_visuals.py` re-runs these checks at start-up and aborts if anything disagrees.

## Skipped

- `04_pitha_gallery_grid`: no permission-safe example images in `examples/`.
- `10_live_predictions`: needs a `demo_photos/` folder with your own photos (none exists).
