# Evaluation reports

## Current Stage‑4 reports

- `stage4_checkpoint645_report.json`: final held-out validation report (905/996; 90.86%)
- `stage4_checkpoint_comparison.json`: checkpoint 500, 600, and 645 comparison
- `stage4_checkpoint645_per_class.csv`: 27-class per-class accuracy
- `stage4_checkpoint645_confusion_summary.json`: sanitized aggregate confusions
- `stage4_summary.json`: compact Stage‑4 run summary
- `stage4_manifest_audit.json`: aggregate train/validation manifest audit (produced by `evaluation/audit_manifests.py`; counts only)
- `stage4_content_audit.json`: PithaNet content-level train/validation audit, SHA-256 and perceptual hashes (produced by `evaluation/audit_content.py`; counts only)

No raw predictions, images, or absolute-path manifests are published.

## Historical Stage‑3 reports

`stage3_checkpoint348_*` records remain for historical reference. The Stage‑3 95-image benchmark and Stage‑4 996-image held-out validation split are different evaluations and should not be compared as like-for-like headline metrics.
