---
base_model: Qwen/Qwen3-VL-2B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
tags: [lora, qlora, vision-language, bangladeshi-food, qwen3-vl, pithanet]
---

# Model Card: Deshi Digest Vision Model — Stage‑4

## Summary

This release is the Stage‑4 PEFT LoRA adapter for `Qwen/Qwen3-VL-2B-Instruct`. It supports 27 controlled Bangladeshi food labels: 19 existing labels and 8 PithaNet labels. The selected adapter is exported from **checkpoint‑645**.

| Field | Value |
| --- | --- |
| Fine-tuning | LoRA / 4-bit NF4 QLoRA |
| LoRA rank / alpha / dropout | 4 / 16 / 0.05 |
| Training hardware | 2 × NVIDIA Tesla T4 |
| Training samples | 6,872 |
| Held-out validation | 996 |
| Overall / macro accuracy | 90.86% / 89.09% |
| Existing-food / PithaNet accuracy | 90.42% / 91.64% |

The adapter requires the separately obtained compatible Qwen3-VL base model. It is not a standalone model.

## Intended and out-of-scope use

Appropriate use is research, education, and prototype closed-set food identification with user confirmation. It is not suitable for medical, allergy, ingredient, calorie, nutrition, portion-size, safety-critical, or autonomous decisions. The 27-label closed set does not establish unknown/non-food rejection or confidence calibration.

## Training and data

Stage‑4 used 2,720 rehearsal samples and 4,152 unique PithaNet training images. PithaNet was used with direct permission from its author(s). No raw images, datasets, manifests, private correspondence, base-model weights, or checkpoint state are included. See [`docs/DATASETS.md`](docs/DATASETS.md) and [`metadata/training_config.json`](metadata/training_config.json).

## Evaluation

Checkpoint‑645 achieved 905/996 (90.86%) on the Stage‑4 held-out validation split, with 89.09% macro accuracy. See [`evaluation/reports/stage4_checkpoint645_report.json`](evaluation/reports/stage4_checkpoint645_report.json) and the checkpoint comparison report. These results are split- and data-distribution-specific and are not a production-performance guarantee.

## Historical note

Stage‑3 `checkpoint-348` was a 19-class historical baseline with 83/95 (87.37%) on a different fixed internal benchmark. Stage‑3 reports and visual assets are retained for history; their score is not directly comparable with Stage‑4.

## License and attribution

Review the Qwen base-model terms, PEFT/MS-Swift dependencies, and every dataset’s terms before use. This repository does not redistribute raw datasets.
