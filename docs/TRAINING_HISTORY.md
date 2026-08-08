# Training history

## Current Stage‑4 release

| Stage | Evaluation | Best checkpoint | Result | Status |
| --- | --- | --- | --- | --- |
| **Stage 4** | 996-image held-out 27-class validation | **checkpoint‑645** | **90.86% overall; 89.09% macro** | Current release |
| Stage 3 | Fixed internal 95-image 19-class benchmark | checkpoint‑348 | 87.37% | Historical baseline |
| Stage 2 | Fixed internal 95-image benchmark | Not verified | 82.11% | Historical |
| Stage 1 | Fixed internal 95-image benchmark | checkpoint‑901 | 83.16% | Historical |

Stage‑4 uses a different and larger validation split, so its result must not be presented as a direct numerical continuation of the Stage‑1–3 95-image benchmark.

## Stage‑4 recorded configuration

- Base: `Qwen/Qwen3-VL-2B-Instruct`
- LoRA / QLoRA: rank 4, alpha 16, dropout 0.05; bitsandbytes 4-bit NF4
- Hardware: 2 × NVIDIA Tesla T4
- Population: 2,720 existing-food rehearsal samples + 4,152 unique PithaNet images
- Training: 1.5 epochs, `2e-5` learning rate, cosine schedule, 5% warmup, batch 1, accumulation 8
- Selection: checkpoint‑645 outperformed checkpoints 500 and 600 on reported overall and macro accuracy

See `metadata/training_config.json` and `evaluation/reports/stage4_checkpoint_comparison.json`.

## Historical Stage‑3 note

Stage‑3 remains documented as a rollback/reference baseline. Its observed configuration was LoRA rank 8, alpha 32, dropout 0.05, one epoch, `3e-5` learning rate, batch 1 and accumulation 4 on two T4 GPUs. It achieved 83/95 (87.37%) on its fixed internal benchmark. Preserve its artifacts and use a separate governed evaluation if comparing future releases.
