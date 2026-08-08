# Stage‑4 release

Stage‑4 expands Deshi Digest from 19 to 27 controlled food labels using Qwen3‑VL‑2B‑Instruct with 4-bit NF4 QLoRA. The final model is checkpoint‑645.

## Selection result

| Checkpoint | Existing foods | PithaNet | Overall | Macro |
| --- | ---: | ---: | ---: | ---: |
| 500 | 89.95% | 91.64% | 90.56% | 88.89% |
| 600 | 90.11% | 91.64% | 90.66% | 88.89% |
| **645** | **90.42%** | **91.64%** | **90.86%** | **89.09%** |

Reports in `evaluation/reports/stage4_*` are the source of truth. No raw predictions, images, or absolute-path manifests are published.

## Recorded run configuration

- 6,872 training samples: 2,720 existing-food rehearsal samples plus 4,152 unique PithaNet images
- 996 held-out validation samples
- 1.5 epochs; learning rate `2e-5`; cosine schedule; 5% warmup
- Per-device batch size 1; gradient accumulation 8; max pixels 262,144
- Two NVIDIA Tesla T4 GPUs

See `metadata/training_config.json` and `metadata/environment.json` for exact recorded values.

## Historical documentation

The prior Stage‑3 release documents are retained under [`docs/historical`](historical/) for provenance and rollback context. They are explicitly historical and their 95-image benchmark is not directly comparable to the Stage‑4 validation split.
