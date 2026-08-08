# Deshi Digest Vision Model

The current Deshi Digest vision model is a **Stage‑4** LoRA/QLoRA adapter for [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). It identifies 27 controlled Bangladeshi food labels: 19 existing food classes and 8 PithaNet classes. It is a research/prototype model, not a nutrition, medical, allergy, or portion-estimation system.

## Current Stage‑4 release

| Item | Verified value |
| --- | --- |
| Selected checkpoint | `checkpoint-645` |
| Adapter | LoRA, rank 4, alpha 16, dropout 0.05 |
| Quantization during training | bitsandbytes 4-bit NF4 QLoRA |
| Hardware | 2 × NVIDIA Tesla T4 |
| Training population | 6,872 samples (2,720 rehearsal + 4,152 unique PithaNet) |
| Held-out validation | 996 images |
| Overall accuracy | **90.86%** (905/996) |
| Macro accuracy | **89.09%** |
| Existing 19-food accuracy | 90.42% (576/637) |
| PithaNet 8-class accuracy | 91.64% (329/359) |

The only current model weight published here is [`models/stage4_27class/adapter_model.safetensors`](models/stage4_27class/adapter_model.safetensors), with its matching [`adapter_config.json`](models/stage4_27class/adapter_config.json). This is an adapter, not the ~4 GB base model.

## Supported labels

The authoritative 27-label list is [`metadata/labels.json`](metadata/labels.json). It includes the original 19 food labels plus: `bhapa_pitha`, `chitoi_pitha`, `jamai_pitha`, `nakshi_pitha`, `naru`, `patishapta_pitha`, `puli_pitha`, and `teler_pitha`.

## Evaluation and selection

| Checkpoint | Existing foods | PithaNet | Overall | Macro |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-500 | 89.95% | 91.64% | 90.56% | 88.89% |
| checkpoint-600 | 90.11% | 91.64% | 90.66% | 88.89% |
| **checkpoint-645** | **90.42%** | **91.64%** | **90.86%** | **89.09%** |

Machine-readable reports are in [`evaluation/reports`](evaluation/reports). The validation split is a held-out 996-image Stage‑4 split; it is not evidence of universal performance or calibrated confidence.

## Use the adapter

Install a compatible PyTorch build, then the pinned inference stack:

```bash
pip install -r requirements.txt
python inference/run_stage4.py --image /path/to/food.jpg
```

The launcher uses `model_type=qwen3_vl`, `template=qwen3_vl`, SDPA attention, `max_batch_size=1`, and `temperature=0`. Review its command before adding `--run`. See [inference guidance](docs/INFERENCE.md).

## Data and permission

PithaNet was used with direct permission from its author(s). Raw PithaNet images, other raw food datasets, manifests, and private correspondence are not distributed here. See [dataset provenance and permission](docs/DATASETS.md).

## Historical Stage‑3 release

Stage‑3 `checkpoint-348` remains documented as a historical 19-class baseline: 83/95 (87.37%) on its separate fixed internal benchmark. Its reports and visual assets remain under `evaluation/reports/stage3_*` and `assets/readme/`; they are not directly comparable to the Stage‑4 held-out 996-image validation result. See [training history](docs/TRAINING_HISTORY.md).

## Safety and limitations

A closed-set label is a candidate only. Validate exact output, handle unknown or non-food images separately, and require user confirmation before any nutrition lookup or meal record. Do not use this model for medical, allergy, ingredient-safety, calorie, nutrient, or portion-weight decisions. See [limitations](docs/LIMITATIONS.md) and [deployment guidance](docs/DEPLOYMENT.md).
