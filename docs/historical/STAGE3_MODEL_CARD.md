---
base_model: Qwen/Qwen3-VL-2B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - lora
  - vision-language
  - bangladeshi-food
  - qwen3-vl
---

# Model Card: Deshi Digest Vision Model

## Model summary

The Deshi Digest Vision Model is a PEFT LoRA adapter for
`Qwen/Qwen3-VL-2B-Instruct`. Its purpose is controlled, image-based
classification of 19 Bangladeshi and Bengali food categories for research and
prototype use.

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` |
| Model type | Vision-language model |
| Adapter | LoRA, rank 8, alpha 32, dropout 0.05 |
| Current stage | Stage 3 |
| Best checkpoint | `checkpoint-348` |
| Framework | MS-SWIFT 4.4.1 (observed) |
| Benchmark result | 83/95, or 87.37% |
| Benchmark scope | Fixed internal benchmark; 19 classes × 5 images |

Supported canonical classes are `bakorkhani`, `bangladeshi_biriyani`,
`beguni`, `chickpea_curry`, `egg_omelette`, `fuchka`, `haleem`, `hilsa_fish`,
`kacha_golla`, `kala_bhuna`, `kebab`, `khichuri`, `morog_polao`, `nehari`,
`paratha`, `potato_bhorta`, `roshogolla`, `roshmalai`, and `sweet_yogurt`.

## Intended use

Suitable uses include:

- Research and educational demonstrations
- Closed-set Bangladeshi food classification
- Deshi Digest prototype integration with user confirmation
- Baseline comparisons
- Controlled food-recognition experiments

Any prototype should use explicit unknown/non-food handling and a separate,
verified nutrition database.

## Out-of-scope use

This model is not suitable for:

- Medical diagnosis or treatment decisions
- Allergy or ingredient-safety detection
- Exact nutrition or calorie calculation
- Exact portion-weight estimation
- Universal food recognition
- Unsupported dishes without unknown handling
- Safety-critical or fully autonomous decisions

## Training information

The adapter was trained on `Qwen/Qwen3-VL-2B-Instruct` with LoRA and
MS-SWIFT in Kaggle Notebooks using two NVIDIA T4 GPUs. Exported metadata records
MS-SWIFT 4.4.1, PEFT 0.19.1, FP16, one epoch, per-device batch size 1,
gradient accumulation 4, learning rate `3e-5`, cosine scheduling, and 5%
warm-up for Stage 3. It also records frozen vision and aligner modules.

Stage history:

| Stage | Correct | Accuracy | Relevant checkpoint |
|---|---:|---:|---|
| Stage 1 | 79/95 | 83.16% | `checkpoint-901` |
| Stage 2 | 78/95 | 82.11% | Not verified in available artifacts |
| Stage 3 | 83/95 | 87.37% | `checkpoint-348` |

Stage 2 did not outperform Stage 1. Stage 3 is the current best model only with
respect to the fixed internal benchmark. See
[TRAINING_HISTORY.md](docs/TRAINING_HISTORY.md).

## Evaluation

The benchmark contains 95 images: five for each of 19 supported classes.
Scoring uses normalized exact-label comparison. Stage 3 achieved **87.37%
accuracy on the fixed internal 95-image, 19-class benchmark**.

| Class | Correct | Total | Accuracy |
|---|---:|---:|---:|
| `bakorkhani` | 4 | 5 | 80% |
| `bangladeshi_biriyani` | 5 | 5 | 100% |
| `beguni` | 5 | 5 | 100% |
| `chickpea_curry` | 5 | 5 | 100% |
| `egg_omelette` | 5 | 5 | 100% |
| `fuchka` | 4 | 5 | 80% |
| `haleem` | 5 | 5 | 100% |
| `hilsa_fish` | 5 | 5 | 100% |
| `kacha_golla` | 3 | 5 | 60% |
| `kala_bhuna` | 5 | 5 | 100% |
| `kebab` | 5 | 5 | 100% |
| `khichuri` | 5 | 5 | 100% |
| `morog_polao` | 2 | 5 | 40% |
| `nehari` | 4 | 5 | 80% |
| `paratha` | 5 | 5 | 100% |
| `potato_bhorta` | 4 | 5 | 80% |
| `roshogolla` | 5 | 5 | 100% |
| `roshmalai` | 2 | 5 | 40% |
| `sweet_yogurt` | 5 | 5 | 100% |

The most frequent observed confusion was `roshmalai` → `roshogolla` (2).
All other recorded confusion pairs occurred once. The primary weak classes are
`morog_polao`, `roshmalai`, and `kacha_golla`.

The five samples per class provide limited statistical evidence and may not
represent new cameras, households, restaurants, regions, or presentation
styles. This benchmark cannot establish production-level generalization.

## Bias and limitations

- Plate style, lighting, blur, camera quality, and viewpoint can change results.
- Restaurant and home presentations may differ from benchmark imagery.
- Recipes, naming, and appearance vary regionally.
- Similar-looking desserts are a demonstrated weakness.
- Mixed plates and multiple visible foods are not adequately evaluated.
- The model has a limited, closed vocabulary and may force unsupported images
  into a known class.
- Class balance in the benchmark does not imply balance in deployment traffic.
- Data duplication, near duplication, label noise, and train/benchmark leakage
  have not been independently ruled out.
- Categorical output is not a calibrated probability.

## Ethical and privacy considerations

Do not train on private user images without informed consent, a defined
purpose, appropriate access controls, and a retention/deletion policy. Avoid
logging raw images or personal nutrition records by default. Users must be able
to correct predictions. Do not infer health status, religion, ethnicity, or
other sensitive attributes from meal images.

## Model release

This is a LoRA adapter release. It requires the separately obtained compatible
Qwen3-VL base model. The adapter archive was not found in GitHub Releases during
the 2026-07-17 audit; consult the repository README for owner upload steps.

## Citation

No paper or archival release citation has been verified. Until one exists, use
this provisional repository citation and include the commit or release tag:

```bibtex
@software{deshi_digest_vision_model,
  title  = {Deshi Digest Vision Model},
  author = {Tasin, Md. Tasfiq},
  year   = {2026},
  url    = {https://github.com/muhammadTasin/Vison-model-for-project-deshi-digest-and-releated-and-releated-Bangladeshi-food-identifying-project},
  note   = {Provisional repository citation; cite a specific commit or release}
}
```

## License

No open-source license has been selected. The adapter, Qwen3-VL base model,
software dependencies, and datasets may have separate terms. Review
[LICENSE](LICENSE) and all upstream licenses before use or redistribution.
