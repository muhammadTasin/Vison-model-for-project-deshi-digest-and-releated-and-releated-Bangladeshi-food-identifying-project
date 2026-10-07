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

## Stage‑4 manifest audit

Script: [`evaluation/audit_manifests.py`](../evaluation/audit_manifests.py) (standard library only). It reads the train and validation manifests and prints aggregate counts only; image references and file names are never written out. The manifests themselves are not published. Recorded output: [`evaluation/reports/stage4_manifest_audit.json`](../evaluation/reports/stage4_manifest_audit.json).

```bash
python3 evaluation/audit_manifests.py <train.jsonl> <validation.jsonl> \
  --output evaluation/reports/stage4_manifest_audit.json
```

Output recorded for the Stage‑4 manifests (6,872 train rows, 996 validation rows):

| Check | Result |
| --- | --- |
| Train / validation rows | 6,872 / 996 (2,720 + 4,152 train; 637 + 359 validation, existing foods + PithaNet) |
| Classes present in each split | 27 / 27 |
| Exact image-reference overlap, train vs validation | **0** |
| Duplicate references inside train / inside validation | 0 / 0 |
| References with conflicting labels | 0 |
| Labels outside the 27-label set | none |
| PithaNet rows whose source folder name differs from the label | 0 |
| Same label and same file name in both splits | **356** (all PithaNet; 0 existing foods) |

How to read it:

- By exact image reference, Stage‑4 has **no** train/validation overlap (0).
- The 356 same-name pairs are PithaNet files that carry the same label and file name in the train and validation folders. The manifests cannot say whether these are the same image; the content audit below resolves it.
- The manifests contain no image bytes, so this audit cannot see content-level or near-duplicate images, or label noise.

## Stage‑4 content audit (PithaNet)

Script: [`evaluation/audit_content.py`](../evaluation/audit_content.py) (Pillow and numpy). It needs the original PithaNet image files in addition to the manifests, hashes every PithaNet image in the Stage‑4 manifests, and prints aggregate counts only: no file names, paths or image copies. Recorded output: [`evaluation/reports/stage4_content_audit.json`](../evaluation/reports/stage4_content_audit.json).

```bash
python3 evaluation/audit_content.py <train.jsonl> <validation.jsonl> \
  --image-root <directory containing Crop_Resize> --anchor Crop_Resize \
  --output evaluation/reports/stage4_content_audit.json
```

Scope: the 8 PithaNet classes only, 4,152 training and 359 validation images, all hashed (0 missing, 0 undecodable). Near-duplicate threshold: Hamming distance ≤ 5 on 64-bit hashes.

| Check | Result |
| --- | --- |
| SHA-256: validation images with a byte-identical training image | **0** |
| SHA-256: duplicate groups inside train / inside validation | 0 / 0 |
| dHash ≤ 5: train/validation pairs (validation images affected) | **2** (2), both same class |
| pHash ≤ 5: train/validation pairs (validation images affected) | **0** (0) |
| Validation images whose nearest training image is at distance 6–10 (dHash / pHash) | 6 / 6 (not reviewed) |
| The 356 same-name pairs: byte-identical / different content | **0 / 356** |
| The 356 same-name pairs: different content but within threshold (dHash / pHash) | 0 / 0 |

How to read it:

- No validation image is byte-identical to a training image, and the 356 same-name pairs are all different images, so they are file-name coincidences rather than copies.
- This is **not** a "no leakage" result. dHash found 2 of the 359 PithaNet validation images (2 of 996 overall) with a near-duplicate training image of the same class. A one-off visual check of those two pairs (not part of the script) showed the same dish on the same plate and background photographed from a slightly different position. pHash did not flag them at the threshold (their pHash distances were 10 and 12), so shifted, rotated or cropped copies can slip past a fixed threshold; further near-duplicates may exist beyond what these hashes detect.
- Size of the effect: removing those two images could move overall accuracy anywhere between 90.85% (903/994) and 91.05% (905/994). These are bounds computed from the published 905/996, not a re-evaluation.
- Not covered: the 19 existing-food classes (their images were not available to this audit) and label noise.

## Historical Stage‑3 note

Stage‑3 remains documented as a rollback/reference baseline. Its observed configuration was LoRA rank 8, alpha 32, dropout 0.05, one epoch, `3e-5` learning rate, batch 1 and accumulation 4 on two T4 GPUs. It achieved 83/95 (87.37%) on its fixed internal benchmark. Preserve its artifacts and use a separate governed evaluation if comparing future releases.
