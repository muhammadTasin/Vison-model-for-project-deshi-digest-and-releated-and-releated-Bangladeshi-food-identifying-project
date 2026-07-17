# Training History

## Summary

| Stage | Correct | Accuracy | Checkpoint evidence | Outcome |
|---|---:|---:|---|---|
| Stage 1 | 79/95 | 83.16% | `checkpoint-901` | Previous baseline |
| Stage 2 | 78/95 | 82.11% | Checkpoint not verified | Did not outperform Stage 1 |
| Stage 3 | 83/95 | 87.37% | `checkpoint-348` | Current best on the fixed benchmark |

The comparison uses the same stated 95-image benchmark. Stage 3 improved by
4.21 percentage points over Stage 1 and 5.26 percentage points over Stage 2.

## Stage 1

- Accuracy: 83.16%
- Correct: 79/95
- Relevant checkpoint: `checkpoint-901`

The exported Stage-3 arguments refer to an adapter and checkpoint directory
ending in `stage3_v2_ready/checkpoint-901`, which supports the checkpoint name.
The complete Stage-1 run configuration and reports were not available in the
repository and have not been independently reproduced.

## Stage 2

- Accuracy: 82.11%
- Correct: 78/95

Stage 2 did not outperform Stage 1. The available artifacts do not identify a
verified Stage-2 checkpoint, manifest, or complete hyperparameter record, so no
causal explanation is claimed.

More examples alone do not guarantee an improvement. Performance can decline
when classes are imbalanced, similar negative examples are missing, labels are
noisy, duplicates leak across splits, the learning rate is too high, or
rehearsal examples from previously strong classes are insufficient.

## Stage 3

- Accuracy: 87.37%
- Correct: 83/95
- Best checkpoint: `checkpoint-348`
- Exported checkpoint path:
  `/kaggle/working/deshi_digest_stage3_2gpu_20260717-164850/checkpoint-348`

Stage 3 is the current best because it has the highest verified accuracy among
the three reported stages on the fixed internal benchmark. This selection does
not establish superiority on external or real-world data.

### Observed Stage-3 configuration

| Parameter | Value recovered from checkpoint metadata |
|---|---|
| Base model | `Qwen/Qwen3-VL-2B-Instruct` |
| Framework | MS-SWIFT 4.4.1 |
| PEFT | 0.19.1 |
| Adapter | LoRA |
| Rank / alpha / dropout | 8 / 32 / 0.05 |
| Target | Language-model linear projection modules |
| Vision tower / aligner | Frozen |
| Hardware | 2 × NVIDIA T4 |
| Precision | FP16 |
| Epochs | 1 |
| Per-device train batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | `3e-5` |
| Weight decay | 0.1 |
| Scheduler | Cosine |
| Warm-up ratio | 0.05 |
| Maximum sequence length | 1024 |
| Seed / data seed | 42 / 42 |

The exported state records a world size of two and references Stage-3 train and
validation manifests. Reproduction also requires the exact base-model revision,
images, manifest lineage, package lock, notebook/runtime configuration, and
evaluation benchmark, none of which is completely captured in this repository.

## Why Stage 3 remains the fallback

The next stage should be accepted only if it improves the primary weak classes
without severe regression in previously strong classes. Keep Stage 3 immutable,
evaluate multiple Stage-4 checkpoints against the unchanged 95-image benchmark,
and add a separately governed external benchmark before changing the default.

## Stage-4 experiment plan

1. Resume from Stage-3 `checkpoint-348`.
2. Use a small learning rate and record every effective configuration value.
3. Build a balanced targeted manifest for `morog_polao`, `roshmalai`, and
   `kacha_golla`.
4. Add hard negatives from `bangladeshi_biriyani`, `khichuri`, `beguni`,
   `roshogolla`, `sweet_yogurt`, `bakorkhani`, and `potato_bhorta`.
5. Include rehearsal examples from all high-performing classes.
6. Include secondary targets `fuchka`, `nehari`, `paratha`, and `haleem`.
7. Evaluate multiple checkpoints; do not select on training loss alone.
8. Keep the original benchmark unchanged and add a separate external set.
9. Reject checkpoints that trade weak-class gains for severe broad regression.
10. Preserve Stage 3 as the stable fallback.

No training is started by this repository documentation.
