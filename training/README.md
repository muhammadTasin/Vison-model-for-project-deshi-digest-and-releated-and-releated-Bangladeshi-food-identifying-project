# Training utilities

This directory contains validation helpers and a non-executing Stage-4 command
template. It does not contain datasets, credentials, or a complete reproducible
training environment.

## Validate manifests

```bash
python training/validate_manifest.py path/to/train.jsonl
python training/validate_images.py path/to/train.jsonl --check-decode
```

The manifest validator expects MS-SWIFT-style JSONL records with a `messages`
array, one assistant label, and a non-empty `images` array. Labels are
canonicalized through the repository's exact alias map. The image validator
checks local paths and optionally decodes files with Pillow.

## Stage-4 template

`train_lora.example.sh` records the recommended shape of the next experiment.
Review and replace every placeholder, confirm dataset provenance and benchmark
isolation, pin the full environment, and test the command in a disposable
workspace before running it. Training must not be launched automatically.

Preserve Stage-3 `checkpoint-348` as a fallback and evaluate multiple Stage-4
checkpoints against both the unchanged internal benchmark and a separate
external benchmark.
