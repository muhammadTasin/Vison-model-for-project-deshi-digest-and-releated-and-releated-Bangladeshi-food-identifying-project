#!/usr/bin/env bash
set -euo pipefail

# REVIEW BEFORE RUNNING. This template intentionally uses placeholders and does
# not start from repository data. Pin the exact environment and verify licenses,
# deduplication, split isolation, and manifest balance first.

BASE_MODEL="Qwen/Qwen3-VL-2B-Instruct"
STAGE3_ADAPTER="/path/to/checkpoint-348"
TRAIN_MANIFEST="/path/to/stage4_train.jsonl"
VALIDATION_MANIFEST="/path/to/stage4_validation.jsonl"
OUTPUT_DIR="/path/to/deshi_digest_stage4"

swift sft \
  --model "${BASE_MODEL}" \
  --adapters "${STAGE3_ADAPTER}" \
  --dataset "${TRAIN_MANIFEST}" \
  --val_dataset "${VALIDATION_MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_type qwen3_vl \
  --template qwen3_vl \
  --tuner_type lora \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --freeze_vit true \
  --freeze_aligner true \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --save_steps 100 \
  --save_total_limit 4 \
  --max_length 1024 \
  --seed 42 \
  --fp16 true
