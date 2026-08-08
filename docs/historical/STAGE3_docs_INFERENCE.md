# Inference

## Adapter requirements

This release is a LoRA adapter, not the complete model. You need:

1. The original compatible `Qwen/Qwen3-VL-2B-Instruct` base model
2. A Transformers version with Qwen3-VL support
3. PEFT
4. `qwen-vl-utils` or an equivalent compatible processor
5. The extracted adapter folder
6. A PyTorch build suitable for the target CPU or GPU

Exported checkpoint metadata records `transformers>=4.57`,
`qwen_vl_utils>=0.0.14`, and PEFT 0.19.1. Newer versions can introduce API or
output differences; validate the complete environment before deployment.

Before loading, verify:

```text
checkpoint-348/adapter_config.json
checkpoint-348/adapter_model.safetensors
```

## Conceptual loading sequence

```python
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH = "./checkpoint-348"

# Load the compatible Qwen3-VL model class supported by the pinned
# Transformers environment, then load the LoRA adapter.
#
# base_model = CompatibleQwen3VLClass.from_pretrained(BASE_MODEL, ...)
# model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
```

[`inference/predict.py`](../inference/predict.py) expands this into an
experimental command-line example based on the class name recorded in exported
metadata. It has not been rerun in the original Kaggle environment, so treat it
as a starting point, not guaranteed copy-paste deployment code.

## Closed-set output

The historical benchmark prompt asked for exactly one identifier from its
allowed list. Production integration needs more than that because a closed-set
model can force unsupported food or non-food images into a known class.

Normalize only exact labels through
[`label_aliases.json`](../inference/label_aliases.json). Do not use substring
matching: a sentence that happens to contain `kebab` is not a valid structured
prediction.

## Recommended application schema

Successful prediction after schema validation and an application-defined
confidence policy:

```json
{
  "status": "ok",
  "food_id": "hilsa_fish",
  "display_name": "Hilsa Fish",
  "confidence": "high",
  "alternatives": []
}
```

Uncertain prediction:

```json
{
  "status": "unknown",
  "food_id": null,
  "display_name": null,
  "confidence": "low",
  "alternatives": [
    "morog_polao",
    "bangladeshi_biriyani"
  ]
}
```

Non-food image:

```json
{
  "status": "not_food",
  "food_id": null,
  "display_name": null,
  "confidence": "high",
  "alternatives": []
}
```

These are interface examples, not claims that the current adapter already
provides calibrated confidence or reliable unknown/non-food detection. Do not
invent numeric probabilities. The provided CLI labels accepted predictions as
`uncalibrated`; downstream services must not convert that to `high` without a
validated confidence mechanism.

## Validation checklist

- Pin and record base-model revision, package versions, dtype, and device map.
- Confirm the adapter targets match the base model.
- Validate image type, dimensions, byte size, and decoder behavior.
- Bound generation length and request time.
- Reject output that is not an exact supported identifier or valid schema.
- Test unsupported dishes, non-food inputs, corrupt images, and mixed plates.
- Confirm user corrections before nutrition lookup and meal logging.
