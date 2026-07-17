# Inference example

`predict.py` is a cautious, experimental adapter-loading example. It validates
the two required adapter files, loads a compatible Qwen3-VL base model, attaches
the PEFT adapter, generates a short closed-set response, and accepts only an
exact canonical label or exact alias.

Install a PyTorch build suitable for your platform, then install the repository
requirements. Extract the release archive and run:

```bash
python inference/predict.py image.jpg \
  --adapter ./checkpoint-348 \
  --aliases inference/label_aliases.json
```

The script intentionally reports `confidence: "uncalibrated"`. The available
artifacts do not provide calibrated probabilities or validated rejection
thresholds. A syntactically valid class is not automatically a high-confidence
prediction.

This script has not been rerun in the original Kaggle environment. Pin the
base-model revision and dependencies, validate it on the fixed benchmark, and
test unknown/non-food inputs before application use. See
[`docs/INFERENCE.md`](../docs/INFERENCE.md).
