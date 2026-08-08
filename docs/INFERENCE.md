# Stage‑4 inference

The published Stage‑4 artifact is a LoRA adapter, not the Qwen base model. Obtain `Qwen/Qwen3-VL-2B-Instruct` separately and use the adapter in `models/stage4_27class`.

Install the pinned environment in `requirements.txt`, then use:

```bash
python inference/run_stage4.py --image /path/to/food.jpg
```

The reviewed command uses `model_type=qwen3_vl`, `template=qwen3_vl`, SDPA attention, `max_batch_size=1`, and `temperature=0`. It prints before execution unless `--run` is specified.

Treat generated labels as uncalibrated candidates. Accept only exact labels from `metadata/labels.json`, separately handle unknown/non-food inputs, and require user confirmation before nutrition lookup or logging. The older `inference/predict.py` remains a historical experimental Transformers/PEFT example.
