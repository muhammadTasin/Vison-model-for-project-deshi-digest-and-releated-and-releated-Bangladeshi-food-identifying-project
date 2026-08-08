# Inference

The current adapter is `../models/stage4_27class`; it requires the separately obtained `Qwen/Qwen3-VL-2B-Instruct` base model. Install the pinned stack in `../requirements.txt`, then review the deterministic MS-Swift command:

```bash
python inference/run_stage4.py --image /path/to/image.jpg
python inference/run_stage4.py --image /path/to/image.jpg --run
```

The Stage‑4 launcher specifies `model_type=qwen3_vl`, `template=qwen3_vl`, SDPA attention, `max_batch_size=1`, and `temperature=0`. Variable image token lengths are why batching remains one. `predict.py` is retained as a historical experimental Transformers/PEFT example; it is not the recommended Stage‑4 launch path.
