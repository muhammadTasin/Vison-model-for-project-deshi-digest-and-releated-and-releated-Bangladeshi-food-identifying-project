# Training utilities

Stage‑4 uses Qwen3‑VL‑2B‑Instruct with 4-bit NF4 QLoRA (rank 4, alpha 16, dropout 0.05). `train_stage4.py` prints the recorded MS-Swift command by default and executes only with `--run`:

```bash
python training/train_stage4.py --train-manifest /secure/train.jsonl --val-manifest /secure/val.jsonl
```

The local manifests and images are deliberately not included. Validate authorized local inputs with `validate_manifest.py` and `validate_images.py`. Historical `train_lora.example.sh` is retained as a Stage‑3-to‑Stage‑4 planning artifact, not as the completed Stage‑4 configuration.
