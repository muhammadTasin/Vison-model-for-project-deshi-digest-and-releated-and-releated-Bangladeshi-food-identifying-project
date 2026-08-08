# Deployment guidance

Deploy the immutable Stage‑4 adapter from `models/stage4_27class` with the recorded runtime configuration: `qwen3_vl`, `qwen3_vl` template, SDPA, one-image batches, and temperature zero. Record the adapter checksum, base-model revision, prompt, alias-map version, and post-processing version with each prediction.

Use a bounded image-validation and inference service, validate exact supported labels, provide explicit unknown/non-food handling, and require user confirmation before a separate verified nutrition lookup or meal record. Do not retain raw images or reuse them for training without consent. Maintain Stage‑3 only as a historical rollback/reference baseline; do not claim direct benchmark comparability.
