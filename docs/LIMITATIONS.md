# Limitations

## Evidence and scope

Stage‑4 achieved 90.86% overall and 89.09% macro accuracy on a 996-image held-out 27-class validation split. This does not prove performance across new cameras, households, restaurants, regions, presentation styles, or unsupported foods. Split independence, near-duplicate leakage, label noise, and external generalization still require continued audit.

## Closed-set and confidence limits

The vocabulary has 27 labels. Unsupported dishes, non-food images, mixed plates, blur, and similar recipes can still receive a known label. No published result establishes calibrated confidence, unknown detection, non-food rejection, or mixed-plate component recall. Treat a label as a candidate and require user confirmation.

## Health, privacy, and release limits

The model cannot reliably infer ingredients, allergens, portion weight, calories, nutrients, or health status. Do not use it for medical, allergy, or safety-critical decisions. Meal images may expose sensitive context; minimize retention and never train on private images without consent. The released artifact is an adapter requiring a compatible Qwen base model and pinned runtime; hardware and dependency changes can affect output.
