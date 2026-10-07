# Limitations

## Evidence and scope

Stage‑4 achieved 90.86% overall and 89.09% macro accuracy on a 996-image held-out 27-class validation split. This does not prove performance across new cameras, households, restaurants, regions, presentation styles, or unsupported foods. Two audits of the Stage‑4 split (see [training history](TRAINING_HISTORY.md)) found no exact image-reference overlap between the train and validation manifests (0), no duplicate references and no conflicting labels. A content audit of the 8 PithaNet classes (4,152 train and 359 validation images, all hashed) found no byte-identical image across train and validation, and showed that the 356 same-name PithaNet pairs are different images. It did find 2 PithaNet validation images with a near-duplicate training image of the same class (dHash distance ≤ 5; visually the same scene from a slightly different position), so the validation split is not fully free of near-duplicate leakage. The 19 existing-food classes were not content-audited, label noise was not checked, and shifted or cropped near-duplicates beyond the hash thresholds can remain undetected. External generalization also still requires continued audit.

## Closed-set and confidence limits

The vocabulary has 27 labels. Unsupported dishes, non-food images, mixed plates, blur, and similar recipes can still receive a known label. No published result establishes calibrated confidence, unknown detection, non-food rejection, or mixed-plate component recall. Treat a label as a candidate and require user confirmation.

## Health, privacy, and release limits

The model cannot reliably infer ingredients, allergens, portion weight, calories, nutrients, or health status. Do not use it for medical, allergy, or safety-critical decisions. Meal images may expose sensitive context; minimize retention and never train on private images without consent. The released artifact is an adapter requiring a compatible Qwen base model and pinned runtime; hardware and dependency changes can affect output.
