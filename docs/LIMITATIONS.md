# Limitations

## Evidence limitations

The headline result is **87.37% accuracy on the fixed internal 95-image,
19-class benchmark**. With only five images per class, one result changes a
per-class rate by 20 percentage points. Dataset provenance, near-duplicate
leakage, source independence, and external generalization have not been fully
audited. Repeated model selection on the same set may also overfit development
decisions to this benchmark.

## Visual and cultural variation

Performance can change with plate style, lighting, blur, camera quality, angle,
occlusion, restaurant versus home presentation, regional recipes, garnish,
portion size, and serving vessel. A canonical class identifier can hide real
regional naming and recipe variation; product copy should not imply that one
presentation is universally definitive.

## Similar and mixed foods

Stage 3 demonstrates confusion among similar desserts and rice dishes,
especially `roshmalai`, `morog_polao`, and `kacha_golla`. Mixed plates and
multiple visible foods have not been adequately evaluated. The model should not
be assumed to enumerate ingredients or every dish in an image.

## Closed-set bias

The validated vocabulary contains only 19 classes. A closed-set prompt can
force unsupported foods, non-food images, or ambiguous inputs into a supported
class. Current evidence does not establish unknown-detection precision/recall
or non-food rejection accuracy. User confirmation and explicit rejection paths
are required.

## Confidence

Generated labels are not calibrated probabilities. The current artifacts do
not establish reliable confidence thresholds. Do not derive a numeric score
from token text or map every syntactically valid class to `high` confidence.

## Nutrition and health

Images alone do not provide exact ingredients, cooking method, hidden fats,
serving weight, calories, nutrients, allergens, or contamination risk. The
model must not make medical, allergy, or safety-critical decisions. Use a
verified nutrition database only after the user confirms dish and portion.

## Privacy and misuse

Meal images can reveal location, routine, health context, religion, culture,
household conditions, faces, documents, and device metadata. Minimize
collection and retention. Do not repurpose private user images for training
without explicit consent. Do not use food predictions to infer sensitive
traits or eligibility for services.

## Software and release limitations

The distributed artifact is a LoRA adapter and requires the compatible
Qwen3-VL base model and processor stack. Dependency, hardware, quantization,
and dtype differences can change behavior. The example inference script has not
been rerun in the original Kaggle environment. No GitHub Release asset or
open-source repository license was found during the 2026-07-17 audit.

## Required validation before broader use

- Audit source licenses and exact Stage-3 dataset lineage.
- Deduplicate and verify split isolation.
- Add held-out external data across devices, regions, and presentation styles.
- Measure macro F1, balanced accuracy, calibration, unknown/non-food behavior,
  JSON validity, and mixed-plate recall.
- Perform privacy, security, accessibility, latency, and failure-mode reviews.
- Establish user correction, monitoring, versioning, rollback, and incident
  processes.
