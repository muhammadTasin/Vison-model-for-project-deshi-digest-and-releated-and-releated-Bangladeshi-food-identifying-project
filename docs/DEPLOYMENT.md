# Deployment Guidance

## Safe integration flow

```text
User image
    |
    v
Image validation
    |
    v
Vision-model inference
    |
    v
Schema validation
    |
    v
Confidence / unknown handling
    |
    v
User confirmation
    |
    v
Verified nutrition database lookup
    |
    v
Meal log
```

A low-confidence or uncalibrated prediction must never silently become a final
meal record. Nutrition must come from a separately maintained, verified
database after food identity and portion details are confirmed; it must not be
guessed from visual output alone.

## API wrapper

Place the model behind a versioned API that accepts one bounded image upload and
returns a validated schema. Keep model loading outside request handlers, apply
authentication and rate limits where appropriate, and reject unsupported media
types before decoding. Separate inference output from user-confirmed meal data.

Record the base-model revision, adapter release/tag or checksum, prompt version,
alias-map version, and post-processing version with every prediction event.

## Input controls

- Enforce maximum compressed bytes, decoded pixels, dimensions, and image count.
- Decode in a resource-constrained worker and reject malformed images.
- Re-encode uploads to remove unnecessary metadata when retention is allowed.
- Apply request and generation timeouts.
- Limit generated tokens and exact accepted labels.
- Do not fetch arbitrary remote image URLs from the inference worker unless SSRF
  protections and network isolation are in place.

## Confidence and unknown handling

The current benchmark does not validate confidence calibration, unknown
detection, or non-food rejection. Treat a closed-set label as a candidate, not
a confirmed fact. Add a separately validated rejection mechanism, expose
alternatives where supported, and ask the user to confirm or correct the dish.

Categorical values such as `high` and `low` must be tied to documented,
measured policies. Do not invent numeric confidence probabilities from generated
text.

## Compute and performance

GPU inference is the expected practical route for interactive latency. CPU
operation may be possible but should be load-tested for memory, latency, and
timeouts before being offered. The observed benchmark throughput of about 2.08
samples/second is environment-specific and is not a service-level guarantee.

Use bounded queues, concurrency limits, health checks, graceful overload
responses, and warm workers. Cache immutable model artifacts by verified
revision/checksum. Cache user-specific predictions only with consent and a
defined expiry; a content hash can itself be sensitive and must be protected.

## Logging and privacy

Log operational metadata, validation outcomes, latency, model version, and
schema status. Avoid raw images, prompts containing personal data, exact user
location, and nutrition records by default. Encrypt retained data, restrict
access, define deletion and retention periods, and provide a correction path.

Never reuse private user images for training without informed consent specific
to that purpose.

## Monitoring

Monitor:

- Invalid image and schema rates
- Unknown/non-food and user-correction rates
- Per-class confirmation rates and confusion trends
- Latency, timeout, memory, queue depth, and failure distributions
- Data drift across devices, lighting, presentation, and regions
- Nutrition-lookup mismatches and failed user confirmations
- Model, adapter, prompt, processor, and alias-map versions

Use confirmed user corrections only through a reviewed, privacy-preserving data
pipeline. A correction is evidence for review, not automatically a training
label.

## Release and rollback

Deploy immutable adapter artifacts identified by checksum. Run schema, safety,
latency, unknown/non-food, regression, and external benchmark checks before
promotion. Preserve Stage 3 as a rollback target. Stage 4 should not replace it
if targeted gains cause severe regression in previously strong classes.
