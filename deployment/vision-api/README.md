# Deshi Digest CP645 Vision API

Production FastAPI inference service for the Deshi Digest Stage-4 CP645 model.

## Model

Base model:
- Qwen/Qwen3-VL-2B-Instruct

Fine-tuned adapter:
- ../../models/stage4_27class/

The Qwen base weights are intentionally NOT stored in this repository.

## API

Public health endpoint:

GET /health

Protected inference endpoint:

POST /predict-food

The inference endpoint requires:

Authorization: Bearer <DESHI_VISION_API_TOKEN>

## Local configuration

Set:

DESHI_BASE_MODEL=/path/to/qwen3_vl_2b_instruct
DESHI_ADAPTER_PATH=/path/to/models/stage4_27class
DESHI_VISION_API_TOKEN=<strong-secret>

Never commit the real API token.
