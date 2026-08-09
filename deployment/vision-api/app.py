
import os
import secrets

os.environ.setdefault(
    "IMAGE_MAX_TOKEN_NUM",
    "128"
)

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false"
)

os.environ.setdefault(
    "SWIFT_SINGLE_DEVICE_MODE",
    "1"
)


from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
import asyncio
import tempfile
import time

import torch

from PIL import Image

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Header,
)

from transformers import BitsAndBytesConfig

from swift.infer_engine import (
    TransformersEngine,
    InferRequest,
    RequestConfig,
)


# =====================================================================
# CONFIG
# =====================================================================

BASE_MODEL = os.environ.get(
    "DESHI_BASE_MODEL",
    "/models/qwen3_vl_2b_instruct"
)

ADAPTER = os.environ.get(
    "DESHI_ADAPTER_PATH",
    "/app/model_adapter"
)

DEVICE_ID = int(
    os.environ.get(
        "DESHI_GPU_ID",
        "0"
    )
)

MAX_UPLOAD_MB = int(
    os.environ.get(
        "DESHI_MAX_UPLOAD_MB",
        "10"
    )
)

MAX_UPLOAD_BYTES = (
    MAX_UPLOAD_MB
    *
    1024
    *
    1024
)


API_TOKEN = os.environ.get(
    "DESHI_VISION_API_TOKEN",
    ""
).strip()


VALID_LABELS = {

    "bakorkhani",
    "bangladeshi_biryani",
    "beguni",
    "chickpea_curry",
    "egg_omelette",
    "fuchka",
    "haleem",
    "hilsa_fish",
    "kacha_golla",
    "kala_bhuna",
    "kebab",
    "khichuri",
    "morog_polao",
    "nehari",
    "paratha",
    "potato_bhorta",
    "roshgolla",
    "roshmalai",
    "sweet_yogurt",

    "bhapa_pitha",
    "chitoi_pitha",
    "jamai_pitha",
    "nakshi_pitha",
    "naru",
    "patishapta_pitha",
    "puli_pitha",
    "teler_pitha",
}


PROMPT_MESSAGES = [

    {
        "role":
            "system",

        "content":
            (
                "You are a closed-set Bangladeshi food image classifier. "
                "Choose exactly one food_id from this list:\n"
                "bakorkhani, bangladeshi_biryani, beguni, "
                "chickpea_curry, egg_omelette, fuchka, haleem, "
                "hilsa_fish, kacha_golla, kala_bhuna, kebab, "
                "khichuri, morog_polao, nehari, paratha, "
                "potato_bhorta, roshgolla, roshmalai, sweet_yogurt, "
                "bhapa_pitha, chitoi_pitha, jamai_pitha, "
                "nakshi_pitha, naru, patishapta_pitha, "
                "puli_pitha, teler_pitha\n"
                "Return only the exact food_id. "
                "Do not return JSON, explanation, punctuation, "
                "or extra text."
            )
    },

    {
        "role":
            "user",

        "content":
            (
                "Classify the main visible food. "
                "Return only its exact food_id "
                "from the allowed list."
            )
    }
]


ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


STATE = {
    "engine": None,
    "request_config": None,
    "load_seconds": None,
}


# One model on one GPU.
INFERENCE_LOCK = asyncio.Lock()


# =====================================================================
# MODEL
# =====================================================================

def load_engine():

    adapter_path = Path(
        ADAPTER
    )

    if not adapter_path.exists():

        raise RuntimeError(
            f"Adapter not found: {ADAPTER}"
        )


    quantization_config = (
        BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=
                torch.float16,

            bnb_4bit_use_double_quant=True,
        )
    )


    start = time.perf_counter()


    engine = TransformersEngine(

        BASE_MODEL,

        adapters=[
            ADAPTER
        ],

        model_type="qwen3_vl",

        template_type="qwen3_vl",

        torch_dtype=torch.float16,

        attn_impl="sdpa",

        device_map={
            "": DEVICE_ID
        },

        quantization_config=
            quantization_config,

        max_batch_size=1,
    )


    request_config = RequestConfig(

        max_tokens=16,

        temperature=0,
    )


    return (
        engine,
        request_config,
        time.perf_counter()
        -
        start,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):

    (
        STATE["engine"],
        STATE["request_config"],
        STATE["load_seconds"],
    ) = load_engine()


    yield


    STATE["engine"] = None

    STATE["request_config"] = None


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


app = FastAPI(

    title=
        "Deshi Digest Vision API",

    version=
        "1.0.0",

    lifespan=
        lifespan,
)


# =====================================================================
# INFERENCE
# =====================================================================

def normalize_food_id(text):

    return (
        str(text)
        .strip()
        .strip(
            " \n\t\r`'\""
        )
    )


def run_prediction(
    image_path
):

    request = InferRequest(

        messages=
            deepcopy(
                PROMPT_MESSAGES
            ),

        images=[
            str(image_path)
        ],
    )


    start = time.perf_counter()


    responses = (
        STATE[
            "engine"
        ].infer(

            [request],

            request_config=
                STATE[
                    "request_config"
                ],

            use_tqdm=False,
        )
    )


    latency = (
        time.perf_counter()
        -
        start
    )


    raw = (
        responses[0]
        .choices[0]
        .message
        .content
    )


    return (
        normalize_food_id(
            raw
        ),
        str(raw),
        latency,
    )


def require_api_token(
    authorization: str | None
):

    if not API_TOKEN:

        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    prefix = "Bearer "

    if (
        not authorization
        or not authorization.startswith(prefix)
    ):

        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
            headers={
                "WWW-Authenticate":
                    "Bearer"
            },
        )

    supplied_token = (
        authorization[
            len(prefix):
        ]
        .strip()
    )

    if (
        not supplied_token
        or not secrets.compare_digest(
            supplied_token,
            API_TOKEN,
        )
    ):

        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
            headers={
                "WWW-Authenticate":
                    "Bearer"
            },
        )


# =====================================================================
# ROUTES
# =====================================================================

@app.get("/")
async def root():

    return {
        "service":
            "Deshi Digest Vision API",

        "version":
            "1.0.0",

        "endpoint":
            "/predict-food",
    }


@app.get("/health")
async def health():

    return {

        "status":
            "ok",

        "model_loaded":
            STATE[
                "engine"
            ] is not None,

        "checkpoint":
            "checkpoint-645",

        "classes":
            27,

        "device":
            (
                torch.cuda.get_device_name(
                    DEVICE_ID
                )
                if torch.cuda.is_available()
                else
                "cpu"
            ),

        "model_load_seconds":
            STATE[
                "load_seconds"
            ],
    }


@app.post("/predict-food")
async def predict_food(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):

    require_api_token(
        authorization
    )

    if STATE["engine"] is None:

        raise HTTPException(
            status_code=503,
            detail="Model unavailable.",
        )


    if (
        file.content_type
        not in
        ALLOWED_MIME
    ):

        raise HTTPException(
            status_code=415,
            detail="Unsupported image type.",
        )


    contents = await file.read()


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Empty image.",
        )


    if len(contents) > MAX_UPLOAD_BYTES:

        raise HTTPException(

            status_code=413,

            detail=(
                f"Maximum upload size "
                f"is {MAX_UPLOAD_MB} MB."
            ),
        )


    suffix = {

        "image/jpeg":
            ".jpg",

        "image/png":
            ".png",

        "image/webp":
            ".webp",

        "image/bmp":
            ".bmp",

    }[
        file.content_type
    ]


    temp_path = None


    try:

        with tempfile.NamedTemporaryFile(

            suffix=suffix,

            delete=False,

        ) as tmp:

            tmp.write(
                contents
            )

            temp_path = Path(
                tmp.name
            )


        try:

            with Image.open(
                temp_path
            ) as image:

                image.verify()

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Invalid image.",
            )


        async with INFERENCE_LOCK:

            (
                food_id,
                raw_output,
                latency,
            ) = await asyncio.to_thread(

                run_prediction,

                temp_path,
            )


        if food_id not in VALID_LABELS:

            raise HTTPException(

                status_code=500,

                detail={
                    "message":
                        "Invalid model output.",

                    "raw_output":
                        raw_output,
                },
            )


        return {

            "food_id":
                food_id,

            "valid":
                True,

            "latency_seconds":
                round(
                    latency,
                    4
                ),

            "model":
                "deshi-digest-cp645",

            "filename":
                file.filename,
        }


    finally:

        if (
            temp_path is not None
            and
            temp_path.exists()
        ):

            try:
                temp_path.unlink()

            except Exception:
                pass
