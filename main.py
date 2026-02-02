"""
FastAPI server for TartuNLP Tahetorn_9B translation.
Runs Sami ↔ Finnish/Norwegian translation models on NVIDIA GPU.
Uses TartuNLP's Tahetorn_9B model (Tower-Plus-9B-based) with Transformers.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from translation_service import TranslationService
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure FastAPI with OpenAPI/docs under /translation/* prefix
app = FastAPI(
    title="Translation API",
    version="2.3.1",
    description="An API that provides translations using neural machine translation models. Developed by TartuNLP - the NLP research group of the University of Tartu.",
    openapi_url="/translation/openapi.json",
    docs_url="/translation/docs",
    redoc_url="/translation/redoc",
)


@app.middleware("http")
async def forwarded_prefix_middleware(request, call_next):
    """Honor X-Forwarded-Prefix so docs/OpenAPI work behind a path prefix."""
    prefix = request.headers.get("x-forwarded-prefix")
    if prefix:
        # Normalize: ensure leading slash and no trailing slash
        normalized = "/" + prefix.lstrip("/")
        request.scope["root_path"] = normalized.rstrip("/")
    return await call_next(request)

# CORS configuration - allow the frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173", "*"],  # Vite dev and preview ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize translation service (loads models on startup)
translation_service = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global translation_service
    logger.info("Starting translation service...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    translation_service = TranslationService()
    enabled_langs = translation_service.get_supported_languages()
    if enabled_langs:
        logger.info("Enabled API languages: %s", ", ".join(enabled_langs))
    logger.info("Translation service ready!")

from typing import List, Optional, Union


class Request(BaseModel):
    text: Union[str, List[str]]
    src: str  # Sami 639-3 code, "fin", or "nor"
    tgt: str  # Sami 639-3 code, "fin", or "nor"
    domain: Optional[str] = "general"
    application: Optional[str] = None


class Response(BaseModel):
    result: Union[str, List[str]]


class Domain(BaseModel):
    name: str
    code: str
    languages: List[str]


class Config(BaseModel):
    xml_support: Optional[bool] = True
    domains: List[Domain]

@app.get("/")
async def root():
    """Health check endpoint (kept for convenience)"""
    return {
        "status": "ok",
        "service": "Sami Translation API (Tahetorn_9B)",
        "cuda_available": torch.cuda.is_available(),
        "device": str(translation_service.device) if translation_service else "not initialized"
    }


# Provide the configuration endpoint under /translation to match TartuNLP API
@app.get("/translation", response_model=Config, tags=["translation"])
async def get_config(x_api_key: Optional[str] = None):
    """Get the configuration of available NMT models."""
    if not translation_service:
        raise HTTPException(status_code=503, detail="Translation service not initialized")

    supported_langs = translation_service.get_supported_languages()
    if not supported_langs:
        raise HTTPException(status_code=503, detail="No languages available")
    # TartuNLP's public /translation/v2 includes same-language pairs (e.g. eng-eng).
    # Keep ordering stable for clients by sorting.
    sorted_langs = sorted(supported_langs)
    language_pairs = [f"{src}-{tgt}" for src in sorted_langs for tgt in sorted_langs]

    # Return a minimal but compatible configuration describing available domains
    domains = [
        Domain(
            name="General",
            code="general",
            # language pairs are hyphen-separated 3-letter-ish codes (we use our 3-letter style)
            languages=language_pairs,
        )
    ]
    return Config(domains=domains)

@app.post("/translation", response_model=Response, tags=["translation"])
async def translate(request: Request, x_api_key: Optional[str] = None, application: Optional[str] = None):
    """
    Translate text between Sami languages, Finnish, and Norwegian.
    Compatible with TartuNLP API format
    """
    if not translation_service:
        raise HTTPException(status_code=503, detail="Translation service not initialized")
    
    # Validate language codes
    valid_langs = set(translation_service.get_supported_languages())
    if not valid_langs:
        raise HTTPException(status_code=503, detail="No languages available")
    if request.src not in valid_langs or request.tgt not in valid_langs:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid language codes. Supported values: "
                + ", ".join(sorted(valid_langs))
                + f". Got src={request.src}, tgt={request.tgt}"
            )
        )

    # Helper to translate a single string
    def _translate_str(s: str) -> str:
        if request.src == request.tgt:
            return s
        return translation_service.translate(text=s, src_lang=request.src, tgt_lang=request.tgt)

    try:
        # Support both single string and list of sentences
        if isinstance(request.text, list):
            translated = [ _translate_str(s) for s in request.text ]
        else:
            translated = _translate_str(request.text)

        # Return in a shape compatible with TartuNLP Response (result: string or array)
        return Response(result=translated)
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
