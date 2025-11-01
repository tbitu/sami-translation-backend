"""
FastAPI server for TartuNLP smugri3_14 translation
Runs Northern Sami ↔ Norwegian translation models on NVIDIA GPU
Uses TartuNLP's Finno-Ugric NMT model with Fairseq
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

app = FastAPI(title="Sami Translation API")

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
    logger.info("Translation service ready!")

class TranslationRequest(BaseModel):
    text: str
    src: str  # "sme" (Northern Sami) or "nor" (Norwegian)
    tgt: str  # "sme" or "nor"

class TranslationResponse(BaseModel):
    result: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Sami Translation API",
        "cuda_available": torch.cuda.is_available(),
        "device": str(translation_service.device) if translation_service else "not initialized"
    }

@app.post("/translation/v2", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text between Northern Sami and Norwegian
    Compatible with TartuNLP API format
    """
    if not translation_service:
        raise HTTPException(status_code=503, detail="Translation service not initialized")
    
    # Validate language codes
    valid_langs = {"sme", "nor"}
    if request.src not in valid_langs or request.tgt not in valid_langs:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid language codes. Must be 'sme' or 'nor'. Got src={request.src}, tgt={request.tgt}"
        )
    
    if request.src == request.tgt:
        # No translation needed
        return TranslationResponse(result=request.text)
    
    try:
        logger.info(f"Translating {len(request.text)} chars from {request.src} to {request.tgt}")
        result = translation_service.translate(
            text=request.text,
            src_lang=request.src,
            tgt_lang=request.tgt
        )
        logger.info(f"Translation complete: {result[:50]}...")
        return TranslationResponse(result=result)
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
