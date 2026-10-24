import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.schemas import (
    SentimentRequest,
    SentimentResult,
    EmotionResult,
    BatchSentimentRequest,
    BatchSentimentResponse,
    HealthResponse
)
from app.model import get_classifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload models into memory (and warm up the pipeline)
    print("[FastAPI] Warming up PyTorch NLP pipelines...")
    classifier = get_classifier()
    # Warm-up inference
    classifier.predict_sentiment("Warmup inference run")
    classifier.predict_emotion("Warmup inference run")
    print("[FastAPI] Model warmup complete. Server ready to receive traffic.")
    yield
    print("[FastAPI] Shutting down application.")

app = FastAPI(
    title="Real-Time Sentiment & Emotion Analysis API",
    description=(
        "High-performance REST API powered by FastAPI, PyTorch, and Hugging Face DistilBERT. "
        "Provides single and batch sentiment analysis, multi-class emotion classification, and confidence scoring."
    ),
    version=__version__,
    lifespan=lifespan
)

# Enable CORS for cross-origin frontend consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", summary="Interactive Web Playground", include_in_schema=False)
async def serve_playground():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Sentiment API is running. Visit /docs for OpenAPI documentation."}

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check & Hardware Diagnostics"
)
async def health_check():
    classifier = get_classifier()
    return HealthResponse(
        status="healthy",
        model_name=classifier.sentiment_model_name,
        device=classifier.device_name,
        cuda_available=torch.cuda.is_available(),
        version=__version__
    )

@app.post(
    "/api/v1/sentiment",
    response_model=SentimentResult,
    tags=["Inference"],
    summary="Analyze Single Sentiment (Positive / Negative / Neutral)"
)
async def analyze_sentiment(payload: SentimentRequest):
    try:
        classifier = get_classifier()
        result = classifier.predict_sentiment(payload.text)
        return SentimentResult(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@app.post(
    "/api/v1/emotion",
    response_model=EmotionResult,
    tags=["Inference"],
    summary="Multi-Class Emotion Classification (Joy, Sadness, Anger, Fear, Love, Surprise)"
)
async def analyze_emotion(payload: SentimentRequest):
    try:
        classifier = get_classifier()
        result = classifier.predict_emotion(payload.text)
        return EmotionResult(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@app.post(
    "/api/v1/batch",
    response_model=BatchSentimentResponse,
    tags=["Inference"],
    summary="Vectorized High-Throughput Batch Sentiment Inference"
)
async def analyze_batch(payload: BatchSentimentRequest):
    try:
        classifier = get_classifier()
        result = classifier.predict_batch_sentiment(payload.texts)
        return BatchSentimentResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )
