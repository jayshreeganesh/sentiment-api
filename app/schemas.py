from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The text to analyze for sentiment or emotion.",
        examples=["I absolutely love using this new framework, it saved me hours of debugging!"]
    )

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="A list of texts (up to 100) to analyze in batch.",
        examples=[
            [
                "The product arrived on time and exceeded expectations!",
                "Terrible customer service, nobody answered my email.",
                "The package was delivered yesterday."
            ]
        ]
    )

class ScoreItem(BaseModel):
    label: str
    score: float

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: List[ScoreItem]
    inference_time_ms: float

class EmotionResult(BaseModel):
    text: str
    dominant_emotion: str
    confidence: float
    emotions: Dict[str, float]
    inference_time_ms: float

class BatchSentimentResponse(BaseModel):
    total_processed: int
    total_inference_time_ms: float
    average_latency_ms: float
    device: str
    results: List[SentimentResult]

class HealthResponse(BaseModel):
    status: str
    model_name: str
    device: str
    cuda_available: bool
    version: str
