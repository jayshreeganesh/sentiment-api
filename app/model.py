import time
from typing import List, Dict, Any, Tuple
import torch
from transformers import pipeline

class SentimentClassifier:
    """
    Singleton classifier for real-time sentiment and emotion inference using PyTorch and DistilBERT.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentClassifier, cls).__new__(cls)
            cls._instance._initialize_models()
        return cls._instance

    def _initialize_models(self):
        # Determine execution hardware (CUDA GPU vs. CPU)
        if torch.cuda.is_available():
            self.device_id = 0
            self.device_name = f"cuda:0 ({torch.cuda.get_device_name(0)})"
        else:
            self.device_id = -1
            self.device_name = "cpu"

        print(f"[Model Loader] Initializing sentiment models on device: {self.device_name}")

        self.sentiment_model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        self.emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"

        # Load Sentiment Pipeline
        self.sentiment_pipe = pipeline(
            "text-classification",
            model=self.sentiment_model_name,
            top_k=None,  # Return all class probabilities
            device=self.device_id,
            truncation=True,
            max_length=512
        )

        # Load Emotion Pipeline
        self.emotion_pipe = pipeline(
            "text-classification",
            model=self.emotion_model_name,
            top_k=None,
            device=self.device_id,
            truncation=True,
            max_length=512
        )

        print("[Model Loader] Models loaded successfully into memory.")

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict POSITIVE, NEGATIVE, or NEUTRAL with confidence breakdown.
        """
        start_time = time.perf_counter()
        raw_output = self.sentiment_pipe(text)[0]
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Standardize score breakdown
        scores = [{"label": item["label"].upper(), "score": round(item["score"], 4)} for item in raw_output]
        scores_by_label = {s["label"]: s["score"] for s in scores}

        pos_score = scores_by_label.get("POSITIVE", 0.0)
        neg_score = scores_by_label.get("NEGATIVE", 0.0)

        # Determine sentiment with threshold for Neutral
        if abs(pos_score - neg_score) < 0.2:
            sentiment = "NEUTRAL"
            confidence = round(1.0 - abs(pos_score - neg_score), 4)
        elif pos_score > neg_score:
            sentiment = "POSITIVE"
            confidence = pos_score
        else:
            sentiment = "NEGATIVE"
            confidence = neg_score

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": scores,
            "inference_time_ms": inference_time_ms
        }

    def predict_emotion(self, text: str) -> Dict[str, Any]:
        """
        Predict multi-class emotion: joy, sadness, anger, fear, love, surprise.
        """
        start_time = time.perf_counter()
        raw_output = self.emotion_pipe(text)[0]
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Map to dictionary
        emotions = {item["label"].lower(): round(item["score"], 4) for item in raw_output}
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]

        return {
            "text": text,
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "emotions": emotions,
            "inference_time_ms": inference_time_ms
        }

    def predict_batch_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Vectorized high-throughput batch inference.
        """
        start_time = time.perf_counter()
        raw_outputs = self.sentiment_pipe(texts, batch_size=32)
        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        avg_latency_ms = round(total_time_ms / len(texts), 2) if texts else 0.0

        results = []
        for text, raw_output in zip(texts, raw_outputs):
            scores = [{"label": item["label"].upper(), "score": round(item["score"], 4)} for item in raw_output]
            scores_by_label = {s["label"]: s["score"] for s in scores}
            pos_score = scores_by_label.get("POSITIVE", 0.0)
            neg_score = scores_by_label.get("NEGATIVE", 0.0)

            if abs(pos_score - neg_score) < 0.2:
                sentiment = "NEUTRAL"
                confidence = round(1.0 - abs(pos_score - neg_score), 4)
            elif pos_score > neg_score:
                sentiment = "POSITIVE"
                confidence = pos_score
            else:
                sentiment = "NEGATIVE"
                confidence = neg_score

            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "scores": scores,
                "inference_time_ms": avg_latency_ms
            })

        return {
            "total_processed": len(texts),
            "total_inference_time_ms": total_time_ms,
            "average_latency_ms": avg_latency_ms,
            "device": self.device_name,
            "results": results
        }

# Global accessor
def get_classifier() -> SentimentClassifier:
    return SentimentClassifier()
