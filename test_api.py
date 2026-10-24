import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "model_name" in data

def test_positive_sentiment():
    payload = {"text": "I am absolutely thrilled with the incredible performance of this application!"}
    response = client.post("/api/v1/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "POSITIVE"
    assert data["confidence"] > 0.8
    assert "inference_time_ms" in data

def test_negative_sentiment():
    payload = {"text": "This was the worst customer experience I have ever had. Truly awful."}
    response = client.post("/api/v1/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "NEGATIVE"
    assert data["confidence"] > 0.8

def test_emotion_classification():
    payload = {"text": "I am terrifyingly scared that the server might explode in flames!"}
    response = client.post("/api/v1/emotion", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dominant_emotion"] in ["fear", "surprise"]
    assert "emotions" in data
    assert len(data["emotions"]) >= 6

def test_batch_sentiment():
    payload = {
        "texts": [
            "We had a fantastic time at the team celebration!",
            "Broken beyond repair, complete waste of money."
        ]
    }
    response = client.post("/api/v1/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["sentiment"] == "POSITIVE"
    assert data["results"][1]["sentiment"] == "NEGATIVE"
