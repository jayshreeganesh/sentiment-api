# ⚡ Real-Time Sentiment & Emotion Analysis API

A high-throughput, production-ready REST API for sentiment and emotion classification powered by **FastAPI**, **PyTorch**, and **Hugging Face Transformers (DistilBERT)**.

Includes an interactive web dashboard, automated hardware acceleration (CUDA/CPU), Swagger OpenAPI documentation, and vectorized batch inference.

---

## 🌟 Key Features

- **Dual-Model Inference Pipeline:**
  - **Sentiment Classification:** Fast binary/neutral sentiment detection (`POSITIVE`, `NEGATIVE`, `NEUTRAL`) with confidence scores.
  - **Fine-Grained Emotion Detection:** Multi-class classification across 6 distinct human emotions (`Joy`, `Sadness`, `Anger`, `Fear`, `Surprise`, `Love`).
- **Hardware-Accelerated (PyTorch):** Automatically detects and utilizes NVIDIA CUDA GPUs if present, with optimized CPU tensor fallback.
- **High-Throughput Batch Processing:** Parallel vectorized inference processing up to 100 sentences per request with sub-second turnaround.
- **Built-in Interactive UI Playground:** Web dashboard served directly by FastAPI for immediate testing and real-time visualization.
- **Auto-Generated OpenAPI / Swagger:** Interactive API docs at `/docs` with request/response schemas.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Client or Web UI] --> B[FastAPI Router]
    B --> C[Pydantic Validation]
    C --> D[PyTorch DistilBERT Engine]
    D --> E[Tokenizer]
    E --> F[Tensor Inference - CUDA or CPU]
    F --> G[Softmax and Confidence Scoring]
    G --> H[JSON Output with Latency]
```

---

## 🛠️ Tech Stack

- **Framework:** FastAPI (Python 3.10+)
- **ASGI Server:** Uvicorn
- **Deep Learning Engine:** PyTorch (`torch`)
- **Model Architecture:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english` & `distilbert-base-uncased-emotion`)
- **Validation:** Pydantic v2
- **Testing:** Pytest & HTTPX TestClient
- **Frontend Playground:** HTML5 + Tailwind CSS

---

## 🚀 Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/jayshreeganesh/sentiment-api.git
cd sentiment-api

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive Web Playground:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Reference

### 1. Single Sentiment Analysis
`POST /api/v1/sentiment`

**Request:**
```json
{
  "text": "I absolutely love using this new framework, it saved me hours of debugging!"
}
```

**Response:**
```json
{
  "text": "I absolutely love using this new framework, it saved me hours of debugging!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998,
  "scores": [
    { "label": "NEGATIVE", "score": 0.0002 },
    { "label": "POSITIVE", "score": 0.9998 }
  ],
  "inference_time_ms": 18.42
}
```

---

### 2. Fine-Grained Emotion Detection
`POST /api/v1/emotion`

**Request:**
```json
{
  "text": "I am terrifyingly scared that the server might crash under heavy load!"
}
```

**Response:**
```json
{
  "text": "I am terrifyingly scared that the server might crash under heavy load!",
  "dominant_emotion": "fear",
  "confidence": 0.9641,
  "emotions": {
    "fear": 0.9641,
    "sadness": 0.0182,
    "anger": 0.0104,
    "surprise": 0.0042,
    "joy": 0.0021,
    "love": 0.0010
  },
  "inference_time_ms": 22.15
}
```

---

### 3. Vectorized Batch Analysis
`POST /api/v1/batch`

**Request:**
```json
{
  "texts": [
    "The delivery arrived on time and exceeded expectations!",
    "Terrible customer service, nobody answered my email.",
    "The package was delivered yesterday."
  ]
}
```

**Response:**
```json
{
  "total_processed": 3,
  "total_inference_time_ms": 38.6,
  "average_latency_ms": 12.87,
  "device": "cpu",
  "results": [ ... ]
}
```

---

## 🧪 Running Automated Tests

Run the full automated test suite:

```bash
pytest test_api.py -v
```

---

## 📂 Project Structure

```text
sentiment-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application, routes, and lifespan warmup
│   ├── model.py         # PyTorch & HuggingFace DistilBERT classifier singleton
│   └── schemas.py       # Pydantic v2 validation models
├── static/
│   └── index.html       # Interactive Tailwind UI dashboard
├── requirements.txt     # Python dependencies
├── test_api.py          # Pytest automated test suite
├── .gitignore
└── README.md
```

---

## 📄 License

MIT
