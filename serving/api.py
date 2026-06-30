"""
FastAPI REST API for MedRAG — medical Q&A with fine-tuned Llama 3.3 70B + RAG.

Endpoints:
  GET  /health          — liveness check
  POST /predict         — answer without RAG
  POST /predict_rag     — answer with RAG (top-k=3 textbook chunks)

Run locally (requires GPU):
  uvicorn serving.api:app --host 0.0.0.0 --port 8000

Environment variables:
  BASE_MODEL      HuggingFace model ID (default: meta-llama/Llama-3.3-70B-Instruct)
  ADAPTER_PATH    Path to LoRA adapters directory
  VECTORSTORE     Path to FAISS vectorstore directory
  HF_TOKEN        HuggingFace token for gated model access
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.inference import MedRAGInferenceEngine

engine: MedRAGInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = MedRAGInferenceEngine(
        base_model=os.getenv("BASE_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        adapter_path=os.getenv("ADAPTER_PATH"),
        vectorstore_path=os.getenv("VECTORSTORE"),
        top_k=int(os.getenv("TOP_K", "3")),
    )
    yield
    engine = None


app = FastAPI(
    title="MedRAG API",
    description="Medical Q&A with Llama 3.3 70B + QLoRA fine-tuning + RAG over medical textbooks.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    question: str = Field(..., description="Full question text including options (A/B/C/D)")
    max_new_tokens: int = Field(256, ge=64, le=512)


class PredictResponse(BaseModel):
    answer: str = Field(..., description="Predicted answer letter (A/B/C/D)")
    explanation: str = Field(..., description="Model's explanation")
    rag_used: bool
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    result = engine.generate(req.question, use_rag=False, max_new_tokens=req.max_new_tokens)
    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(**result, latency_ms=round(latency_ms, 1))


@app.post("/predict_rag", response_model=PredictResponse)
def predict_rag(req: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if engine.rag is None:
        raise HTTPException(status_code=400, detail="RAG not configured (VECTORSTORE not set)")
    t0 = time.perf_counter()
    result = engine.generate(req.question, use_rag=True, max_new_tokens=req.max_new_tokens)
    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(**result, latency_ms=round(latency_ms, 1))
