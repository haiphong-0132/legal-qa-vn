import os
import json
import argparse
from typing import List, Optional

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = str(os.getenv("HF_TOKEN", ""))

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "AITeamVN/Vietnamese_Embedding")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "AITeamVN/Vietnamese_Reranker")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-8B")

API_EMBED_DIM = int(os.getenv("API_EMBED_DIM", "1024"))

# Tự ưu tiên GPU NVIDIA nếu có
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

embed_model: Optional[SentenceTransformer] = None
reranker_model: Optional[CrossEncoder] = None
llm_tokenizer: Optional[AutoTokenizer] = None
llm_model: Optional[AutoModelForCausalLM] = None


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7


class RerankRequest(BaseModel):
    query: str
    documents: List[str] = Field(..., min_length=1)
    top_k: int = 5


app = FastAPI(
    title="Local Multi-Model API",
    description="Embedding + Generation + Rerank APIs chạy local bằng GPU/CPU",
    version="1.0.0",
)


def ensure_embed_dim(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    current_dim = arr.shape[1]
    if current_dim == API_EMBED_DIM:
        return arr
    if current_dim > API_EMBED_DIM:
        return arr[:, :API_EMBED_DIM]

    pad_width = API_EMBED_DIM - current_dim
    return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def get_llm_dtype():
    if DEVICE == "cuda":
        # Đa số GPU chạy float16 ổn hơn và tiết kiệm VRAM
        return torch.float16
    return torch.float32


def load_models() -> None:
    global embed_model, reranker_model, llm_tokenizer, llm_model

    if embed_model is None:
        print(f"[LOAD] Embedding model: {EMBED_MODEL_NAME}")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
        print("[OK] Embedding model loaded")

    if reranker_model is None:
        print(f"[LOAD] Reranker model: {RERANK_MODEL_NAME}")
        reranker_model = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
        print("[OK] Reranker model loaded")

    if llm_tokenizer is None or llm_model is None:
        print(f"[LOAD] LLM model: {LLM_MODEL_NAME}")
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=get_llm_dtype(),
            low_cpu_mem_usage=True,
        )

        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        if llm_model.config.pad_token_id is None:
            llm_model.config.pad_token_id = llm_tokenizer.eos_token_id

        llm_model.to(DEVICE)
        llm_model.eval()
        print("[OK] LLM model loaded")


@app.on_event("startup")
def startup_event() -> None:
    print("[START] Server starting...")
    print(f"[INFO] HOST={HOST} PORT={PORT} DEVICE={DEVICE}")

    if DEVICE == "cuda":
        print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
        print(f"[INFO] CUDA device name: {torch.cuda.get_device_name(0)}")

    load_models()
    print("[READY] All models loaded successfully")


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "FastAPI server is running",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "endpoints": ["/embed", "/generate", "/rerank", "/docs"],
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        assert embed_model is not None

        embeddings = embed_model.encode(
            req.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = ensure_embed_dim(np.asarray(embeddings, dtype=np.float32))
        return {
            "embeddings": embeddings.tolist(),
            "dimension": API_EMBED_DIM,
            "device": DEVICE,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}") from exc


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        assert llm_tokenizer is not None and llm_model is not None

        max_new_tokens = max(1, min(req.max_length, 512))

        inputs = llm_tokenizer(
            req.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": llm_tokenizer.eos_token_id,
                "eos_token_id": llm_tokenizer.eos_token_id,
            }
            
            # Only add sampling parameters if model supports them
            if req.temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = max(req.temperature, 1e-5)
                gen_kwargs["top_p"] = 0.95
            
            try:
                output_ids = llm_model.generate(**inputs, **gen_kwargs)
            except TypeError:
                # Fallback: model doesn't support these parameters
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": llm_tokenizer.eos_token_id,
                    "eos_token_id": llm_tokenizer.eos_token_id,
                }
                output_ids = llm_model.generate(**inputs, **gen_kwargs)

        full_text = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if full_text.startswith(req.prompt):
            answer = full_text[len(req.prompt):].strip()
        else:
            answer = full_text.strip()

        return {
            "prompt": req.prompt,
            "answer": answer,
            "device": DEVICE,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}") from exc


@app.post("/rerank")
def rerank(req: RerankRequest):
    try:
        assert reranker_model is not None

        pairs = [[req.query, doc] for doc in req.documents]
        scores = reranker_model.predict(pairs)

        ranked = sorted(
            [
                {"document": doc, "score": float(score)}
                for doc, score in zip(req.documents, scores)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        top_k = max(1, min(req.top_k, len(ranked)))
        results = []
        for i, item in enumerate(ranked[:top_k], start=1):
            results.append(
                {
                    "rank": i,
                    "document": item["document"],
                    "score": item["score"],
                }
            )

        return {
            "query": req.query,
            "results": results,
            "device": DEVICE,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rerank error: {exc}") from exc


def pretty_print(title: str, data) -> None:
    print(f"\n--- {title} ---")
    try:
        print(json.dumps(data, ensure_ascii=False, indent=2)[:3000])
    except Exception:
        print(data)


def run_self_test(base_url: str) -> None:
    print(f"[TEST] Testing APIs via: {base_url}")

    embed_payload = {
        "texts": [
            "Xin chào, đây là câu tiếng Việt.",
            "FastAPI chạy local bằng GPU.",
        ]
    }
    r1 = requests.post(f"{base_url}/embed", json=embed_payload, timeout=120)
    embed_result = r1.json()
    pretty_print(
        "TEST /embed",
        {
            "status_code": r1.status_code,
            "dimension": embed_result.get("dimension"),
            "num_embeddings": len(embed_result.get("embeddings", [])),
            "device": embed_result.get("device"),
        },
    )

    generate_payload = {
        "prompt": "Question: FastAPI là gì?\nAnswer:",
        "max_length": 80,
        "temperature": 0.7,
    }
    r2 = requests.post(f"{base_url}/generate", json=generate_payload, timeout=180)
    pretty_print("TEST /generate", {"status_code": r2.status_code, "response": r2.json()})

    rerank_payload = {
        "query": "framework để build API Python nhanh",
        "documents": [
            "FastAPI là framework hiện đại để xây dựng API với Python.",
            "NumPy dùng cho tính toán ma trận.",
            "Localhost là địa chỉ loopback trên máy tính.",
            "Transformers hỗ trợ mô hình ngôn ngữ.",
        ],
        "top_k": 3,
    }
    r3 = requests.post(f"{base_url}/rerank", json=rerank_payload, timeout=120)
    pretty_print("TEST /rerank", {"status_code": r3.status_code, "response": r3.json()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local NLP FastAPI service")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    parser.add_argument("--self-test", action="store_true", help="Run HTTP tests after server starts")
    args = parser.parse_args()

    if args.self_test:
        import threading
        import time

        def run_server() -> None:
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        base_url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"

        start = time.time()
        while time.time() - start < 180:
            try:
                r = requests.get(f"{base_url}/", timeout=3)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("FastAPI server did not start in time.")

        run_self_test(base_url)
    else:
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)