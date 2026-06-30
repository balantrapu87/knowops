"""
Ollama bge-m3 embedding client.

Why bge-m3:
  - Runs fully on CPU via Ollama (no GPU required on a 32 GB host)
  - 1024-dimensional output — good quality for English technical text
  - Output vectors are already L2-normalised → use COSINE metric in Milvus
  - No per-call API cost, enabling unlimited re-embeddings during development
"""

import os
import httpx
from typing import List

_DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = "bge-m3"


def embed(text: str, base_url: str = _DEFAULT_BASE_URL) -> List[float]:
    """Embed a single text string using bge-m3 via Ollama.

    Returns a 1024-dimensional float vector, L2-normalised.
    Raises httpx.HTTPStatusError on API failure.
    """
    response = httpx.post(
        f"{base_url}/api/embeddings",
        json={"model": _MODEL, "prompt": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: List[str], base_url: str = _DEFAULT_BASE_URL) -> List[List[float]]:
    """Embed a list of strings sequentially.

    Ollama (CPU mode) processes one request at a time; we call embed() in a
    simple loop rather than spawning concurrent requests to avoid OOM.
    Batch size is controlled at the call site — keep it at ≤ 8 for bge-m3.
    """
    return [embed(text, base_url=base_url) for text in texts]


def check_ollama_health(base_url: str = _DEFAULT_BASE_URL) -> bool:
    """Return True if the Ollama server is reachable and bge-m3 is loaded."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(_MODEL in m for m in models)
    except Exception:
        return False
