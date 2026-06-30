"""
Ollama bge-m3 embedding client.

Why bge-m3:
  - Runs fully on CPU via Ollama (no GPU required on a 32 GB host)
  - 1024-dimensional output — good quality for English technical text
  - Output vectors are already L2-normalised → use COSINE metric in Milvus
  - No per-call API cost, enabling unlimited re-embeddings during development
"""

import os
import hashlib
import math
import re
import httpx
from typing import List

from knowops.config import SETTINGS

_DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = "bge-m3"
_OFFLINE_DIM = 1024

# Minimal stop-list so shared topical words dominate offline similarity instead
# of ubiquitous filler words. Not needed for real bge-m3.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "this", "that", "with", "as", "by", "at", "from", "it", "we", "you",
    "your", "our", "i", "should", "use", "using", "what", "which", "how",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def embed_offline(text: str, dim: int = _OFFLINE_DIM) -> List[float]:
    """Deterministic lexical embedding for offline mode (no Ollama required).

    Feature-hashing: each non-stopword token is hashed (stable MD5, unlike
    Python's salted ``hash``) into one of ``dim`` buckets and counted. The
    L2-normalised vector makes cosine similarity ≈ shared-vocabulary overlap,
    which is enough to surface topically related documents in the demo and
    tests. Reproducible across processes — same text always yields the same
    vector.
    """
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS or len(tok) == 1:
            continue
        idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def embed(text: str, base_url: str = _DEFAULT_BASE_URL, offline: bool | None = None) -> List[float]:
    """Embed a single text string.

    In offline mode returns a deterministic lexical vector; otherwise calls
    bge-m3 via Ollama and returns its 1024-dim L2-normalised vector.
    Raises httpx.HTTPStatusError on API failure (live mode only).
    """
    if offline is None:
        offline = SETTINGS.offline
    if offline:
        return embed_offline(text)
    response = httpx.post(
        f"{base_url}/api/embeddings",
        json={"model": _MODEL, "prompt": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(
    texts: List[str], base_url: str = _DEFAULT_BASE_URL, offline: bool | None = None
) -> List[List[float]]:
    """Embed a list of strings sequentially.

    Ollama (CPU mode) processes one request at a time; we call embed() in a
    simple loop rather than spawning concurrent requests to avoid OOM.
    Batch size is controlled at the call site — keep it at ≤ 8 for bge-m3.
    """
    return [embed(text, base_url=base_url, offline=offline) for text in texts]


def check_ollama_health(base_url: str = _DEFAULT_BASE_URL) -> bool:
    """Return True if the Ollama server is reachable and bge-m3 is loaded."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(_MODEL in m for m in models)
    except Exception:
        return False
