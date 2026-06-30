"""
Unit tests for knowops/embedder.py.

All httpx calls are mocked — Ollama does NOT need to be running locally.
(Ollama runs on the remote server; tests must be runnable offline.)

Run:
    source .venv/bin/activate
    pip install pytest
    pytest tests/test_embedder.py -v
"""

import pytest
from unittest.mock import MagicMock, call, patch

from knowops.embedder import embed, embed_batch, check_ollama_health

# A realistic 1024-dim bge-m3 vector (all zeros is fine for testing shape/type)
FAKE_VECTOR = [0.0] * 1024
FAKE_BASE_URL = "http://fake-server:11434"


@pytest.fixture(autouse=True)
def _pin_live_mode(monkeypatch):
    """Force live (non-offline) routing so these mocked-httpx tests are
    independent of the KNOWOPS_OFFLINE environment variable."""
    from knowops.config import SETTINGS
    monkeypatch.setattr(SETTINGS, "offline", False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Build a fake httpx.Response that returns `json_data` from .json()."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    if status_code >= 400:
        import httpx
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock,
        )
    else:
        mock.raise_for_status.return_value = None
    return mock


# ── embed() ───────────────────────────────────────────────────────────────────

class TestEmbed:

    @patch("knowops.embedder.httpx.post")
    def test_returns_vector_on_success(self, mock_post):
        """embed() returns the embedding list from the Ollama JSON response."""
        mock_post.return_value = make_mock_response({"embedding": FAKE_VECTOR})

        result = embed("hello world", base_url=FAKE_BASE_URL)

        assert result == FAKE_VECTOR
        assert len(result) == 1024

    @patch("knowops.embedder.httpx.post")
    def test_posts_to_correct_endpoint(self, mock_post):
        """embed() calls the Ollama /api/embeddings endpoint."""
        mock_post.return_value = make_mock_response({"embedding": FAKE_VECTOR})

        embed("test query", base_url=FAKE_BASE_URL)

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{FAKE_BASE_URL}/api/embeddings"

    @patch("knowops.embedder.httpx.post")
    def test_sends_correct_model_and_prompt(self, mock_post):
        """embed() sends model='bge-m3' and the correct prompt in the JSON body."""
        mock_post.return_value = make_mock_response({"embedding": FAKE_VECTOR})

        embed("my query text", base_url=FAKE_BASE_URL)

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "bge-m3"
        assert kwargs["json"]["prompt"] == "my query text"

    @patch("knowops.embedder.httpx.post")
    def test_uses_60s_timeout(self, mock_post):
        """embed() uses a 60-second timeout — bge-m3 on CPU can be slow."""
        mock_post.return_value = make_mock_response({"embedding": FAKE_VECTOR})

        embed("text", base_url=FAKE_BASE_URL)

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 60.0

    @patch("knowops.embedder.httpx.post")
    def test_raises_on_http_error(self, mock_post):
        """embed() propagates HTTPStatusError on non-2xx responses."""
        import httpx
        mock_post.return_value = make_mock_response({}, status_code=500)

        with pytest.raises(httpx.HTTPStatusError):
            embed("text", base_url=FAKE_BASE_URL)

    @patch("knowops.embedder.httpx.post")
    def test_raises_on_404(self, mock_post):
        """embed() raises on 404 — e.g. bge-m3 not loaded in Ollama."""
        import httpx
        mock_post.return_value = make_mock_response({}, status_code=404)

        with pytest.raises(httpx.HTTPStatusError):
            embed("text", base_url=FAKE_BASE_URL)

    @patch("knowops.embedder.httpx.post")
    def test_empty_string_input(self, mock_post):
        """embed() accepts an empty string without error (Ollama handles it)."""
        mock_post.return_value = make_mock_response({"embedding": FAKE_VECTOR})

        result = embed("", base_url=FAKE_BASE_URL)

        assert result == FAKE_VECTOR

    @patch("knowops.embedder.httpx.post")
    def test_long_text_input(self, mock_post):
        """embed() passes long text through unchanged — truncation is Ollama's job."""
        mock_post.return_value = make_mock_response({"embedding": FAKE_VECTOR})
        long_text = "word " * 1000  # ~5000 chars

        result = embed(long_text, base_url=FAKE_BASE_URL)

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["prompt"] == long_text
        assert result == FAKE_VECTOR


# ── embed_batch() ────────────────────────────────────────────────────────────

class TestEmbedBatch:

    @patch("knowops.embedder.embed")
    def test_calls_embed_once_per_text(self, mock_embed):
        """embed_batch() calls embed() exactly N times for N inputs."""
        mock_embed.return_value = FAKE_VECTOR
        texts = ["a", "b", "c"]

        result = embed_batch(texts, base_url=FAKE_BASE_URL)

        assert mock_embed.call_count == len(texts)
        assert len(result) == len(texts)

    @patch("knowops.embedder.embed")
    def test_preserves_order(self, mock_embed):
        """embed_batch() returns vectors in the same order as input texts."""
        vectors = [[float(i)] * 1024 for i in range(3)]
        mock_embed.side_effect = vectors
        texts = ["first", "second", "third"]

        result = embed_batch(texts, base_url=FAKE_BASE_URL)

        assert result == vectors

    @patch("knowops.embedder.embed")
    def test_passes_base_url_to_each_call(self, mock_embed):
        """embed_batch() forwards the base_url to every embed() call."""
        mock_embed.return_value = FAKE_VECTOR
        texts = ["x", "y"]

        embed_batch(texts, base_url=FAKE_BASE_URL)

        for c in mock_embed.call_args_list:
            assert c.kwargs.get("base_url") == FAKE_BASE_URL or c.args[1] == FAKE_BASE_URL

    @patch("knowops.embedder.embed")
    def test_empty_list_returns_empty(self, mock_embed):
        """embed_batch([]) returns [] and never calls embed()."""
        result = embed_batch([], base_url=FAKE_BASE_URL)

        assert result == []
        mock_embed.assert_not_called()

    @patch("knowops.embedder.embed")
    def test_single_item_list(self, mock_embed):
        """embed_batch() works correctly with a single-element list."""
        mock_embed.return_value = FAKE_VECTOR

        result = embed_batch(["only one"], base_url=FAKE_BASE_URL)

        assert result == [FAKE_VECTOR]
        mock_embed.assert_called_once()

    def test_propagates_embed_error(self):
        """embed_batch() propagates any exception raised by embed()."""
        import httpx
        with patch("knowops.embedder.embed") as mock_embed:
            mock_embed.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
            with pytest.raises(httpx.HTTPStatusError):
                embed_batch(["text"], base_url=FAKE_BASE_URL)


# ── check_ollama_health() ─────────────────────────────────────────────────────

class TestCheckOllamaHealth:

    @patch("knowops.embedder.httpx.get")
    def test_returns_true_when_bge_m3_loaded(self, mock_get):
        """check_ollama_health() returns True when bge-m3 appears in /api/tags."""
        mock_get.return_value = make_mock_response({
            "models": [{"name": "bge-m3:latest"}, {"name": "llama3:8b"}]
        })

        assert check_ollama_health(base_url=FAKE_BASE_URL) is True

    @patch("knowops.embedder.httpx.get")
    def test_returns_false_when_model_missing(self, mock_get):
        """check_ollama_health() returns False when bge-m3 is not in the model list."""
        mock_get.return_value = make_mock_response({
            "models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]
        })

        assert check_ollama_health(base_url=FAKE_BASE_URL) is False

    @patch("knowops.embedder.httpx.get")
    def test_returns_false_when_model_list_empty(self, mock_get):
        """check_ollama_health() returns False when no models are loaded."""
        mock_get.return_value = make_mock_response({"models": []})

        assert check_ollama_health(base_url=FAKE_BASE_URL) is False

    @patch("knowops.embedder.httpx.get")
    def test_returns_false_on_connection_error(self, mock_get):
        """check_ollama_health() returns False (not raises) on network error."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        assert check_ollama_health(base_url=FAKE_BASE_URL) is False

    @patch("knowops.embedder.httpx.get")
    def test_returns_false_on_timeout(self, mock_get):
        """check_ollama_health() returns False (not raises) on timeout."""
        import httpx
        mock_get.side_effect = httpx.TimeoutException("Timed out")

        assert check_ollama_health(base_url=FAKE_BASE_URL) is False

    @patch("knowops.embedder.httpx.get")
    def test_returns_false_on_http_error(self, mock_get):
        """check_ollama_health() returns False (not raises) on 5xx responses."""
        mock_get.return_value = make_mock_response({}, status_code=503)

        assert check_ollama_health(base_url=FAKE_BASE_URL) is False

    @patch("knowops.embedder.httpx.get")
    def test_queries_correct_endpoint(self, mock_get):
        """check_ollama_health() calls /api/tags on the given base_url."""
        mock_get.return_value = make_mock_response({"models": [{"name": "bge-m3"}]})

        check_ollama_health(base_url=FAKE_BASE_URL)

        mock_get.assert_called_once()
        args, _ = mock_get.call_args
        assert args[0] == f"{FAKE_BASE_URL}/api/tags"

    @patch("knowops.embedder.httpx.get")
    def test_uses_short_timeout(self, mock_get):
        """check_ollama_health() uses a short timeout (5s) — it's a liveness check."""
        mock_get.return_value = make_mock_response({"models": [{"name": "bge-m3"}]})

        check_ollama_health(base_url=FAKE_BASE_URL)

        _, kwargs = mock_get.call_args
        assert kwargs.get("timeout", None) == 5.0

    @patch("knowops.embedder.httpx.get")
    def test_partial_model_name_match(self, mock_get):
        """check_ollama_health() matches 'bge-m3' anywhere in the model name string."""
        # Ollama may return names like 'bge-m3:latest' or 'bge-m3:567mb'
        mock_get.return_value = make_mock_response({
            "models": [{"name": "bge-m3:567mb"}]
        })

        assert check_ollama_health(base_url=FAKE_BASE_URL) is True
