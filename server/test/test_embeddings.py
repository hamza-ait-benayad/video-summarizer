import pytest
from unittest.mock import MagicMock, patch
from app.services.rag.embeddings import (
    DeterministicHashEmbeddingProvider,
    GeminiEmbeddingProvider,
    get_embedding_provider,
)


def test_deterministic_embedding_provider():
    provider = DeterministicHashEmbeddingProvider(dimension=768)
    assert provider.dimension == 768
    assert provider.model_name == "deterministic-hash"

    texts = [
        "What is machine learning?",
        "Deep learning uses artificial neural networks.",
        "Computer vision allows machines to interpret visual data.",
    ]
    vectors = provider.embed_texts(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 768
        # Ensure vectors are normalized
        norm = sum(x * x for x in vec) ** 0.5
        assert pytest.approx(norm, rel=1e-3) == 1.0

    # Query embedding
    query_vec = provider.embed_query("What is machine learning?")
    assert len(query_vec) == 768
    # Exact match for same text
    assert query_vec == vectors[0]


def test_gemini_embedding_batch_and_dimension_mock():
    # Mock google.genai client
    mock_client = MagicMock()
    mock_embedding_1 = MagicMock()
    mock_embedding_1.values = [0.1] * 768
    mock_embedding_2 = MagicMock()
    mock_embedding_2.values = [0.2] * 768

    mock_resp = MagicMock()
    mock_resp.embeddings = [mock_embedding_1, mock_embedding_2]
    mock_client.models.embed_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        provider = GeminiEmbeddingProvider(
            model_name="text-embedding-004",
            api_key="mock-key",
            dimension=768,
        )

        results = provider.embed_texts(["Chunk 1 text", "Chunk 2 text"])
        assert len(results) == 2
        assert len(results[0]) == 768
        assert len(results[1]) == 768
        assert results[0][0] == 0.1
        assert results[1][0] == 0.2


def test_gemini_embedding_retry_on_rate_limit():
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.5] * 768
    mock_resp = MagicMock()
    mock_resp.embeddings = [mock_embedding]

    # First attempt raises 429 rate limit, second attempt succeeds
    mock_client.models.embed_content.side_effect = [
        RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit exceeded"),
        mock_resp,
    ]

    with patch("google.genai.Client", return_value=mock_client):
        with patch("time.sleep", return_value=None):
            provider = GeminiEmbeddingProvider(
                model_name="text-embedding-004",
                api_key="mock-key",
                dimension=768,
            )
            results = provider.embed_texts(["Test text"])
            assert len(results) == 1
            assert len(results[0]) == 768


def test_gemini_embedding_provider_fallback_after_retries():
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = RuntimeError("503 UNAVAILABLE: Service unavailable")

    with patch("google.genai.Client", return_value=mock_client):
        with patch("time.sleep", return_value=None):
            provider = GeminiEmbeddingProvider(
                model_name="gemini-embedding-001",
                api_key="mock-key",
                dimension=768,
            )
            # When all external models fail or are unreachable, it falls back to deterministic provider
            results = provider.embed_texts(["Test text"])
            assert len(results) == 1
            assert len(results[0]) == 768


def test_missing_api_key_raises_error():
    with patch.dict("os.environ", {}, clear=True):
        with patch("app.core.config.settings.GEMINI_API_KEY", None):
            with patch("app.core.config.settings.GOOGLE_API_KEY", None):
                with pytest.raises(ValueError) as exc:
                    GeminiEmbeddingProvider(api_key=None)
                assert "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is configured" in str(exc.value)


def test_factory_function():
    det_provider = get_embedding_provider("deterministic")
    assert isinstance(det_provider, DeterministicHashEmbeddingProvider)

    with pytest.raises(ValueError):
        get_embedding_provider("unsupported-provider-xyz")
