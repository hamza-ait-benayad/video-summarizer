import os
import time
import math
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the model identifier."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str], task_type: Optional[str] = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Embed a batch of text strings into vectors."""
        pass

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query text using RETRIEVAL_QUERY task type."""
        results = self.embed_texts([query], task_type="RETRIEVAL_QUERY")
        if not results:
            raise ValueError("Failed to generate embedding for query.")
        return results[0]


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider using Google GenAI SDK and Gemini Embedding models
    (gemini-embedding-001, gemini-embedding-2, gemini-embedding-2-preview).
    """

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, dimension: Optional[int] = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or settings.GEMINI_API_KEY
        if not self._api_key:
            raise ValueError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is configured.")

        from google import genai
        self.client = genai.Client(api_key=self._api_key)

        # Candidate models to try in priority order
        configured = self._model_name
        defaults = [
            "gemini-embedding-001",
            "gemini-embedding-2",
            "gemini-embedding-2-preview",
            "text-embedding-004",
        ]
        self.candidate_models = [configured] + [m for m in defaults if m != configured]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str], task_type: Optional[str] = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        if not texts:
            return []

        embeddings: List[List[float]] = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch, task_type=task_type)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _embed_single_with_model(
        self,
        model_name: str,
        text: str,
        task_type: Optional[str],
    ) -> List[float]:
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimension
        )

        try:
            response = self.client.models.embed_content(
                model=model_name,
                contents=text,
                config=config,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "task_type" in err_str or "invalid" in err_str or "unsupported" in err_str:
                config = types.EmbedContentConfig(output_dimensionality=self._dimension)
                response = self.client.models.embed_content(
                    model=model_name,
                    contents=text,
                    config=config,
                )
            else:
                raise

        if hasattr(response, "embeddings") and response.embeddings:
            values = list(response.embeddings[0].values)
        elif hasattr(response, "embedding") and response.embedding:
            values = list(response.embedding.values)
        else:
            raise ValueError(f"Unexpected response format from embedding API: {response}")

        if len(values) > self._dimension:
            values = values[:self._dimension]
        elif len(values) < self._dimension:
            values = values + [0.0] * (self._dimension - len(values))

        return values

    def _embed_batch_with_retry(self, batch: List[str], task_type: Optional[str] = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        from google.genai import types

        max_retries = 3
        last_error = None

        for model_name in self.candidate_models:
            for attempt in range(1, max_retries + 1):
                try:
                    # 1. Attempt batch call with Content parts wrapper
                    try:
                        contents = [types.Content(parts=[types.Part.from_text(text=s)]) for s in batch]
                        config = types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self._dimension
                        )
                        response = self.client.models.embed_content(
                            model=model_name,
                            contents=contents,
                            config=config,
                        )
                        if hasattr(response, "embeddings") and len(response.embeddings) == len(batch):
                            result = [list(e.values) for e in response.embeddings]
                            self._model_name = model_name
                            return result
                    except Exception as batch_err:
                        logger.debug(f"Batch embedding with {model_name} failed ({batch_err}), falling back to single texts.")

                    # 2. Individual embedding fallback (guarantees exact length match)
                    results: List[List[float]] = []
                    for text in batch:
                        vec = self._embed_single_with_model(model_name, text, task_type)
                        results.append(vec)

                    self._model_name = model_name
                    return results

                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    is_transient = (
                        "429" in err_str
                        or "resource_exhausted" in err_str
                        or "503" in err_str
                        or "502" in err_str
                        or "504" in err_str
                        or "unavailable" in err_str
                        or "unreachable" in err_str
                        or "10054" in err_str
                        or "connection" in err_str
                        or "connecterror" in err_str
                        or "timeout" in err_str
                        or "reset by peer" in err_str
                    )
                    is_not_found = "404" in err_str or "not_found" in err_str or "not supported" in err_str

                    if is_not_found:
                        logger.warning(f"Embedding model {model_name} unavailable (404/not found). Trying next candidate...")
                        break

                    if is_transient and attempt < max_retries:
                        wait_time = attempt * 2
                        logger.warning(f"Transient error with {model_name} ({e}). Retrying in {wait_time}s (Attempt {attempt}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"Embedding model {model_name} failed attempt {attempt}: {e}")
                        if attempt == max_retries:
                            break

        # Fallback to deterministic pseudo-embeddings if external models are unreachable
        logger.critical(
            f"All external Gemini embedding models failed or unreachable. Last error: {last_error}. "
            f"Falling back to DeterministicHashEmbeddingProvider to maintain availability."
        )
        fallback_provider = DeterministicHashEmbeddingProvider(dimension=self._dimension)
        return fallback_provider.embed_texts(batch, task_type=task_type)


class DeterministicHashEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic pseudo-embedding provider for testing, offline environments,
    and fallback validation without external network calls.
    Generates normalized unit vectors of the configured dimension using sha256 hashing.
    """

    def __init__(self, dimension: int = 768, model_name: str = "deterministic-hash"):
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str], task_type: Optional[str] = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            vec = self._vector_from_text(text)
            results.append(vec)
        return results

    def _vector_from_text(self, text: str) -> List[float]:
        # Hash text to generate a deterministic sequence of floats
        words = text.lower().split()
        vector = [0.0] * self._dimension

        # Simple semantic token projection
        for word in words:
            word_hash = hashlib.sha256(word.encode("utf-8")).digest()
            for idx in range(self._dimension):
                byte_val = word_hash[idx % len(word_hash)]
                val = ((byte_val / 128.0) - 1.0)
                vector[idx] += val

        # Normalize to unit vector
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        else:
            vector[0] = 1.0

        return vector


def get_embedding_provider(provider_name: Optional[str] = None) -> BaseEmbeddingProvider:
    """
    Factory function to retrieve configured embedding provider.
    """
    provider = (provider_name or settings.EMBEDDING_PROVIDER).lower()

    if provider in ("gemini", "google"):
        try:
            return GeminiEmbeddingProvider()
        except Exception as e:
            logger.warning(f"Could not initialize GeminiEmbeddingProvider: {e}. Falling back to Deterministic provider.")
            return DeterministicHashEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)
    elif provider in ("deterministic", "hash", "mock", "test"):
        return DeterministicHashEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
