"""
swarm_memory/retrieval/embeddings.py

Embedding model wrapper for SwarmMemory.

Wraps nomic-embed-text-v1.5 via sentence-transformers to produce
normalized, dimension-truncated float32 vectors that are stored
directly in the sqlite-vec virtual table.

Key design decisions:
- Lazy loading: the model is loaded on first use, not at import.
  This keeps server startup fast and avoids loading the model in test environments.
- Prefix protocol: nomic was trained with mandatory prefixes that improve
  retrieval accuracy ~15%. "search_document:" is for facts being stored;
  "search_query:" is for queries being searched.
- Matryoshka truncation: the model supports slicing output dimensions.
  Full 768-dim gives best quality; 256-dim gives 3× storage savings ~5% quality loss.
- Output format: float32 bytes — the exact format sqlite-vec's vec0 expects.

TODO: Implement async background model loading to completely hide cold starts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from swarm_memory.core import config

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Prefix protocol required by nomic-embed-text-v1.5.
# These prefixes were part of the model's training — omitting them degrades
# retrieval accuracy. The model distinguishes "being found" from "looking for".
PREFIX_DOCUMENT = "search_document: "  # used when embedding facts for storage
PREFIX_QUERY = "search_query: "  # used when embedding a search query


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for nomic-embed-text-v1.5.

    Thread safety: fastembed encode() is thread-safe for inference.
    The lazy _load_model() call is NOT thread-safe — in a multi-threaded SSE
    server, ensure the model is warmed up once at startup before concurrent
    requests arrive (call embed() once during initialization).

    Example:
        embedder = EmbeddingModel()
        # For writing a fact:
        vec_bytes = embedder.embed("auth uses session tokens")
        # For searching:
        query_bytes = embedder.embed_query("how does auth work?")
    """

    def __init__(
        self,
        model_name: str = config.EMBED_MODEL,
        dim: int = config.EMBED_DIM,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._model: TextEmbedding | None = None  # loaded lazily

    # ── Private ────────────────────────────────────────────────────────────

    def _load_model(self) -> TextEmbedding:
        """
        Load the fastembed ONNX model on first call.
        """
        if self._model is None:
            logger.info("Loading ONNX embedding model '%s' (first call)…", self._model_name)
            from fastembed import TextEmbedding  # noqa: PLC0415

            self._model = TextEmbedding(model_name=self._model_name)
            logger.info("Embedding model loaded. Configured output dim: %d", self._dim)
        return self._model

    def _to_bytes(self, vector: np.ndarray) -> bytes:
        """Truncate to configured dim, L2-normalize, and serialise to float32 bytes for sqlite-vec."""
        vector = vector[: self._dim]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return np.array(vector, dtype=np.float32).tobytes()

    # ── Public API ──────────────────────────────────────────────────────────

    def embed(self, text: str, prefix: str = PREFIX_DOCUMENT) -> bytes:
        """
        Embed a single piece of text and return float32 bytes.

        Args:
            text:   The raw text to embed (fact content, query string, etc.).
            prefix: The nomic prefix to prepend. Use PREFIX_DOCUMENT for facts
                    being stored, PREFIX_QUERY for search queries.

        Returns:
            float32 bytes of length (dim * 4), ready for sqlite-vec insertion.
        """
        model = self._load_model()
        vec_generator = model.embed([prefix + text])
        vec = next(vec_generator)
        return self._to_bytes(vec)

    def embed_query(self, query: str) -> bytes:
        """
        Embed a search query using the query prefix.

        Convenience wrapper around embed() — avoids callers having to know
        about the prefix protocol.

        Args:
            query: Natural language query string (e.g. "how does auth work?").

        Returns:
            float32 bytes ready for KNN comparison against stored fact vectors.
        """
        return self.embed(query, prefix=PREFIX_QUERY)

    def embed_batch(
        self,
        texts: list[str],
        prefix: str = PREFIX_DOCUMENT,
    ) -> list[bytes]:
        """
        Embed multiple texts in one forward pass (faster than calling embed() N times).

        Primarily used by the writer during bulk fact ingestion. Each text is
        prefixed before encoding.

        Args:
            texts:  List of raw text strings.
            prefix: Prefix to prepend to each text.

        Returns:
            List of float32 byte strings, one per input text, in the same order.
        """
        if not texts:
            return []

        model = self._load_model()
        prefixed = [prefix + t for t in texts]
        
        # fastembed handles batching natively
        vec_generator = model.embed(prefixed, batch_size=32)
        return [self._to_bytes(vec) for vec in vec_generator]

    @property
    def dim(self) -> int:
        """The configured output dimension."""
        return self._dim

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded into memory."""
        return self._model is not None
