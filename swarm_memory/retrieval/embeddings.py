"""
swarm_memory/retrieval/embeddings.py

Embedding model wrapper for SwarmMemory.

Wraps nomic-embed-text-v1.5 via sentence-transformers to produce
normalized, dimension-truncated float32 vectors that are stored
directly in the sqlite-vec virtual table.

Key design decisions:
- Lazy loading: the model (~500MB RAM) is loaded on first use, not at import.
  This keeps server startup fast and avoids loading the model in test environments.
- Prefix protocol: nomic was trained with mandatory prefixes that improve
  retrieval accuracy ~15%. "search_document:" is for facts being stored;
  "search_query:" is for queries being searched.
- Matryoshka truncation: the model supports slicing output dimensions.
  Full 768-dim gives best quality; 256-dim gives 3× storage savings ~5% quality loss.
- Output format: float32 bytes — the exact format sqlite-vec's vec0 expects.

TODO: Migrate to ONNX Runtime for 2-4× faster CPU inference once the pipeline
      is validated. Pre-exported ONNX model: nomic-ai/nomic-embed-text-v1.5-ONNX
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from swarm_memory.core import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Prefix protocol required by nomic-embed-text-v1.5.
# These prefixes were part of the model's training — omitting them degrades
# retrieval accuracy. The model distinguishes "being found" from "looking for".
PREFIX_DOCUMENT = "search_document: "  # used when embedding facts for storage
PREFIX_QUERY = "search_query: "        # used when embedding a search query


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for nomic-embed-text-v1.5.

    Thread safety: sentence-transformers encode() is thread-safe for inference.
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
        self._model: SentenceTransformer | None = None  # loaded lazily

    # ── Private ────────────────────────────────────────────────────────────

    def _load_model(self) -> SentenceTransformer:
        """
        Load the sentence-transformers model on first call.

        nomic-embed-text-v1.5 uses custom pooling code hosted on HuggingFace,
        so trust_remote_code=True is required.
        """
        if self._model is None:
            logger.info("Loading embedding model '%s' (first call)…", self._model_name)
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,  # required: nomic has custom pooling code
            )
            logger.info("Embedding model loaded. Output dim: %d", self._dim)
        return self._model

    def _to_bytes(self, vector: np.ndarray) -> bytes:
        """Truncate to configured dim and serialise to float32 bytes for sqlite-vec."""
        return np.array(vector[: self._dim], dtype=np.float32).tobytes()

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
        vec = model.encode(
            prefix + text,
            normalize_embeddings=True,  # unit-normalize for cosine similarity
            show_progress_bar=False,
        )
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
        vecs = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,  # safe default for CPU inference
        )
        return [self._to_bytes(vec) for vec in vecs]

    @property
    def dim(self) -> int:
        """The configured output dimension."""
        return self._dim

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded into memory."""
        return self._model is not None
