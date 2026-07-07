from unittest.mock import MagicMock

import numpy as np
import pytest

from swarm_memory.core.embeddings import PREFIX_DOCUMENT, PREFIX_QUERY, EmbeddingModel


@pytest.fixture
def mock_embedder() -> EmbeddingModel:
    embedder = EmbeddingModel(model_name="mock-model", dim=768)
    mock_model = MagicMock()
    mock_model.encode = lambda text, **_kwargs: np.ones(768, dtype=np.float32)
    embedder._model = mock_model
    return embedder


class TestEmbeddingModel:
    def test_lazy_loading_not_loaded_at_init(self):
        """Model should NOT be loaded at construction time."""
        embedder = EmbeddingModel(model_name="some-model", dim=768)
        assert embedder._model is None
        assert not embedder.is_loaded

    def test_embed_returns_bytes_of_correct_length(self, mock_embedder):
        """embed() must return exactly (dim * 4) bytes (float32 = 4 bytes each)."""
        result = mock_embedder.embed("test text")
        assert isinstance(result, bytes)
        assert len(result) == 768 * 4

    def test_embed_uses_document_prefix_by_default(self, mock_embedder):
        """embed() should prepend PREFIX_DOCUMENT by default."""
        calls = []
        mock_embedder._model.encode = lambda text, **kw: (
            calls.append(text),
            np.ones(768, dtype=np.float32),
        )[1]
        mock_embedder.embed("some content")
        assert calls[0].startswith(PREFIX_DOCUMENT)

    def test_embed_query_uses_query_prefix(self, mock_embedder):
        """embed_query() must use PREFIX_QUERY, not PREFIX_DOCUMENT."""
        calls = []
        mock_embedder._model.encode = lambda text, **kw: (
            calls.append(text),
            np.ones(768, dtype=np.float32),
        )[1]
        mock_embedder.embed_query("search this")
        assert calls[0].startswith(PREFIX_QUERY)
        assert not calls[0].startswith(PREFIX_DOCUMENT)

    def test_dim_truncation(self):
        """Output should be truncated to configured dim even if model outputs more."""
        embedder = EmbeddingModel(dim=256)
        mock_model = MagicMock()
        mock_model.encode = lambda text, **kw: np.ones(768, dtype=np.float32)
        embedder._model = mock_model

        result = embedder.embed("test")
        assert len(result) == 256 * 4

    def test_embed_batch_returns_correct_count(self, mock_embedder):
        """embed_batch() should return one bytes object per input text."""
        texts = ["fact one", "fact two", "fact three"]
        mock_embedder._model.encode = lambda texts_list, **kw: np.ones(
            (len(texts_list), 768), dtype=np.float32
        )
        results = mock_embedder.embed_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, bytes) for r in results)

    def test_embed_batch_empty_input(self):
        """embed_batch() with empty list should return empty list without calling model."""
        embedder = EmbeddingModel(dim=768)
        mock_encode = MagicMock()
        embedder._model = MagicMock()
        embedder._model.encode = mock_encode

        results = embedder.embed_batch([])
        assert results == []
        mock_encode.assert_not_called()

    def test_dim_property(self, mock_embedder):
        """dim property should return the configured dimension."""
        assert mock_embedder.dim == 768

    def test_is_loaded_after_embed(self, mock_embedder):
        """is_loaded should be True after model has been used."""
        assert mock_embedder.is_loaded
