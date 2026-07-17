from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from swarm_memory.core.embeddings import EmbeddingModel
from swarm_memory.ingestion.supersession import ContradictionDetector
from swarm_memory.ingestion.writer import FactWriter
from swarm_memory.retrieval.reader import FactReader
from swarm_memory.store.db import get_initialized_db


# FIX-02: Module-scoped embedding model to prevent reloading costs across tests
@pytest.fixture(scope="session")
def shared_embedder():
    """Provides a lazily loaded EmbeddingModel that is shared across the test session."""
    return EmbeddingModel()


@pytest.fixture
def mock_embedder():
    """Provides a mocked embedder for fast unit tests without loading the model."""
    embedder = MagicMock()
    # Mock embed to return a fake embedding of length 768 float32s
    embedder.embed.return_value = b"\x00" * (768 * 4)
    embedder.embed_query.return_value = b"\x00" * (768 * 4)
    return embedder


@pytest.fixture
def mock_detector():
    """Provides a mocked ContradictionDetector that returns no contradictions by default."""
    detector = MagicMock(spec=ContradictionDetector)
    # Use side_effect to return a fresh list on each call to prevent in-place mutation bugs
    detector.detect_contradictions = AsyncMock(side_effect=lambda *args, **kwargs: [])
    return detector


@pytest.fixture
def db():
    """Provides an initialized in-memory SQLite database connection."""
    conn = get_initialized_db(":memory:")
    # Insert a dummy run
    conn.execute(
        "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
        ("run_1", "agent_1", "repo", datetime.now(UTC).isoformat()),
    )
    yield conn
    conn.close()


@pytest.fixture
def db_with_vec(db):
    """Provides an initialized DB with sqlite-vec confirmed available (or skips test)."""
    # SQLite vec is initialized in get_initialized_db, let's verify
    try:
        db.execute("SELECT vec_version()")
    except Exception:
        pytest.skip("sqlite-vec is not available in this environment")
    return db


@pytest.fixture
def writer(db, mock_embedder, mock_detector):
    """Provides a FactWriter wired to the in-memory db and mocked services."""
    return FactWriter(db, mock_embedder, mock_detector)


@pytest.fixture
def reader(db, mock_embedder):
    """Provides a FactReader wired to the in-memory db with mocked embedder."""
    return FactReader(db, mock_embedder, vec_available=False)


@pytest.fixture
def seed_facts(db, writer):
    """Helper fixture to easily seed facts for retrieval tests."""

    async def _seed(facts_data: list[dict]):
        # facts_data = [{"content": "...", "scope": "...", "fact_type": "insight"}]
        ids = []
        for d in facts_data:
            res = await writer.write_fact(
                content=d.get("content", "default"),
                scope=d.get("scope", "test_scope"),
                run_id="run_1",
                fact_type=d.get("fact_type", "insight"),
                confidence=d.get("confidence", 1.0),
                valid_from=d.get("valid_from"),
            )
            ids.append(res.fact_id)
        return ids

    return _seed
