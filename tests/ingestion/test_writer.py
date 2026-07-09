from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from swarm_memory.core.models import Fact, Relationship
from swarm_memory.ingestion.writer import FactWriter
from swarm_memory.store.db import get_initialized_db


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    # Mock embed to return a fake embedding of length 768 float32s
    embedder.embed.return_value = b"\x00" * (768 * 4)
    return embedder


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    # Use side_effect to return a fresh list on each call to prevent in-place mutation bugs
    detector.detect_contradictions = AsyncMock(side_effect=lambda *args, **kwargs: [])
    return detector


@pytest.fixture
def db():
    # Use memory database
    conn = get_initialized_db(":memory:")
    # Insert a dummy run
    conn.execute(
        "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
        ("run_1", "agent_1", "repo", datetime.now(UTC).isoformat()),
    )
    yield conn
    conn.close()


@pytest.mark.asyncio
async def test_write_fact_new(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)
    result = await writer.write_fact(
        content="Test content",
        scope="test_scope",
        run_id="run_1",
    )
    assert result.status == "created"
    assert result.fact_id is not None
    assert len(result.superseded_ids) == 0

    # Verify DB state
    fact_row = db.execute("SELECT * FROM facts WHERE id = ?", [result.fact_id]).fetchone()
    assert fact_row is not None
    assert fact_row["content"] == "Test content"
    assert fact_row["scope"] == "test_scope"


@pytest.mark.asyncio
async def test_write_fact_duplicate(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)
    res1 = await writer.write_fact(
        content="Test content",
        scope="test_scope",
        run_id="run_1",
    )
    assert res1.status == "created"

    res2 = await writer.write_fact(
        content="Test content",
        scope="test_scope",
        run_id="run_1",
    )
    assert res2.status == "duplicate"
    assert res2.fact_id == res1.fact_id
    assert len(res2.superseded_ids) == 0


@pytest.mark.asyncio
async def test_write_fact_supersedes(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)

    # We will mock the detector to return SUPERSEDES for the second fact
    res1 = await writer.write_fact(
        content="Old content",
        scope="test_scope",
        run_id="run_1",
    )
    assert res1.status == "created"

    # Mock find_similar to return this fact
    old_fact = Fact(
        id=res1.fact_id,
        content="Old content",
        scope="test_scope",
        valid_from=datetime.now(UTC),
        source_run_id="run_1",
    )
    writer._find_similar_valid_facts = MagicMock(return_value=[old_fact])
    mock_detector.detect_contradictions = AsyncMock(
        return_value=[(old_fact, Relationship.SUPERSEDES)]
    )

    res2 = await writer.write_fact(
        content="New content",
        scope="test_scope",
        run_id="run_1",
    )
    assert res2.status == "created"
    assert res1.fact_id in res2.superseded_ids

    # Check that old fact is superseded
    old_fact_row = db.execute("SELECT * FROM facts WHERE id = ?", [res1.fact_id]).fetchone()
    assert old_fact_row["superseded_by"] == res2.fact_id
    assert old_fact_row["valid_to"] is not None


@pytest.mark.asyncio
async def test_write_fact_rollback(mock_embedder, mock_detector):
    mock_db = MagicMock()

    # 1. content_hash check returns None
    # 2. BEGIN
    # 3. INSERT throws exception
    def side_effect(*args, **kwargs):
        if "SELECT id FROM facts WHERE content_hash" in args[0]:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            return mock_cursor
        elif "BEGIN" in args[0]:
            return MagicMock()
        elif "INSERT INTO facts" in args[0]:
            raise Exception("Mock DB Error")
        return MagicMock()

    mock_db.execute.side_effect = side_effect
    writer = FactWriter(mock_db, mock_embedder, mock_detector)

    with pytest.raises(Exception, match="Mock DB Error"):
        await writer.write_fact(
            content="Error content",
            scope="error_scope",
            run_id="run_1",
        )
    mock_db.execute.assert_any_call("ROLLBACK")


def test_find_similar_valid_facts_mocked(mock_embedder, mock_detector):
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [{"fact_id": "123", "distance": 0.5}]

    mock_fact_cursor = MagicMock()
    mock_fact_cursor.fetchone.return_value = {
        "id": "123",
        "content": "hello",
        "scope": "test_scope",
        "valid_from": "2023-01-01T12:00:00Z",
        "valid_to": "2023-12-31T12:00:00Z",
        "created_at": "2023-01-01T12:00:00Z",
        "fact_type": "insight",
        "confidence": 1.0,
        "superseded_by": None,
        "source_run_id": "r",
        "content_hash": "c",
        "source_branch": None,
        "extraction_method": "llm",
    }

    def execute_side_effect(*args, **kwargs):
        if "facts_vec" in args[0]:
            return mock_cursor
        else:
            return mock_fact_cursor

    mock_db.execute.side_effect = execute_side_effect
    writer = FactWriter(mock_db, mock_embedder, mock_detector)

    facts = writer._find_similar_valid_facts(b"\x00" * 768 * 4, "test_scope")
    assert len(facts) == 1
    assert facts[0].id == "123"


def test_find_similar_valid_facts_mocked_no_fact(mock_embedder, mock_detector):
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    # Add one valid distance and one invalid distance to cover branch
    mock_cursor.fetchall.return_value = [
        {"fact_id": "123", "distance": 0.5},
        {"fact_id": "999", "distance": 1.0},
    ]

    mock_fact_cursor = MagicMock()
    mock_fact_cursor.fetchone.return_value = None

    def execute_side_effect(*args, **kwargs):
        if "facts_vec" in args[0]:
            return mock_cursor
        else:
            return mock_fact_cursor

    mock_db.execute.side_effect = execute_side_effect
    writer = FactWriter(mock_db, mock_embedder, mock_detector)

    facts = writer._find_similar_valid_facts(b"\x00" * 768 * 4, "test_scope")
    assert len(facts) == 0


def test_find_similar_valid_facts(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)

    # Empty DB
    facts = writer._find_similar_valid_facts(b"\x00" * 768 * 4, "test_scope")
    assert len(facts) == 0


@pytest.mark.asyncio
async def test_write_fact_valid_from_normalization(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)
    result = await writer.write_fact(
        content="Valid from content",
        scope="test_scope",
        run_id="run_1",
        valid_from="2023-01-01T12:00:00Z",
    )
    assert result.status == "created"

    fact_row = db.execute("SELECT * FROM facts WHERE id = ?", [result.fact_id]).fetchone()
    assert "2023-01-01" in fact_row["valid_from"]


@pytest.mark.asyncio
async def test_write_fact_invalid_date(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)
    # The valid_from format might be invalid entirely
    with pytest.raises(ValueError):
        await writer.write_fact(
            content="Invalid date content",
            scope="test_scope",
            run_id="run_1",
            valid_from="not-a-date",
        )


@pytest.mark.asyncio
async def test_supersession_chain_a_b_c(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)
    
    # Write A
    res_a = await writer.write_fact(content="A", scope="scope", run_id="run_1")
    
    # Write B, superseding A
    mock_detector.detect_contradictions = AsyncMock(
        return_value=[(Fact(id=res_a.fact_id, content="A", scope="scope", valid_from=datetime.now(UTC), source_run_id="run_1"), Relationship.SUPERSEDES)]
    )
    res_b = await writer.write_fact(content="B", scope="scope", run_id="run_1")
    assert res_a.fact_id in res_b.superseded_ids

    # Write C, superseding B
    mock_detector.detect_contradictions = AsyncMock(
        return_value=[(Fact(id=res_b.fact_id, content="B", scope="scope", valid_from=datetime.now(UTC), source_run_id="run_1"), Relationship.SUPERSEDES)]
    )
    res_c = await writer.write_fact(content="C", scope="scope", run_id="run_1")
    assert res_b.fact_id in res_c.superseded_ids

    # Verify chain: A is superseded by B, B is superseded by C
    row_a = db.execute("SELECT * FROM facts WHERE id = ?", [res_a.fact_id]).fetchone()
    row_b = db.execute("SELECT * FROM facts WHERE id = ?", [res_b.fact_id]).fetchone()
    row_c = db.execute("SELECT * FROM facts WHERE id = ?", [res_c.fact_id]).fetchone()

    assert row_a["superseded_by"] == res_b.fact_id
    assert row_b["superseded_by"] == res_c.fact_id
    assert row_c["superseded_by"] is None


@pytest.mark.asyncio
async def test_hint_valid(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)
    res1 = await writer.write_fact(content="Old", scope="scope", run_id="run_1")
    
    # Write with supersedes_hint, should bypass detector mock logic for that fact
    res2 = await writer.write_fact(
        content="New", scope="scope", run_id="run_1", supersedes_hint=res1.fact_id
    )
    assert res2.status == "created"
    assert res1.fact_id in res2.superseded_ids
    row = db.execute("SELECT * FROM facts WHERE id = ?", [res1.fact_id]).fetchone()
    assert row["superseded_by"] == res2.fact_id


@pytest.mark.asyncio
async def test_hint_invalid(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)
    # Hint points to nonexistent fact, should be ignored
    res = await writer.write_fact(
        content="New", scope="scope", run_id="run_1", supersedes_hint="invalid-id"
    )
    assert res.status == "created"
    assert len(res.superseded_ids) == 0


@pytest.mark.asyncio
async def test_hint_already_superseded(db, mock_embedder, mock_detector):
    writer = FactWriter(db, mock_embedder, mock_detector)
    res1 = await writer.write_fact(content="Old", scope="scope", run_id="run_1")
    res2 = await writer.write_fact(
        content="New", scope="scope", run_id="run_1", supersedes_hint=res1.fact_id
    )
    
    # Try to use res1.fact_id as a hint again
    res3 = await writer.write_fact(
        content="Newer", scope="scope", run_id="run_1", supersedes_hint=res1.fact_id
    )
    assert res3.status == "created"
    # The hint was ignored because it's already superseded, so it shouldn't be in superseded_ids
    assert res1.fact_id not in res3.superseded_ids


@pytest.mark.asyncio
async def test_fts_not_deleted_on_supersession(db, mock_embedder, mock_detector):
    # BUG-3: FTS entries are currently NOT deleted on supersession.
    # This test asserts current behavior.
    writer = FactWriter(db, mock_embedder, mock_detector)
    res1 = await writer.write_fact(content="Old", scope="scope", run_id="run_1")
    
    res2 = await writer.write_fact(
        content="New", scope="scope", run_id="run_1", supersedes_hint=res1.fact_id
    )
    
    # Check FTS
    fts_row = db.execute("SELECT * FROM facts_fts WHERE fact_id = ?", [res1.fact_id]).fetchone()
    assert fts_row is not None, "FTS entry should still exist (BUG-3 behavior)"
