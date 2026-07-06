import sqlite3
import pytest
from datetime import datetime, timezone, timedelta

from swarm_memory.core.models import Fact, Run
from swarm_memory.store.db import get_initialized_db


@pytest.fixture
def db_conn(tmp_path):
    db_path = tmp_path / "test.db"
    conn, vec_loaded = get_initialized_db(str(db_path))
    yield conn, vec_loaded
    conn.close()


def test_schema_creation(db_conn):
    conn, _ = db_conn
    # Verify tables exist
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row["name"] for row in cursor.fetchall()}
    assert "runs" in tables
    assert "facts" in tables
    assert "facts_fts" in tables


def test_insert_and_retrieve_run(db_conn):
    conn, _ = db_conn
    run = Run(
        agent_id="agent-007",
        repo="Nuclues/swarm_memory",
        started_at=datetime.now(timezone.utc),
    )

    with conn:
        conn.execute(
            """
            INSERT INTO runs (id, agent_id, repo, started_at) 
            VALUES (?, ?, ?, ?)
        """,
            (run.id, run.agent_id, run.repo, run.started_at.isoformat()),
        )

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs WHERE id = ?", (run.id,))
    row = cursor.fetchone()
    assert row is not None
    assert row["agent_id"] == "agent-007"
    assert row["repo"] == "Nuclues/swarm_memory"


def test_insert_and_retrieve_fact(db_conn):
    conn, _ = db_conn
    # Setup run first due to FK
    run = Run(agent_id="test", repo="test", started_at=datetime.now(timezone.utc))
    with conn:
        conn.execute(
            "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
            (run.id, run.agent_id, run.repo, run.started_at.isoformat()),
        )

    fact = Fact(
        content="auth uses JWT",
        scope="auth",
        valid_from=datetime.now(timezone.utc),
        source_run_id=run.id,
        content_hash="hash123",
    )

    with conn:
        conn.execute(
            """
            INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                fact.id,
                fact.content,
                fact.fact_type,
                fact.scope,
                fact.confidence,
                fact.valid_from.isoformat(),
                fact.source_run_id,
                fact.content_hash,
            ),
        )

        # Insert into FTS
        conn.execute(
            "INSERT INTO facts_fts (fact_id, content, scope) VALUES (?, ?, ?)",
            (fact.id, fact.content, fact.scope),
        )

    # Read current facts for scope
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM facts 
        WHERE scope = ? AND valid_to IS NULL AND superseded_by IS NULL
    """,
        (fact.scope,),
    )

    row = cursor.fetchone()
    assert row is not None
    assert row["content"] == "auth uses JWT"

    # Test FTS
    cursor.execute("SELECT fact_id FROM facts_fts WHERE facts_fts MATCH ?", ("auth",))
    fts_row = cursor.fetchone()
    assert fts_row is not None
    assert fts_row["fact_id"] == fact.id


def test_update_fact(db_conn):
    conn, _ = db_conn
    # Setup run
    run = Run(agent_id="test2", repo="test2", started_at=datetime.now(timezone.utc))
    with conn:
        conn.execute(
            "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
            (run.id, run.agent_id, run.repo, run.started_at.isoformat()),
        )

    # Insert initial fact
    fact1 = Fact(
        content="Original content",
        scope="test",
        valid_from=datetime.now(timezone.utc),
        source_run_id=run.id,
        content_hash="hash1",
    )
    with conn:
        conn.execute(
            """
            INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                fact1.id,
                fact1.content,
                fact1.fact_type,
                fact1.scope,
                fact1.confidence,
                fact1.valid_from.isoformat(),
                fact1.source_run_id,
                fact1.content_hash,
            ),
        )

    # Update fact (bi-temporal: soft-delete fact1 and insert fact2)
    fact2 = Fact(
        content="Updated content",
        scope="test",
        valid_from=datetime.now(timezone.utc),
        source_run_id=run.id,
        content_hash="hash2",
    )
    now = datetime.now(timezone.utc)

    with conn:
        # 1. Insert new fact
        conn.execute(
            """
            INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                fact2.id,
                fact2.content,
                fact2.fact_type,
                fact2.scope,
                fact2.confidence,
                fact2.valid_from.isoformat(),
                fact2.source_run_id,
                fact2.content_hash,
            ),
        )

        # 2. Invalidate old fact
        conn.execute(
            """
            UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?
        """,
            (now.isoformat(), fact2.id, fact1.id),
        )

    # Verify update
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM facts WHERE id = ?", (fact1.id,))
    old_row = cursor.fetchone()
    assert old_row["valid_to"] is not None
    assert old_row["superseded_by"] == fact2.id

    cursor.execute("SELECT * FROM facts WHERE id = ?", (fact2.id,))
    new_row = cursor.fetchone()
    assert new_row["valid_to"] is None
    assert new_row["superseded_by"] is None
    assert new_row["content"] == "Updated content"


def test_delete_fact(db_conn):
    conn, _ = db_conn
    # Setup run
    run = Run(agent_id="test3", repo="test3", started_at=datetime.now(timezone.utc))
    with conn:
        conn.execute(
            "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
            (run.id, run.agent_id, run.repo, run.started_at.isoformat()),
        )

    # Insert fact
    fact = Fact(
        content="To be deleted",
        scope="test",
        valid_from=datetime.now(timezone.utc),
        source_run_id=run.id,
        content_hash="hash3",
    )
    with conn:
        conn.execute(
            """
            INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                fact.id,
                fact.content,
                fact.fact_type,
                fact.scope,
                fact.confidence,
                fact.valid_from.isoformat(),
                fact.source_run_id,
                fact.content_hash,
            ),
        )

    # Delete fact (bi-temporal: set valid_to)
    now = datetime.now(timezone.utc)
    with conn:
        conn.execute(
            "UPDATE facts SET valid_to = ? WHERE id = ?", (now.isoformat(), fact.id)
        )

    # Verify deletion
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM facts WHERE id = ?", (fact.id,))
    row = cursor.fetchone()
    assert row["valid_to"] is not None

    # Verify fact is excluded from current queries
    cursor.execute("SELECT * FROM facts WHERE valid_to IS NULL")
    active_rows = cursor.fetchall()
    assert len(active_rows) == 0


def test_duplicate_content_hash_rejected(db_conn):
    conn, _ = db_conn
    run = Run(agent_id="test4", repo="test4", started_at=datetime.now(timezone.utc))
    with conn:
        conn.execute(
            "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
            (run.id, run.agent_id, run.repo, run.started_at.isoformat()),
        )

    fact = Fact(
        content="Hash duplicate test",
        scope="test",
        valid_from=datetime.now(timezone.utc),
        source_run_id=run.id,
        content_hash="duplicate_hash",
    )
    with conn:
        conn.execute(
            """
            INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                fact.id,
                fact.content,
                fact.fact_type,
                fact.scope,
                fact.confidence,
                fact.valid_from.isoformat(),
                fact.source_run_id,
                fact.content_hash,
            ),
        )

    fact2 = Fact(
        content="Another content but same hash",
        scope="test",
        valid_from=datetime.now(timezone.utc),
        source_run_id=run.id,
        content_hash="duplicate_hash",
    )

    with pytest.raises(
        sqlite3.IntegrityError, match="UNIQUE constraint failed: facts.content_hash"
    ):
        with conn:
            conn.execute(
                """
                INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    fact2.id,
                    fact2.content,
                    fact2.fact_type,
                    fact2.scope,
                    fact2.confidence,
                    fact2.valid_from.isoformat(),
                    fact2.source_run_id,
                    fact2.content_hash,
                ),
            )


def test_fk_enforcement(db_conn):
    conn, _ = db_conn
    fact = Fact(
        content="FK enforcement test",
        scope="test",
        valid_from=datetime.now(timezone.utc),
        source_run_id="non_existent_run_id",
        content_hash="fk_hash",
    )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        with conn:
            conn.execute(
                """
                INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    fact.id,
                    fact.content,
                    fact.fact_type,
                    fact.scope,
                    fact.confidence,
                    fact.valid_from.isoformat(),
                    fact.source_run_id,
                    fact.content_hash,
                ),
            )


def test_pragma_verification(db_conn):
    conn, _ = db_conn
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode;")
    journal_mode = cursor.fetchone()[0]
    assert journal_mode.lower() == "wal"

    cursor.execute("PRAGMA foreign_keys;")
    fk = cursor.fetchone()[0]
    assert fk == 1


def test_index_existence(db_conn):
    conn, _ = db_conn
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indexes = {row["name"] for row in cursor.fetchall()}
    assert "idx_facts_current" in indexes
    assert "idx_facts_valid_range" in indexes
    assert "idx_facts_superseded_by" in indexes
    assert "idx_facts_source_run" in indexes
    assert "idx_facts_type" in indexes


def test_point_in_time_query(db_conn):
    conn, _ = db_conn
    run = Run(agent_id="test5", repo="test5", started_at=datetime.now(timezone.utc))
    with conn:
        conn.execute(
            "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
            (run.id, run.agent_id, run.repo, run.started_at.isoformat()),
        )

    t0 = datetime.now(timezone.utc) - timedelta(days=2)
    t1 = datetime.now(timezone.utc) - timedelta(days=1)

    fact1 = Fact(
        content="Past fact",
        scope="test",
        valid_from=t0,
        source_run_id=run.id,
        content_hash="hash_past",
    )
    with conn:
        conn.execute(
            """
            INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                fact1.id,
                fact1.content,
                fact1.fact_type,
                fact1.scope,
                fact1.confidence,
                fact1.valid_from.isoformat(),
                fact1.source_run_id,
                fact1.content_hash,
            ),
        )

        # Soft delete at t1
        conn.execute(
            "UPDATE facts SET valid_to = ? WHERE id = ?", (t1.isoformat(), fact1.id)
        )

    cursor = conn.cursor()

    # Query at T=t0 + 12 hours (fact should be valid)
    query_time = (t0 + timedelta(hours=12)).isoformat()
    cursor.execute(
        """
        SELECT * FROM facts 
        WHERE scope = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
    """,
        ("test", query_time, query_time),
    )
    rows = cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["content"] == "Past fact"

    # Query at T=now (fact should be invalid)
    query_time = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        """
        SELECT * FROM facts 
        WHERE scope = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
    """,
        ("test", query_time, query_time),
    )
    rows = cursor.fetchall()
    assert len(rows) == 0


def test_facts_vec_creation(db_conn):
    conn, vec_loaded = db_conn
    if not vec_loaded:
        pytest.skip("sqlite-vec extension not loaded")

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row["name"] for row in cursor.fetchall()}
    assert "facts_vec" in tables


def test_get_db_default_path(monkeypatch, tmp_path):
    import swarm_memory.core.config as config
    from swarm_memory.store.db import get_initialized_db

    # Mock default DB path
    mock_db_path = tmp_path / "default.db"
    monkeypatch.setattr(config, "DB_PATH", str(mock_db_path))

    conn, _ = get_initialized_db()

    # Verify DB was created at mock path
    assert mock_db_path.exists()
    conn.close()
