from datetime import UTC, datetime

import pytest

from swarm_memory.core.models import Fact, Run
from swarm_memory.store.db import get_initialized_db


@pytest.fixture
def db_conn(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_initialized_db(str(db_path))
    yield conn
    conn.close()

def test_schema_creation(db_conn):
    # Verify tables exist
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row['name'] for row in cursor.fetchall()}
    assert "runs" in tables
    assert "facts" in tables
    assert "facts_fts" in tables

def test_insert_and_retrieve_run(db_conn):
    run = Run(
        agent_id="agent-007",
        repo="Nuclues/swarm_memory",
        started_at=datetime.now(UTC)
    )

    with db_conn:
        db_conn.execute("""
            INSERT INTO runs (id, agent_id, repo, started_at)
            VALUES (?, ?, ?, ?)
        """, (run.id, run.agent_id, run.repo, run.started_at.isoformat()))

    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM runs WHERE id = ?", (run.id,))
    row = cursor.fetchone()
    assert row is not None
    assert row['agent_id'] == "agent-007"
    assert row['repo'] == "Nuclues/swarm_memory"

def test_insert_and_retrieve_fact(db_conn):
    # Setup run first due to FK
    run = Run(agent_id="test", repo="test", started_at=datetime.now(UTC))
    with db_conn:
        db_conn.execute("INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
                        (run.id, run.agent_id, run.repo, run.started_at.isoformat()))

    fact = Fact(
        content="auth uses JWT",
        scope="auth",
        valid_from=datetime.now(UTC),
        source_run_id=run.id,
        content_hash="hash123"
    )

    with db_conn:
        db_conn.execute("""
            INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (fact.id, fact.content, fact.fact_type, fact.scope, fact.confidence,
              fact.valid_from.isoformat(), fact.source_run_id, fact.content_hash))

        # Insert into FTS
        db_conn.execute("INSERT INTO facts_fts (fact_id, content, scope) VALUES (?, ?, ?)",
                        (fact.id, fact.content, fact.scope))

    # Read current facts for scope
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT * FROM facts
        WHERE scope = ? AND valid_to IS NULL AND superseded_by IS NULL
    """, (fact.scope,))

    row = cursor.fetchone()
    assert row is not None
    assert row['content'] == "auth uses JWT"

    # Test FTS
    cursor.execute("SELECT fact_id FROM facts_fts WHERE facts_fts MATCH ?", ("auth",))
    fts_row = cursor.fetchone()
    assert fts_row is not None
    assert fts_row['fact_id'] == fact.id
