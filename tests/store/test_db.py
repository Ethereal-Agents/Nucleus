"""
tests/store/test_db.py

Phase 5a — Write-Path Tests: Storage Foundation (§2.2, §2.3, §7 Phase 1).

Tests the SQLite schema, indexes, PRAGMAs, and bi-temporal query patterns described
in implementation_plan.md §2.2 (Key Queries) and §10 (Correctness Invariants).
"""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from swarm_memory.core.models import Fact, Run
from swarm_memory.store.db import get_initialized_db


@pytest.fixture
def db_conn(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_initialized_db(str(db_path))
    yield conn
    conn.close()


@pytest.fixture
def db_mem():
    """In-memory DB (faster, used for pure schema/constraint tests)."""
    conn = get_initialized_db(":memory:")
    yield conn
    conn.close()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _insert_run(conn, run_id="run-1", agent_id="agent", repo="repo"):
    conn.execute(
        "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
        (run_id, agent_id, repo, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return run_id


def _insert_fact(conn, fact_id, content, scope, run_id, valid_from=None, content_hash=None):
    vf = (valid_from or datetime.now(UTC)).isoformat()
    ch = content_hash or f"hash-{fact_id}"
    conn.execute(
        """INSERT INTO facts (id, content, fact_type, scope, confidence, valid_from,
                              source_run_id, content_hash)
           VALUES (?, ?, 'insight', ?, 1.0, ?, ?, ?)""",
        (fact_id, content, scope, vf, run_id, ch),
    )
    conn.execute(
        "INSERT INTO facts_fts (fact_id, content, scope) VALUES (?, ?, ?)",
        (fact_id, content, scope),
    )
    conn.commit()
    return fact_id


# ═══════════════════════════════════════════════════════════════════════════
# Original tests (retained)
# ═══════════════════════════════════════════════════════════════════════════


def test_schema_creation(db_conn):
    """All core tables must exist after get_initialized_db()."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row["name"] for row in cursor.fetchall()}
    assert "runs" in tables
    assert "facts" in tables
    assert "facts_fts" in tables


def test_insert_and_retrieve_run(db_conn):
    run = Run(agent_id="agent-007", repo="Nuclues/swarm_memory", started_at=datetime.now(UTC))

    with db_conn:
        db_conn.execute(
            """
            INSERT INTO runs (id, agent_id, repo, started_at)
            VALUES (?, ?, ?, ?)
        """,
            (run.id, run.agent_id, run.repo, run.started_at.isoformat()),
        )

    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM runs WHERE id = ?", (run.id,))
    row = cursor.fetchone()
    assert row is not None
    assert row["agent_id"] == "agent-007"
    assert row["repo"] == "Nuclues/swarm_memory"


def test_insert_and_retrieve_fact(db_conn):
    # Setup run first due to FK
    run = Run(agent_id="test", repo="test", started_at=datetime.now(UTC))
    with db_conn:
        db_conn.execute(
            "INSERT INTO runs (id, agent_id, repo, started_at) VALUES (?, ?, ?, ?)",
            (run.id, run.agent_id, run.repo, run.started_at.isoformat()),
        )

    fact = Fact(
        content="auth uses JWT",
        scope="auth",
        valid_from=datetime.now(UTC),
        source_run_id=run.id,
        content_hash="hash123",
    )

    with db_conn:
        db_conn.execute(
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
        db_conn.execute(
            "INSERT INTO facts_fts (fact_id, content, scope) VALUES (?, ?, ?)",
            (fact.id, fact.content, fact.scope),
        )

    # Read current facts for scope
    cursor = db_conn.cursor()
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


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5a: New tests — Schema, Indexes, PRAGMAs
# ═══════════════════════════════════════════════════════════════════════════


def test_all_five_indexes_created(db_conn):
    """All 5 indexes from §2.2 must exist after schema init (implementation_plan §2.2)."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    index_names = {row["name"] for row in cursor.fetchall()}

    expected = {
        "idx_facts_current",
        "idx_facts_valid_range",
        "idx_facts_superseded_by",
        "idx_facts_source_run",
        "idx_facts_type",
    }
    for idx in expected:
        assert idx in index_names, f"Missing index: {idx}"


def test_wal_mode_enabled(db_conn):
    """PRAGMA journal_mode=WAL must be set for concurrent-read performance (§7 Phase 1)."""
    row = db_conn.execute("PRAGMA journal_mode").fetchone()
    # In-memory DBs don't support WAL, so we use a file-based DB here (db_conn uses tmp_path)
    assert row[0] == "wal"


def test_foreign_keys_enforced(db_mem):
    """PRAGMA foreign_keys=ON must cause an IntegrityError on FK violation."""
    # Insert a fact referencing a non-existent run — FK should reject it
    with pytest.raises(sqlite3.IntegrityError):
        db_mem.execute(
            """INSERT INTO facts (id, content, fact_type, scope, confidence,
                                  valid_from, source_run_id, content_hash)
               VALUES ('f1', 'content', 'insight', 'repo', 1.0,
                       '2024-01-01T00:00:00Z', 'nonexistent-run-id', 'hash-fk-test')"""
        )
        db_mem.commit()


def test_content_hash_unique_constraint(db_mem):
    """Duplicate content_hash must raise IntegrityError (idempotent write dedup — §10 invariant 6)."""
    _insert_run(db_mem)
    _insert_fact(db_mem, "fact-1", "content A", "repo", "run-1", content_hash="same-hash")

    with pytest.raises(sqlite3.IntegrityError):
        _insert_fact(db_mem, "fact-2", "content B", "repo", "run-1", content_hash="same-hash")


def test_facts_vec_virtual_table_created(db_conn):
    """facts_vec virtual table should be created when sqlite-vec is available."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cursor.fetchall()}
    # facts_vec is created only when sqlite-vec loads successfully
    # If it's present, verify it — if not, the test notes graceful degradation
    try:
        import sqlite_vec  # noqa: F401

        assert "facts_vec" in tables, "sqlite-vec available but facts_vec table not created"
    except ImportError:
        pytest.skip(
            "sqlite-vec not installed — graceful degradation path, skipping vec table check"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5a: New tests — Bi-temporal queries (§2.3)
# ═══════════════════════════════════════════════════════════════════════════


def test_current_facts_query_excludes_superseded(db_mem):
    """
    §2.3 'READ: Current facts for a scope' — superseded facts must not appear.
    Invariant §10.1: no fact is deleted; §10.2: supersession is correct.
    """
    run_id = _insert_run(db_mem)
    past = datetime.now(UTC) - timedelta(hours=2)
    now = datetime.now(UTC)

    _insert_fact(db_mem, "old-fact", "auth uses JWT", "repo/auth", run_id, valid_from=past)
    _insert_fact(db_mem, "new-fact", "auth uses sessions", "repo/auth", run_id, valid_from=now)

    # Supersede old-fact
    db_mem.execute(
        "UPDATE facts SET valid_to = ?, superseded_by = 'new-fact' WHERE id = 'old-fact'",
        (now.isoformat(),),
    )
    db_mem.commit()

    rows = db_mem.execute(
        """SELECT id FROM facts
           WHERE scope = 'repo/auth' AND valid_to IS NULL AND superseded_by IS NULL""",
    ).fetchall()

    ids = [r["id"] for r in rows]
    assert "new-fact" in ids
    assert "old-fact" not in ids, "Superseded fact must be excluded from current query"


def test_point_in_time_as_of_query(db_mem):
    """
    §2.3 'READ: Point-in-time (AS OF) query' — must return the fact that was valid
    at a given timestamp, even if it has since been superseded (§10.3).
    """
    run_id = _insert_run(db_mem)
    t0 = datetime.now(UTC) - timedelta(hours=3)
    t1 = datetime.now(UTC) - timedelta(hours=1)
    t2 = datetime.now(UTC)

    # Insert old fact valid from t0, superseded at t1
    _insert_fact(db_mem, "fact-jwt", "auth uses JWT", "repo/auth", run_id, valid_from=t0)
    # Insert new fact valid from t1 first (to satisfy FK)
    _insert_fact(db_mem, "fact-sessions", "auth uses sessions", "repo/auth", run_id, valid_from=t1)
    db_mem.execute(
        "UPDATE facts SET valid_to = ?, superseded_by = 'fact-sessions' WHERE id = 'fact-jwt'",
        (t1.isoformat(),),
    )
    db_mem.commit()

    # AS OF t0 + 30min (between t0 and t1) — should see JWT fact
    as_of = (t0 + timedelta(minutes=30)).isoformat()
    rows_past = db_mem.execute(
        """SELECT id FROM facts
           WHERE scope = 'repo/auth'
             AND valid_from <= ?
             AND (valid_to IS NULL OR valid_to > ?)""",
        (as_of, as_of),
    ).fetchall()
    ids_past = [r["id"] for r in rows_past]
    assert "fact-jwt" in ids_past, "AS OF query must return fact that was valid at that time"
    assert "fact-sessions" not in ids_past, "Future fact must not appear in AS OF query"

    # AS OF t2 (now) — should see sessions fact only
    as_of_now = t2.isoformat()
    rows_now = db_mem.execute(
        """SELECT id FROM facts
           WHERE scope = 'repo/auth'
             AND valid_from <= ?
             AND (valid_to IS NULL OR valid_to > ?)""",
        (as_of_now, as_of_now),
    ).fetchall()
    ids_now = [r["id"] for r in rows_now]
    assert "fact-sessions" in ids_now
    assert "fact-jwt" not in ids_now


def test_supersession_update_query(db_mem):
    """
    §2.3 'WRITE: Supersede an old fact' — verifies that UPDATE sets valid_to and
    superseded_by, and that the old fact row is preserved (never deleted — §10.1).
    """
    run_id = _insert_run(db_mem)
    past = datetime.now(UTC) - timedelta(hours=1)
    now = datetime.now(UTC)

    _insert_fact(db_mem, "old", "old content", "repo", run_id, valid_from=past)
    _insert_fact(db_mem, "new", "new content", "repo", run_id, valid_from=now)

    # Apply supersession UPDATE (§2.3 pattern)
    db_mem.execute(
        "UPDATE facts SET valid_to = ?, superseded_by = 'new' WHERE id = 'old'",
        (now.isoformat(),),
    )
    db_mem.commit()

    old_row = db_mem.execute("SELECT * FROM facts WHERE id = 'old'").fetchone()

    # Old row must still exist (never deleted — §10.1)
    assert old_row is not None, "Superseded fact must NOT be deleted"
    # valid_to must be set
    assert old_row["valid_to"] is not None, "valid_to must be set on superseded fact"
    # superseded_by FK must point to new fact
    assert old_row["superseded_by"] == "new", "superseded_by must point to the replacement fact"

    # New fact must still be current
    new_row = db_mem.execute("SELECT * FROM facts WHERE id = 'new'").fetchone()
    assert new_row["valid_to"] is None
    assert new_row["superseded_by"] is None
