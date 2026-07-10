import sqlite3

from swarm_memory.core.config import DB_PATH

_SCHEMA = """
-- Core tables
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    repo            TEXT NOT NULL,
    branch          TEXT,
    summary         TEXT,
    model           TEXT,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    total_cost_usd  REAL DEFAULT 0.0,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS facts (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    fact_type       TEXT NOT NULL DEFAULT 'insight',
    scope           TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 1.0,

    valid_from      TEXT NOT NULL,
    valid_to        TEXT,
    superseded_by   TEXT,

    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    source_run_id   TEXT NOT NULL,
    source_branch   TEXT,
    extraction_method TEXT DEFAULT 'llm_summary',
    content_hash    TEXT,

    FOREIGN KEY (superseded_by) REFERENCES facts(id),
    FOREIGN KEY (source_run_id) REFERENCES runs(id),
    UNIQUE(content_hash)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_facts_current
    ON facts(scope, valid_to)
    WHERE valid_to IS NULL AND superseded_by IS NULL;

CREATE INDEX IF NOT EXISTS idx_facts_valid_range
    ON facts(scope, valid_from, valid_to);

CREATE INDEX IF NOT EXISTS idx_facts_superseded_by
    ON facts(superseded_by)
    WHERE superseded_by IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_facts_source_run
    ON facts(source_run_id);

CREATE INDEX IF NOT EXISTS idx_facts_type
    ON facts(fact_type, scope);

-- Full-text search (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    fact_id UNINDEXED,
    content,
    scope UNINDEXED,
    tokenize='trigram'
);
"""

# Vector search (sqlite-vec) is created separately so we can handle environments
# where sqlite-vec might not be installed yet during tests.
_VEC_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(
    fact_id TEXT PRIMARY KEY,
    embedding float[{dim}]
);
"""


def get_db(path: str | None = None) -> sqlite3.Connection:
    if path is None:
        path = DB_PATH

    conn = sqlite3.connect(
        path, isolation_level=None, check_same_thread=False
    )  # Auto-commit mode for setup, we can use transactions manually
    conn.row_factory = sqlite3.Row

    # Performance-critical PRAGMAs
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec extension
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        vec_loaded = True
    except ImportError:
        print("Warning: sqlite_vec not found. Vector search will be disabled.")
        vec_loaded = False

    return conn, vec_loaded


def init_db(conn: sqlite3.Connection, vec_loaded: bool = False, embed_dim: int = 768):
    """Initialize the database schema."""
    conn.executescript(_SCHEMA)
    if vec_loaded:
        conn.executescript(_VEC_SCHEMA.format(dim=embed_dim))


def get_initialized_db(path: str | None = None) -> sqlite3.Connection:
    from swarm_memory.core.config import EMBED_DIM

    conn, vec_loaded = get_db(path)
    init_db(conn, vec_loaded, EMBED_DIM)
    return conn
