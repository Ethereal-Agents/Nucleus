"""
swarm_memory/ingestion/writer.py

Write Pipeline + Supersession for SwarmMemory.

Implements the full write path that powers the memory_write tool call:

  Agent write
    ├─ 1. Content Hash Check       → prevents duplicate facts from re-extraction
    ├─ 2. embed_fact               → float32 bytes for the new fact
    ├─ 3. _find_similar_valid_facts→ fetch top-5 candidates in exact scope
    ├─ 4. detect_contradictions    → concurrently call LLM to evaluate candidates
    └─ 5. Atomic Transaction:
            ├─ INSERT new fact, embedding, and FTS entry
            └─ For each SUPERSEDES:
                 ├─ UPDATE old fact (set valid_to, superseded_by)
                 └─ DELETE old fact from FTS index

See implementation_plan.md §4, §7 for the full design rationale.
"""

import contextlib
import hashlib
import logging
import sqlite3
from datetime import UTC, datetime

from swarm_memory.core.embeddings import EmbeddingModel
from swarm_memory.core.models import Fact, FactType, Relationship, WriteResult
from swarm_memory.core.utils import timed
from swarm_memory.ingestion.supersession import ContradictionDetector

logger = logging.getLogger(__name__)


class FactWriter:
    """
    FactWriter orchestrates the ingestion of new facts into the shared memory hub.

    It ensures that writes are idempotent and that contradictory older facts are
    superseded correctly within atomic transactions.
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        embedder: EmbeddingModel,
        detector: ContradictionDetector,
    ):
        self.db = db
        self.embedder = embedder
        self.detector = detector

    def _find_similar_valid_facts(
        self,
        embedding: bytes,
        scope: str,
        threshold: float = 0.75,
        limit: int = 5,
        exclude_ids: set[str] | None = None,
    ) -> list[Fact]:
        """
        Find existing facts in the same scope that are semantically similar.

        Uses sqlite-vec to find facts with cosine similarity >= threshold.
        These serve as candidates for the contradiction detection LLM.
        """
        # Convert cosine similarity threshold to Euclidean distance (L2) threshold.
        # For normalized vectors (L2 norm = 1), distance^2 = 2 - 2 * cosine_similarity.
        max_distance = (2.0 - 2.0 * threshold) ** 0.5
        exclude_clause = ""
        params = [embedding, limit, scope]
        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            exclude_clause = f" AND f.id NOT IN ({placeholders}) "
            params.extend(list(exclude_ids))

        query_vec = f"""
            SELECT fv.fact_id, fv.distance
            FROM facts_vec fv
            JOIN facts f ON fv.fact_id = f.id
            WHERE fv.embedding MATCH ?
              AND k = ?
              AND f.scope = ?
              AND f.valid_to IS NULL
              AND f.superseded_by IS NULL
              {exclude_clause}
            ORDER BY fv.distance
        """
        with timed("write.find_candidates"):
            try:
                cursor = self.db.execute(query_vec, params)
                results = cursor.fetchall()
            except sqlite3.OperationalError as e:
                # Fallback if sqlite-vec MATCH is not available or virtual table is mocked
                logger.debug(f"sqlite-vec not available or missing facts_vec table: {e}")
                return []

            facts = []
        for row in results:
            if row["distance"] <= max_distance:
                fact_row = self.db.execute(
                    "SELECT * FROM facts WHERE id = ?", [row["fact_id"]]
                ).fetchone()
                if fact_row:
                    facts.append(Fact(**dict(fact_row)))
        return facts

    async def write_fact(
        self,
        content: str,
        scope: str,
        run_id: str,
        valid_from: str | None = None,
        fact_type: str = "insight",
        confidence: float = 1.0,
        supersedes_hint: str | None = None,
    ) -> WriteResult:
        """
        Write a new fact to the shared memory hub.

        Automatically detects and supersedes contradicting older facts.
        Idempotent: writing the same content+scope twice returns the existing fact without re-embedding.

        Args:
            content:    The natural language fact or insight.
            scope:      The exact scope path (e.g., 'myrepo/src/auth').
            run_id:     The session ID of the agent writing this fact (provenance).
            valid_from: Optional ISO-8601 timestamp. Defaults to now UTC.
            fact_type:  Type of fact ('insight', 'gotcha', 'convention', etc.).
            confidence: Float 0.0 - 1.0 representing certainty. Defaults to 1.0.
            supersedes_hint: Optional fact_id that this fact replaces (skips LLM check for that fact).

        Returns:
            WriteResult containing the new fact_id, a list of any superseded_ids, and a status string.
        """
        if not valid_from:
            valid_from = datetime.now(UTC).isoformat()
        else:
            # Normalize to valid format if provided (e.g., timezone aware)
            # Ensure it's parseable
            with contextlib.suppress(ValueError):
                datetime.fromisoformat(valid_from.replace("Z", "+00:00"))

        with timed("write.duplicate_check"):
            content_hash = hashlib.sha256(f"{content}|{scope}".encode()).hexdigest()

            existing = self.db.execute(
                "SELECT id FROM facts WHERE content_hash = ?", [content_hash]
            ).fetchone()

            if existing:
                logger.info("Fact is duplicate of existing ID: %s", existing["id"])
                return WriteResult(fact_id=existing["id"], superseded_ids=[], status="duplicate")

        hint_fact: Fact | None = None
        if supersedes_hint:
            row = self.db.execute(
                "SELECT * FROM facts WHERE id = ? AND valid_to IS NULL AND superseded_by IS NULL",
                [supersedes_hint],
            ).fetchone()
            if row:
                hint_fact = Fact(**dict(row))
            else:
                logger.warning(
                    "supersedes_hint %s is invalid or already superseded; ignoring.",
                    supersedes_hint,
                )

        with timed("write.embed"):
            embedding = self.embedder.embed(content, prefix="search_document: ")

        candidates = self._find_similar_valid_facts(
            embedding,
            scope,
            threshold=0.75,
            limit=5,
            exclude_ids={supersedes_hint} if hint_fact else None,
        )

        with timed("write.detect_contradictions"):
            relationships = await self.detector.detect_contradictions(
                candidates=candidates,
                new_content=content,
                new_scope=scope,
                new_fact_type=fact_type,
            )

        if hint_fact:
            relationships.append((hint_fact, Relationship.SUPERSEDES))

        # create new Fact locally, then write
        new_fact = Fact(
            content=content,
            fact_type=FactType(fact_type),
            scope=scope,
            confidence=confidence,
            valid_from=datetime.fromisoformat(valid_from.replace("Z", "+00:00")),
            source_run_id=run_id,
            content_hash=content_hash,
        )

        superseded_ids = self._commit_fact_transaction(
            new_fact=new_fact,
            embedding=embedding,
            relationships=relationships,
            valid_from=valid_from,
        )

        return WriteResult(fact_id=new_fact.id, superseded_ids=superseded_ids, status="created")

    def invalidate_fact(self, fact_id: str, valid_to: str, superseded_by: str | None = None) -> None:
        """
        Invalidates a fact by setting its valid_to timestamp and optionally its superseded_by FK.
        Also removes it from the FTS index so it no longer appears in keyword searches.
        """
        self.db.execute(
            "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
            [valid_to, superseded_by, fact_id],
        )
        self.db.execute("DELETE FROM facts_fts WHERE fact_id = ?", [fact_id])


    def _commit_fact_transaction(
        self,
        new_fact: Fact,
        embedding: bytes,
        relationships: list[tuple[Fact, Relationship]],
        valid_from: str,
    ) -> list[str]:
        """
        Executes the atomic database transaction to insert a new fact and invalidate any superseded facts.
        """
        superseded_ids = []

        with timed("write.transaction"):
            self.db.execute("BEGIN")
            try:
                self.db.execute(
                    """INSERT INTO facts
                       (id, content, fact_type, scope, confidence, valid_from, created_at, source_run_id, content_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        new_fact.id,
                        new_fact.content,
                        new_fact.fact_type,
                        new_fact.scope,
                        new_fact.confidence,
                        new_fact.valid_from.isoformat(),
                        new_fact.created_at.isoformat(),
                        new_fact.source_run_id,
                        new_fact.content_hash,
                    ],
                )

                self.db.execute(
                    "INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)",
                    [new_fact.id, embedding],
                )
                self.db.execute(
                    "INSERT INTO facts_fts (fact_id, content, scope) VALUES (?, ?, ?)",
                    [new_fact.id, new_fact.content, new_fact.scope],
                )

                for candidate, relationship in relationships:
                    if relationship == Relationship.SUPERSEDES:
                        self.invalidate_fact(candidate.id, valid_from, new_fact.id)
                        superseded_ids.append(candidate.id)
                self.db.commit()
                logger.info(
                    "Fact %s created, superseded %d facts", new_fact.id, len(superseded_ids)
                )
            except Exception as e:
                self.db.execute("ROLLBACK")
                logger.error("Transaction rolled back due to error: %s", e)
                raise e

        return superseded_ids
