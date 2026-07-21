"""
swarm_memory/ingestion/writer.py

Write Pipeline + Supersession for SwarmMemory.
"""

import contextlib
import hashlib
import json
import logging
import sqlite3
from datetime import UTC, datetime

import swarm_memory.core.config as config
from swarm_memory.core.embeddings import EmbeddingModel
from swarm_memory.core.models import Fact, WriteResult, WriteStatus, uuid7, ConsolidationStatus
from swarm_memory.core.utils import timed
from swarm_memory.ingestion.supersession import ConsolidationEngine

logger = logging.getLogger(__name__)


class FactWriter:
    def __init__(
        self,
        db: sqlite3.Connection,
        embedder: EmbeddingModel,
        engine: ConsolidationEngine,
    ):
        self.db = db
        self.embedder = embedder
        self.engine = engine

    def _find_similar_valid_facts(
        self,
        embedding: bytes,
        scope: str,
        threshold: float = 0.75,
        limit: int = 5,
        exclude_ids: set[str] | None = None,
    ) -> list[Fact]:
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
              {exclude_clause}
            ORDER BY fv.distance
        """
        with timed("write.find_candidates"):
            try:
                cursor = self.db.execute(query_vec, params)
                results = cursor.fetchall()
            except sqlite3.OperationalError as e:
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
        if not valid_from:
            valid_from = datetime.now(UTC).isoformat()
        else:
            # Normalize to +00:00 form (SQLite stores whatever we give it;
            # keep it consistent so time-travel queries work reliably).
            valid_from = datetime.fromisoformat(valid_from.replace("Z", "+00:00")).isoformat()

        with timed("write.duplicate_check"):
            content_hash = hashlib.sha256(f"{content}|{scope}".encode()).hexdigest()
            existing = self.db.execute(
                "SELECT id FROM facts WHERE content_hash = ?", [content_hash]
            ).fetchone()
            if existing:
                existing_id = existing["id"]
                return WriteResult(
                    fact_ids=[existing_id], 
                    superseded_ids=[], 
                    status=WriteStatus.DUPLICATE,
                    message=f"Duplicate of existing fact {existing_id} (exact match). No new fact created."
                )

        hint_fact: Fact | None = None
        if supersedes_hint:
            row = self.db.execute(
                "SELECT * FROM facts WHERE id = ? AND valid_to IS NULL",
                [supersedes_hint],
            ).fetchone()
            if row:
                hint_fact = Fact(**dict(row))

        with timed("write.embed"):
            embedding = self.embedder.embed(content, prefix="search_document: ")

        candidates = self._find_similar_valid_facts(
            embedding,
            scope,
            threshold=0.75,
            limit=5,
            exclude_ids={supersedes_hint} if hint_fact else None,
        )

        with timed("write.consolidate_facts"):
            result = await self.engine.consolidate_facts(
                new_content=content,
                existing_facts=candidates,
            )

        status = result.status
        superseded_ids = result.superseded_ids
        merged_text = result.merged_text

        if status == ConsolidationStatus.DUPLICATE:
            existing_id = superseded_ids[0] if superseded_ids else None
            return WriteResult(
                fact_ids=[existing_id] if existing_id else [], 
                superseded_ids=[], 
                status=WriteStatus.DUPLICATE,
                message=f"Duplicate of existing fact {existing_id}. No new fact created."
            )

        # Map to WriteStatus for remaining logic
        write_status = WriteStatus.CREATED if status == ConsolidationStatus.INDEPENDENT else WriteStatus.CONSOLIDATED

        if hint_fact and hint_fact.id not in superseded_ids:
            superseded_ids.append(hint_fact.id)
            write_status = WriteStatus.CONSOLIDATED

        # Word-count based split.
        word_count = len(merged_text.split())
        splits = [merged_text]
        if word_count > config.FACT_WORD_THRESHOLD:
            logger.info(f"Fact exceeds {config.FACT_WORD_THRESHOLD} words, splitting...")
            splits = await self.engine.split_fact(merged_text)
            write_status = WriteStatus.SPLIT

        new_ids = self._execute_write_transaction(
            splits=splits,
            scope=scope,
            fact_type=fact_type,
            confidence=confidence,
            valid_from=valid_from,
            run_id=run_id,
            content=content,
            embedding=embedding,
            superseded_ids=superseded_ids,
        )

        if write_status == WriteStatus.CREATED:
            message = "Inserted as an independent new fact."
        elif write_status == WriteStatus.CONSOLIDATED:
            message = f"Inserted as a consolidated fact. Replaced {len(superseded_ids)} older overlapping fact(s)."
        elif write_status == WriteStatus.SPLIT:
            message = f"Fact exceeded length threshold and was split into {len(new_ids)} independent facts. Replaced {len(superseded_ids)} older overlapping fact(s)."
        else:
            message = f"Fact processed with status: {write_status}"

        return WriteResult(
            fact_ids=new_ids, 
            superseded_ids=superseded_ids, 
            status=write_status,
            message=message
        )

    def _execute_write_transaction(
        self,
        splits: list[str],
        scope: str,
        fact_type: str,
        confidence: float,
        valid_from: str,
        run_id: str,
        content: str,
        embedding: bytes,
        superseded_ids: list[str],
    ) -> list[str]:
        new_ids = []
        with timed("write.transaction"):
            self.db.execute("BEGIN")
            try:
                for split_content in splits:
                    new_id = str(uuid7())
                    split_hash = hashlib.sha256(f"{split_content}|{scope}".encode()).hexdigest()

                    self.db.execute(
                        """INSERT INTO facts
                           (id, content, fact_type, scope, confidence, valid_from, source_run_id, content_hash)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            new_id,
                            split_content,
                            fact_type,
                            scope,
                            confidence,
                            valid_from,
                            run_id,
                            split_hash,
                        ],
                    )

                    split_embedding = (
                        self.embedder.embed(split_content, prefix="search_document: ")
                        if split_content != content
                        else embedding
                    )
                    self.db.execute(
                        "INSERT INTO facts_vec (fact_id, embedding) VALUES (?, ?)",
                        [new_id, split_embedding],
                    )
                    self.db.execute(
                        "INSERT INTO facts_fts (fact_id, content, scope) VALUES (?, ?, ?)",
                        [new_id, split_content, scope],
                    )
                    new_ids.append(new_id)

                    for old_id in superseded_ids:
                        self.db.execute(
                            "INSERT OR IGNORE INTO fact_lineage (predecessor_id, successor_id) VALUES (?, ?)",
                            [old_id, new_id],
                        )
                        self.invalidate_fact(old_id, valid_to=valid_from)

                self.db.commit()
                logger.info(
                    f"Consolidation event {uuid7()} created {len(splits)} facts, superseded {len(superseded_ids)} facts"
                )
            except Exception as e:
                self.db.execute("ROLLBACK")
                logger.error(f"Transaction rolled back due to error: {e}")
                raise e
        return new_ids

    def invalidate_fact(self, fact_id: str, valid_to: str) -> int:
        cursor = self.db.execute(
            "UPDATE facts SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
            [valid_to, fact_id],
        )
        return cursor.rowcount

    def write_trajectory(self, trajectory_json: str, run_id: str) -> tuple[list[dict], list[dict]]:
        saved = []
        errors = []
        try:
            items = json.loads(trajectory_json)
            if not isinstance(items, list):
                raise ValueError("Trajectory is not a JSON list")

            for step in items:
                if not isinstance(step, dict):
                    continue

                content = json.dumps(step)
                try:
                    vec_bytes = self.embedder.embed(content)
                    traj_id = str(uuid7())

                    self.db.execute("BEGIN")
                    try:
                        self.db.execute(
                            "INSERT INTO trajectories (id, content, run_id) VALUES (?, ?, ?)",
                            [traj_id, content, run_id],
                        )
                        if vec_bytes:
                            with contextlib.suppress(sqlite3.OperationalError):
                                self.db.execute(
                                    "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
                                    [traj_id, vec_bytes],
                                )
                        self.db.commit()
                        saved.append(
                            {"content": content[:80], "fact_id": traj_id, "status": "created"}
                        )
                    except Exception as inner_e:
                        self.db.execute("ROLLBACK")
                        raise inner_e
                except Exception as e:
                    errors.append({"content": content[:80], "error": str(e)})
        except Exception as e:
            errors.append({"content": "Trajectory parsing failed", "error": str(e)})

        return saved, errors
