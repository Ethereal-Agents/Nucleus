"""
swarm_memory/server/session.py

Session deduplication manager for SwarmMemory.

This module provides a SessionManager class that tracks which facts an agent has
already seen during its current run. This prevents the same facts (especially gotchas)
from cluttering the context window repeatedly across multiple memory queries.
"""

import logging
import time

from swarm_memory.core import config

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages deduplication state for agent sessions.

    Agents query multiple times per run. Without dedup, the same fact
    would appear in every response, wasting tokens and LLM attention.
    This tracks which fact IDs have been shown for a given run_id.
    """

    def __init__(self):
        # Structure: run_id -> (set of fact IDs already shown, last-access unix timestamp)
        self._session_seen: dict[str, tuple[set[str], float]] = {}

    def _evict_stale_sessions(self) -> None:
        """Remove sessions that have been idle for longer than SESSION_TTL_SECONDS."""
        cutoff = time.time() - config.SESSION_TTL_SECONDS
        stale_ids = [
            run_id
            for run_id, (_seen_ids, last_access) in self._session_seen.items()
            if last_access < cutoff
        ]
        for run_id in stale_ids:
            del self._session_seen[run_id]
            logger.debug("Evicted stale session: %s", run_id)

    def end_session(self, run_id: str) -> None:
        """Explicitly evict a session from the deduplication store."""
        self._session_seen.pop(run_id, None)
        logger.debug("Session ended and evicted: %s", run_id)

    def get_seen_ids(self, run_id: str) -> set[str]:
        """Get the set of fact IDs the agent has already seen this session."""
        self._evict_stale_sessions()
        seen_ids, _ = self._session_seen.get(run_id, (set(), 0.0))
        return seen_ids

    def mark_seen(self, run_id: str, fact_ids: list[str]) -> None:
        """Mark a list of fact IDs as seen by the agent, updating the session TTL."""
        if not fact_ids:
            return

        seen_ids, _ = self._session_seen.get(run_id, (set(), 0.0))
        new_seen = seen_ids | set(fact_ids)
        self._session_seen[run_id] = (new_seen, time.time())
