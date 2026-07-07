import time

import pytest

from swarm_memory.server.session import SessionManager


@pytest.fixture
def session_manager():
    return SessionManager()


class TestSessionDedup:
    def test_end_session_removes_run_id(self, session_manager):
        session_manager._session_seen["run-abc"] = ({"fact-1"}, time.time())
        session_manager.end_session("run-abc")
        assert "run-abc" not in session_manager._session_seen

    def test_end_session_is_idempotent(self, session_manager):
        session_manager.end_session("run-does-not-exist")  # must not raise

    def test_evict_stale_sessions_removes_old_entries(self, session_manager):
        old_timestamp = time.time() - 99999
        session_manager._session_seen["stale-run"] = ({"fact-1"}, old_timestamp)
        session_manager._session_seen["fresh-run"] = ({"fact-2"}, time.time())

        session_manager._evict_stale_sessions()

        assert "stale-run" not in session_manager._session_seen
        assert "fresh-run" in session_manager._session_seen

    def test_evict_stale_sessions_keeps_recent_entries(self, session_manager):
        session_manager._session_seen["recent-run"] = ({"fact-1"}, time.time())
        session_manager._evict_stale_sessions()
        assert "recent-run" in session_manager._session_seen

    def test_get_seen_ids_and_mark_seen(self, session_manager):
        run_id = "test-run"

        # Initially empty
        assert session_manager.get_seen_ids(run_id) == set()

        # Mark some facts
        session_manager.mark_seen(run_id, ["fact-1", "fact-2"])
        assert session_manager.get_seen_ids(run_id) == {"fact-1", "fact-2"}

        # Mark more facts
        session_manager.mark_seen(run_id, ["fact-2", "fact-3"])
        assert session_manager.get_seen_ids(run_id) == {"fact-1", "fact-2", "fact-3"}
