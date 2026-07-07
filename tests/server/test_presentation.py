from datetime import UTC, datetime

from swarm_memory.core.models import Fact, FactType, SearchResult
from swarm_memory.server.presentation import format_results_for_agent


class TestFormatResultsForAgent:
    def _make_result(self, fact_id: str, fact_type: FactType, content: str) -> SearchResult:
        fact = Fact(
            id=fact_id,
            content=content,
            fact_type=fact_type,
            scope="test-repo/src",
            valid_from=datetime.now(UTC),
            source_run_id="run-1",
            content_hash="hash",
        )
        return SearchResult(fact=fact, relevance_score=0.8, retrieval_method="hybrid")

    def test_empty_results_returns_no_facts_message(self):
        output = format_results_for_agent([], scope="myrepo/src")
        assert "no relevant facts found" in output.lower()
        assert "myrepo/src" in output

    def test_header_contains_count_and_scope(self):
        results = [self._make_result("A", FactType.INSIGHT, "test fact")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "1 fact(s)" in output
        assert "myrepo" in output

    def test_gotcha_has_warning_prefix(self):
        results = [self._make_result("G", FactType.GOTCHA, "do not do this")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "⚠" in output

    def test_non_gotcha_has_no_warning_prefix(self):
        results = [self._make_result("A", FactType.INSIGHT, "some insight")]
        output = format_results_for_agent(results, scope="myrepo")
        lines = output.split("\n")
        fact_line = next(line for line in lines if "insight" in line)
        assert "⚠" not in fact_line

    def test_contains_fact_id_and_known_since(self):
        results = [self._make_result("fact-123", FactType.INSIGHT, "test")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "fact-123" in output
        assert "known since" in output

    def test_no_raw_scores_in_output(self):
        results = [self._make_result("A", FactType.INSIGHT, "test")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "0.8" not in output
        assert "relevance" not in output.lower()

    def test_invalidate_reminder_at_end(self):
        results = [self._make_result("A", FactType.INSIGHT, "test")]
        output = format_results_for_agent(results, scope="myrepo")
        assert "memory_invalidate" in output
