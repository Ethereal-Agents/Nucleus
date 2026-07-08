from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from swarm_memory.core.llm_service import LLMService
from swarm_memory.core.models import Fact, Relationship
from swarm_memory.ingestion.supersession import ContradictionDetector


@pytest.fixture
def mock_candidate():
    return Fact(
        id="fact-1",
        content="auth uses JWT",
        scope="auth",
        valid_from=datetime(2023, 1, 1, tzinfo=UTC),
        fact_type="insight",
        source_run_id="run_1",
        content_hash="hash",
    )


@pytest.mark.asyncio
async def test_detect_contradiction_success(mock_candidate):
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_json_async = AsyncMock(
        return_value={"relationship": "SUPERSEDES", "reason": "test"}
    )

    detector = ContradictionDetector(llm_service=mock_llm)
    result = await detector.detect_contradiction(
        mock_candidate, "auth uses session tokens", "auth", "insight"
    )
    assert result == Relationship.SUPERSEDES


@pytest.mark.asyncio
async def test_detect_contradiction_refines(mock_candidate):
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_json_async = AsyncMock(
        return_value={"relationship": "REFINES", "reason": "test"}
    )

    detector = ContradictionDetector(llm_service=mock_llm)
    result = await detector.detect_contradiction(
        mock_candidate, "auth uses session tokens", "auth", "insight"
    )
    assert result == Relationship.REFINES


@pytest.mark.asyncio
async def test_detect_contradiction_fallback_independent(mock_candidate):
    # LLMService returns empty dict on API error or JSON decode error
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_json_async = AsyncMock(return_value={})

    detector = ContradictionDetector(llm_service=mock_llm)
    result = await detector.detect_contradiction(
        mock_candidate, "auth uses session tokens", "auth", "insight"
    )
    assert result == Relationship.INDEPENDENT


@pytest.mark.asyncio
async def test_detect_contradiction_invalid_enum(mock_candidate):
    # LLMService returns a relationship that doesn't exist
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_json_async = AsyncMock(return_value={"relationship": "INVALID_RELATIONSHIP"})

    detector = ContradictionDetector(llm_service=mock_llm)
    result = await detector.detect_contradiction(
        mock_candidate, "auth uses session tokens", "auth", "insight"
    )
    assert result == Relationship.INDEPENDENT


@pytest.mark.asyncio
async def test_detect_contradictions_batch(mock_candidate):
    mock_llm = MagicMock(spec=LLMService)
    # mock it to return SUPERSEDES for the first, REFINES for the second
    mock_llm.generate_json_async = AsyncMock(
        side_effect=[
            {"relationship": "SUPERSEDES", "reason": "1"},
            {"relationship": "REFINES", "reason": "2"},
        ]
    )

    detector = ContradictionDetector(llm_service=mock_llm)
    candidate_2 = mock_candidate.model_copy(update={"id": "fact-2"})

    results = await detector.detect_contradictions(
        [mock_candidate, candidate_2], "auth uses session tokens", "auth", "insight"
    )

    assert len(results) == 2
    assert results[0][0] == mock_candidate
    assert results[0][1] == Relationship.SUPERSEDES
    assert results[1][0] == candidate_2
    assert results[1][1] == Relationship.REFINES
