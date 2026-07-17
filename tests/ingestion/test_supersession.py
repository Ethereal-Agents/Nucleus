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


@pytest.mark.asyncio
async def test_detect_contradiction_llm_exception(mock_candidate):
    # If the LLM call raises an exception, it should propagate (or handle gracefully if we decide).
    # Current behavior is exception propagates since there's no try/except.
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_json_async = AsyncMock(side_effect=Exception("API Timeout"))

    detector = ContradictionDetector(llm_service=mock_llm)
    with pytest.raises(Exception, match="API Timeout"):
        await detector.detect_contradiction(
            mock_candidate, "auth uses session tokens", "auth", "insight"
        )


@pytest.mark.asyncio
async def test_detect_contradictions_empty_list():
    mock_llm = MagicMock(spec=LLMService)
    detector = ContradictionDetector(llm_service=mock_llm)

    results = await detector.detect_contradictions([], "content", "scope", "insight")
    assert len(results) == 0
    # generate_json_async should not have been called
    mock_llm.generate_json_async.assert_not_called()


@pytest.mark.asyncio
async def test_prompt_contains_fact_content(mock_candidate):
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_json_async = AsyncMock(return_value={"relationship": "INDEPENDENT"})

    detector = ContradictionDetector(llm_service=mock_llm)
    await detector.detect_contradiction(mock_candidate, "new session auth", "auth", "insight")

    # Check that prompt contains the right fields
    call_kwargs = mock_llm.generate_json_async.call_args.kwargs
    prompt = call_kwargs["user_prompt"]
    assert "auth uses JWT" in prompt
    assert "new session auth" in prompt
    assert "auth" in prompt  # scope

@pytest.mark.asyncio
async def test_detect_contradiction_concurrent_batch_ordering(mock_candidate):
    # SUP-01: Multiple candidates — results maintain correct pairing with input facts
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_json_async = AsyncMock(
        side_effect=[
            {"relationship": "SUPERSEDES", "reason": "1"},
            {"relationship": "REFINES", "reason": "2"},
            {"relationship": "INDEPENDENT", "reason": "3"}
        ]
    )
    detector = ContradictionDetector(llm_service=mock_llm)
    c1 = mock_candidate.model_copy(update={"id": "fact-1"})
    c2 = mock_candidate.model_copy(update={"id": "fact-2"})
    c3 = mock_candidate.model_copy(update={"id": "fact-3"})
    
    results = await detector.detect_contradictions(
        [c1, c2, c3], "new text", "auth", "insight"
    )
    
    assert len(results) == 3
    assert results[0][0] == c1
    assert results[0][1] == Relationship.SUPERSEDES
    assert results[1][0] == c2
    assert results[1][1] == Relationship.REFINES
    assert results[2][0] == c3
    assert results[2][1] == Relationship.INDEPENDENT

@pytest.mark.asyncio
async def test_detect_contradiction_with_refines_relationship(mock_candidate):
    # SUP-02: LLM returns REFINES — verify it's correctly classified
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_json_async = AsyncMock(
        return_value={"relationship": "REFINES", "reason": "Refines the existing fact."}
    )
    detector = ContradictionDetector(llm_service=mock_llm)
    
    result = await detector.detect_contradiction(
        mock_candidate, "new refined info", "auth", "insight"
    )
    
    assert result == Relationship.REFINES

