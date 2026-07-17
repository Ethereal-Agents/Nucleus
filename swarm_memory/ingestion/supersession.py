"""
swarm_memory/ingestion/supersession.py

Temporal Supersession Engine for SwarmMemory.

This module provides LLM-based contradiction detection (§4.3 of the implementation plan).
When a new fact is ingested, it is compared against similar existing facts. The LLM
classifies the relationship as SUPERSEDES, REFINES, or INDEPENDENT. Only facts marked
as SUPERSEDES cause the existing fact to be invalidated (valid_to is set).
"""

import asyncio
import logging

from swarm_memory.core.llm_service import LLMService
from swarm_memory.core.models import Fact, Relationship
from swarm_memory.core.utils import timed

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Principal Engineer and codebase knowledge validator.
Your job is to determine whether a NEW fact about a codebase supersedes (contradicts or replaces) an EXISTING fact, refines it, or is entirely independent.

You must ALWAYS output your response as a valid JSON object matching this schema:
{
  "relationship": "SUPERSEDES" | "REFINES" | "INDEPENDENT",
  "reason": "A concise one-sentence explanation"
}

Definitions:
- SUPERSEDES: The new fact contradicts, deprecates, or replaces the existing fact (e.g., technology migration, API signature change, config change).
- REFINES: The new fact adds nuance, caveats, or detail without invalidating the existing fact.
- INDEPENDENT: The facts are unrelated or discuss different aspects of the codebase."""


USER_PROMPT_TEMPLATE = """Please evaluate the relationship between the following two facts:

EXISTING FACT:
  Content: "{existing_content}"
  Scope: {existing_scope}
  Valid since: {existing_valid_from}
  Type: {existing_fact_type}

NEW FACT:
  Content: "{new_content}"
  Scope: {new_scope}
  Type: {new_fact_type}"""


class ContradictionDetector:
    """
    Detects whether a newly extracted fact contradicts an existing fact using an LLM.

    This operates outside the database transaction to prevent holding locks during
    slow LLM inference steps. When multiple candidates are found, it evaluates
    them concurrently using asyncio.gather.

    Args:
        llm_service: An instance of LLMService for generating JSON responses.
    """

    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    async def detect_contradiction(
        self, candidate: Fact, new_content: str, new_scope: str, new_fact_type: str
    ) -> Relationship:
        """
        Evaluate a single candidate fact against the new fact.

        Args:
            candidate:     The existing Fact pulled from the database.
            new_content:   The text content of the new fact.
            new_scope:     The scope of the new fact.
            new_fact_type: The type of the new fact.

        Returns:
            Relationship Enum (SUPERSEDES, REFINES, INDEPENDENT).
        """
        valid_from_str = candidate.valid_from.isoformat() if candidate.valid_from else "Unknown"
        prompt = USER_PROMPT_TEMPLATE.format(
            existing_content=candidate.content,
            existing_scope=candidate.scope,
            existing_valid_from=valid_from_str,
            existing_fact_type=candidate.fact_type,
            new_content=new_content,
            new_scope=new_scope,
            new_fact_type=new_fact_type,
        )

        with timed("supersession.detect"):
            data = await self.llm_service.generate_json_async(
                system_prompt=SYSTEM_PROMPT, user_prompt=prompt
            )

        if data is None:
            logger.warning(
                "LLM call failed for contradiction check against fact %s, skipping", candidate.id
            )
            return Relationship.INDEPENDENT

        rel_str = data.get("relationship", "INDEPENDENT").upper()
        if rel_str in Relationship.__members__:
            return Relationship[rel_str]
        return Relationship.INDEPENDENT

    async def detect_contradictions(
        self, candidates: list[Fact], new_content: str, new_scope: str, new_fact_type: str
    ) -> list[tuple[Fact, Relationship]]:
        """
        Evaluate multiple candidate facts concurrently against the new fact.

        Args:
            candidates:    List of existing Facts to evaluate.
            new_content:   The text content of the new fact.
            new_scope:     The scope of the new fact.
            new_fact_type: The type of the new fact.

        Returns:
            List of tuples pairing each candidate with its determined Relationship.
        """
        tasks = [
            self.detect_contradiction(candidate, new_content, new_scope, new_fact_type)
            for candidate in candidates
        ]
        results = await asyncio.gather(*tasks)
        return list(zip(candidates, results, strict=False))
