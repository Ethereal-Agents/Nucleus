import json
import logging

from pydantic import ValidationError

from swarm_memory.core.llm_service import LLMService
from swarm_memory.core.models import ConsolidationResult, ConsolidationStatus, Fact, SplitFactResult

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """
    Detects contradictions and consolidates overlapping facts within the same scope.
    Also handles splitting large facts into independent components.
    """

    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    async def consolidate_facts(
        self, new_content: str, existing_facts: list[Fact]
    ) -> ConsolidationResult:
        """
        Compare new_content against existing facts using LLM.
        Returns:
            ConsolidationResult containing status, superseded_ids, and merged_text
        """
        if not existing_facts:
            return ConsolidationResult(
                status=ConsolidationStatus.INDEPENDENT, superseded_ids=[], merged_text=new_content
            )

        system_prompt = """You are a strict knowledge consolidation engine.
You will be provided with a NEW FACT and a list of EXISTING FACTS (each with an ID).
Your task is to analyze the NEW FACT against the EXISTING FACTS and determine the precise relationship.

Classify the NEW FACT into EXACTLY ONE of the following categories:

1. INDEPENDENT
   - Definition: The NEW FACT discusses entirely distinct concepts from all EXISTING FACTS.
   - Action: It should be stored as a separate, new piece of knowledge.

2. DUPLICATE
   - Definition: The NEW FACT conveys the exact same semantic meaning as one of the EXISTING FACTS. There is no new information.
   - Action: It should be discarded in favor of the existing fact.

3. CONSOLIDATED (Overlaps / Refines / Contradicts)
   - Definition: The NEW FACT shares subject matter with one or more EXISTING FACTS. It might add details, correct previous information, or overlap significantly.
   - Action: You must merge the NEW FACT and all affected EXISTING FACTS into a single, comprehensive fact. Resolve contradictions by trusting the NEW FACT (it is more recent), but ensure no non-contradictory details from the EXISTING FACTS are lost.
   - Keyword Preservation: You MUST retain all specific terminology, technical jargon, proper nouns, error codes, and identifiers from the source texts. Do not generalize specific terms into broader categories.

You must output ONLY a valid JSON object in the following format:
{
  "status": "independent" | "duplicate" | "consolidated",
  "superseded_ids": [ ... ],
  "merged_text": "..."
}

CRITICAL RULES for JSON fields:
- If status is "independent":
    - `superseded_ids` MUST be an empty array [].
    - `merged_text` MUST be the exact text of the NEW FACT.
- If status is "duplicate":
    - `superseded_ids` MUST contain exactly one ID: the ID of the existing fact that it duplicates.
    - `merged_text` MUST be the exact text of the NEW FACT.
- If status is "consolidated":
    - `superseded_ids` MUST contain the IDs of ALL existing facts being replaced or merged.
    - `merged_text` MUST be the newly written comprehensive fact that flawlessly combines all valid information.
"""

        facts_json = [{"id": f.id, "content": f.content} for f in existing_facts]
        user_prompt = (
            f"NEW FACT: {new_content}\n\nEXISTING FACTS:\n{json.dumps(facts_json, indent=2)}"
        )

        try:
            data = await self.llm_service.generate_json_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            if not data:
                return ConsolidationResult(
                    status=ConsolidationStatus.INDEPENDENT,
                    superseded_ids=[],
                    merged_text=new_content,
                )

            try:
                result = ConsolidationResult.model_validate(data)
            except ValidationError as e:
                logger.error(f"Pydantic Validation Error during consolidation: {e} | Data: {data}")
                return ConsolidationResult(
                    status=ConsolidationStatus.INDEPENDENT,
                    superseded_ids=[],
                    merged_text=new_content,
                )

            # Ensure valid IDs
            valid_ids = {f.id for f in existing_facts}
            result.superseded_ids = [fid for fid in result.superseded_ids if fid in valid_ids]

            return result
        except Exception as e:
            logger.error(f"Error during consolidation: {e}")
            return ConsolidationResult(
                status=ConsolidationStatus.INDEPENDENT, superseded_ids=[], merged_text=new_content
            )

    async def split_fact(self, content: str) -> list[str]:
        """
        Splits a single large fact into multiple semantically independent facts.
        """
        system_prompt = """You are an expert at breaking down large technical texts into manageable, self-contained knowledge blocks.
Split the provided text into a list of self-contained facts, concepts, or procedures.

IMPORTANT : DO NOT OVER-FRAGMENT

CRITICAL INSTRUCTIONS FOR HYBRID SEARCH:
1. Contextual Depth (For Semantic Search): Do not over-fragment. Group closely related details together into a single, cohesive block. The chunk must provide meaningful context on its own.
2. Semantic Completeness: Resolve all pronouns and implicit references (e.g., replace "it" with the specific entity name). Each fact must stand completely alone.
3. Keyword Preservation (For Keyword Search): NEVER abstract or summarize away specific terminology, technical jargon, error codes, IDs, or acronyms. Retain the exact vocabulary used in the source text.

Return JSON only in this format:
{
  "facts": ["comprehensive fact 1", "comprehensive fact 2", "comprehensive fact 3"]
}
"""
        try:
            data = await self.llm_service.generate_json_async(
                system_prompt=system_prompt,
                user_prompt=content,
            )

            facts = [content]
            if data:
                try:
                    result = SplitFactResult.model_validate(data)
                    facts = result.facts or [content]
                except ValidationError as e:
                    logger.error(f"Pydantic Validation Error during splitting: {e} | Data: {data}")
            return facts
        except Exception as e:
            logger.error(f"Error splitting fact: {e}")
            return [content]
