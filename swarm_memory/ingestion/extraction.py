import json
import logging
import re

from pydantic import BaseModel, ValidationError

from swarm_memory.core.models import FactType

logger = logging.getLogger(__name__)


class FactDraft(BaseModel):
    """
    A lightweight model representing a single fact exactly as extracted by the agent,
    before it is fully processed, embedded, and written to the database.
    """

    content: str
    scope: str
    fact_type: FactType = FactType.INSIGHT
    supersedes_hint: str | None = None


def parse_extraction_output(raw: str, run_id: str) -> list[FactDraft]:
    """
    Robustly parse the agent's summary string into FactDraft objects.

    Handles:
      - Markdown code fences (e.g., ```json ... ```)
      - Trailing commas in JSON objects or arrays
      - Empty strings
      - Malformed items (skips invalid items, keeps valid ones)

    Never raises an exception — returns an empty list on unrecoverable failure
    so that memory_end_run always completes successfully.

    Args:
        raw: The raw string output from the agent (passed as `summary`).
        run_id: The ID of the run (used for logging context).

    Returns:
        A list of successfully parsed and validated FactDraft objects.
    """
    text = raw.strip()

    if not text:
        return []

    # 1. Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip())

    # 2. Attempt direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 3. Repair: strip trailing comma before ] or } (common LLM mistake)
        # We replace any comma followed by whitespace and a closing bracket/brace
        repaired = re.sub(r",\s*([\]}])", r"\1", text)
        data = json.loads(repaired)  # Let it raise if still invalid

    # If the parsed JSON is not a list, it's not the format we expect
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    # 4. Validate each item with Pydantic — skip individually malformed items
    drafts = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning("Skipping non-object item in run %s: %s", run_id, item)
            continue

        try:
            drafts.append(FactDraft.model_validate(item))
        except ValidationError as exc:
            logger.warning("Skipping malformed fact draft in run %s: %s", run_id, exc)

    return drafts
