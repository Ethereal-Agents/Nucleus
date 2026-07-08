"""
swarm_memory/server/presentation.py

Formatting and presentation logic for SwarmMemory results.
"""

from swarm_memory.core.models import FactType, SearchResult


def format_results_for_agent(results: list[SearchResult], scope: str) -> str:
    """
    Format search results as a human/LLM-readable context block.

    Agents are LLMs. Raw JSON with RRF scores is noise to them. This function
    returns a structured plain-text block that agents can inject directly into
    their reasoning context.

    Key formatting rules (§14.2):
    - Gotchas get a ⚠ prefix and always appear first (see apply_gotcha_priority).
    - fact_id and known-since date are included so agents can call
      memory_invalidate() without a second lookup.
    - Numeric scores (RRF, confidence, distance) are stripped — meaningless to LLMs.
    - Empty results still return a message (never silently empty).

    Args:
        results: Ordered list of SearchResult objects to format.
        scope:   The scope that was queried (used in the header).

    Returns:
        A formatted multi-line string, ready to inject into agent context.
    """
    if not results:
        return f"[MEMORY HUB — no relevant facts found for scope: {scope}]"

    lines = [f"[MEMORY HUB — {len(results)} fact(s) for {scope}]", ""]

    for result in results:
        fact = result.fact
        prefix = "⚠ " if fact.fact_type == FactType.GOTCHA else ""
        lines.append(f"{prefix}[{fact.fact_type.value}] {fact.scope}")
        lines.append(fact.content)
        # known_since is the valid_from date (when the fact became true)
        known_since = fact.valid_from.strftime("%Y-%m-%d") if fact.valid_from else "unknown"
        lines.append(f"→ id: {fact.id}  |  known since: {known_since}")
        lines.append("")

    lines.append('If any fact above is outdated, call: memory_invalidate(fact_id, reason="...")')
    return "\n".join(lines)
