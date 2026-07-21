import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

# Provide a fallback if uuid7 is not installed/patched in standard library
try:
    from uuid7 import uuid7
except ImportError:
    # fallback for typing if package is missing during test init
    def uuid7():
        return uuid.uuid4()


from enum import StrEnum


class FactType(StrEnum):
    INSIGHT = "insight"
    CONVENTION = "convention"
    ARCHITECTURE = "architecture"
    GOTCHA = "gotcha"
    DEPENDENCY = "dependency"


class WriteStatus(StrEnum):
    CREATED = "created"
    DUPLICATE = "duplicate"
    CONSOLIDATED = "consolidated"
    SPLIT = "split"


class ConsolidationStatus(StrEnum):
    INDEPENDENT = "independent"
    DUPLICATE = "duplicate"
    CONSOLIDATED = "consolidated"


class ConsolidationResult(BaseModel):
    status: ConsolidationStatus
    superseded_ids: list[str] = Field(default_factory=list)
    merged_text: str


class SplitFactResult(BaseModel):
    facts: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid7()))
    content: str
    fact_type: FactType = FactType.INSIGHT
    scope: str
    confidence: float = 1.0
    valid_from: datetime
    valid_to: datetime | None = None
    superseded_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_run_id: str
    source_branch: str | None = None
    content_hash: str | None = None
    extraction_method: str = "llm_summary"


class Run(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid7()))
    agent_id: str
    repo: str
    branch: str | None = None
    summary: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    started_at: datetime
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SearchResult(BaseModel):
    fact: Fact
    relevance_score: float
    retrieval_method: str


class WriteResult(BaseModel):
    fact_ids: list[str]
    superseded_ids: list[str]
    status: WriteStatus
    message: str
