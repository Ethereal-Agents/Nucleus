import json

from swarm_memory.core.models import FactType
from swarm_memory.ingestion.extraction import parse_extraction_output


def test_parse_clean_json():
    raw = json.dumps(
        [
            {
                "content": "Auth uses JWT.",
                "scope": "src/auth",
                "fact_type": "architecture",
                "supersedes_hint": "fact-123",
            }
        ]
    )
    drafts = parse_extraction_output(raw, run_id="run-1")
    assert len(drafts) == 1
    assert drafts[0].content == "Auth uses JWT."
    assert drafts[0].scope == "src/auth"
    assert drafts[0].fact_type == FactType.ARCHITECTURE
    assert drafts[0].supersedes_hint == "fact-123"


def test_parse_fenced_json():
    raw = """```json
[
  {
    "content": "We use pytest for testing.",
    "scope": "tests",
    "fact_type": "convention"
  }
]
```"""
    drafts = parse_extraction_output(raw, run_id="run-2")
    assert len(drafts) == 1
    assert drafts[0].content == "We use pytest for testing."
    assert drafts[0].scope == "tests"


def test_parse_trailing_comma():
    # Invalid JSON due to trailing comma in dict and array
    raw = """
    [
      {
        "content": "Trailing commas are bad",
        "scope": "global",
      },
    ]
    """
    drafts = parse_extraction_output(raw, run_id="run-3")
    assert len(drafts) == 1
    assert drafts[0].content == "Trailing commas are bad"
    assert drafts[0].scope == "global"


def test_parse_garbage_returns_empty():
    raw = "I'm sorry, I cannot extract facts right now."
    drafts = parse_extraction_output(raw, run_id="run-4")
    assert len(drafts) == 0


def test_parse_empty_string():
    drafts = parse_extraction_output("", run_id="run-5")
    assert len(drafts) == 0

    drafts_whitespace = parse_extraction_output("   \n  ", run_id="run-5")
    assert len(drafts_whitespace) == 0


def test_parse_empty_array():
    drafts = parse_extraction_output("[]", run_id="run-6")
    assert len(drafts) == 0


def test_parse_partial_valid():
    raw = json.dumps(
        [
            {
                "content": "Valid fact",
                "scope": "valid/scope",
            },
            {
                # Missing 'scope', which is required by FactDraft
                "content": "Invalid fact missing scope",
            },
            "Not even a dictionary",
        ]
    )
    drafts = parse_extraction_output(raw, run_id="run-7")
    assert len(drafts) == 1
    assert drafts[0].content == "Valid fact"


def test_fact_type_defaults_to_insight():
    raw = json.dumps(
        [
            {
                "content": "Fact with no type",
                "scope": "my/scope",
            }
        ]
    )
    drafts = parse_extraction_output(raw, run_id="run-8")
    assert len(drafts) == 1
    assert drafts[0].fact_type == FactType.INSIGHT


def test_supersedes_hint_preserved():
    raw = json.dumps(
        [
            {
                "content": "Hint fact",
                "scope": "hint/scope",
                "supersedes_hint": "fact-abc-def",
            }
        ]
    )
    drafts = parse_extraction_output(raw, run_id="run-9")
    assert len(drafts) == 1
    assert drafts[0].supersedes_hint == "fact-abc-def"


def test_parse_not_an_array():
    raw = json.dumps(
        {
            "content": "I returned a dict instead of a list of dicts",
            "scope": "wrong",
        }
    )
    drafts = parse_extraction_output(raw, run_id="run-10")
    assert len(drafts) == 0
