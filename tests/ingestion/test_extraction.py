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


def test_parse_garbage_raises():
    raw = "I'm sorry, I cannot extract facts right now."
    import pytest
    with pytest.raises(ValueError):
        parse_extraction_output(raw, run_id="run-4")


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
    import pytest
    with pytest.raises(ValueError):
        parse_extraction_output(raw, run_id="run-10")

import json
import pytest
from swarm_memory.core.models import FactType
from swarm_memory.ingestion.extraction import parse_extraction_output

def test_parse_null_json_input():
    # EXT-01: None input -> ValueError or returns []
    with pytest.raises(Exception):
        parse_extraction_output(None, run_id="run-null")

def test_parse_deeply_nested_json():
    # EXT-02: JSON with extra nested keys -> only relevant fields extracted
    raw = json.dumps([
        {
            "content": "Nested fact",
            "scope": "nested",
            "fact_type": "insight",
            "extra_key": {"ignored": True}
        }
    ])
    drafts = parse_extraction_output(raw, run_id="run-nested")
    assert len(drafts) == 1
    assert drafts[0].content == "Nested fact"
    assert drafts[0].scope == "nested"
    assert drafts[0].fact_type == FactType.INSIGHT
    assert not hasattr(drafts[0], "extra_key")

def test_parse_unicode_content():
    # EXT-03: Facts with Unicode/emoji content are preserved
    raw = json.dumps([
        {
            "content": "Fact with emoji 🚀 and unicode 漢字",
            "scope": "emoji/scope"
        }
    ])
    drafts = parse_extraction_output(raw, run_id="run-unicode")
    assert len(drafts) == 1
    assert drafts[0].content == "Fact with emoji 🚀 and unicode 漢字"

def test_parse_very_large_array():
    # EXT-04: 100+ fact drafts parsed correctly
    items = [{"content": f"Fact {i}", "scope": "large"} for i in range(150)]
    raw = json.dumps(items)
    drafts = parse_extraction_output(raw, run_id="run-large")
    assert len(drafts) == 150
    assert drafts[0].content == "Fact 0"
    assert drafts[149].content == "Fact 149"

