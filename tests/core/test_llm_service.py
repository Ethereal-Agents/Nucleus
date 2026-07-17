import os
from unittest import mock

import pytest

from swarm_memory.core.llm_service import LLMService


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


@pytest.mark.asyncio
@mock.patch("litellm.acompletion")
async def test_generate_json_returns_dict(mock_acompletion):
    mock_acompletion.return_value = MockResponse('{"key": "value"}')
    service = LLMService()
    result = await service.generate_json_async("system", "user")
    assert result == {"key": "value"}


@pytest.mark.asyncio
@mock.patch("litellm.acompletion")
async def test_generate_json_handles_markdown_fences(mock_acompletion):
    mock_acompletion.return_value = MockResponse('```json\n{"key": "value"}\n```')
    service = LLMService()
    result = await service.generate_json_async("system", "user")
    assert result == {"key": "value"}


@pytest.mark.asyncio
@mock.patch("litellm.acompletion")
async def test_generate_json_returns_none_on_empty_response(mock_acompletion):
    mock_acompletion.return_value = MockResponse("")
    service = LLMService()
    result = await service.generate_json_async("system", "user")
    assert result is None


@pytest.mark.asyncio
@mock.patch("litellm.acompletion")
async def test_generate_json_returns_none_on_invalid_json(mock_acompletion):
    mock_acompletion.return_value = MockResponse("This is not json")
    service = LLMService()
    result = await service.generate_json_async("system", "user")
    assert result is None


@pytest.mark.asyncio
@mock.patch("litellm.acompletion")
async def test_generate_json_returns_none_on_api_error(mock_acompletion):
    mock_acompletion.side_effect = Exception("API Error")
    service = LLMService()
    result = await service.generate_json_async("system", "user")
    assert result is None


@pytest.mark.asyncio
@mock.patch("litellm.acompletion")
async def test_generate_json_uses_correct_model(mock_acompletion):
    mock_acompletion.return_value = MockResponse("{}")
    service = LLMService(model_name="my-test-model")
    await service.generate_json_async("system", "user")

    mock_acompletion.assert_called_once()
    assert mock_acompletion.call_args.kwargs["model"] == "my-test-model"


@pytest.mark.asyncio
@mock.patch("litellm.acompletion")
async def test_generate_json_json_mode_enabled(mock_acompletion):
    mock_acompletion.return_value = MockResponse("{}")
    service = LLMService()
    await service.generate_json_async("system", "user")

    mock_acompletion.assert_called_once()
    assert mock_acompletion.call_args.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
@mock.patch("litellm.acompletion")
async def test_generate_json_temperature_zero(mock_acompletion):
    mock_acompletion.return_value = MockResponse("{}")
    service = LLMService()
    await service.generate_json_async("system", "user")

    mock_acompletion.assert_called_once()
    assert mock_acompletion.call_args.kwargs["temperature"] == 0.0
    assert mock_acompletion.call_args.kwargs["max_tokens"] == 200


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="No API key")
async def test_generate_json_real_api():
    # Use a lightweight/fast model for the test
    service = LLMService(model_name="openrouter/google/gemini-2.5-flash")
    result = await service.generate_json_async(
        system_prompt="Return a JSON object with a single key 'status' set to 'ok'.",
        user_prompt="{}",
    )
    assert result is not None
    assert "status" in result
    assert result["status"] == "ok"
