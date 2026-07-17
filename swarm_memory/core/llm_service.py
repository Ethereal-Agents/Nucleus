"""
swarm_memory/core/llm_service.py

Service wrapper for interacting with Language Models via LiteLLM.
"""

import json
import logging
import re
from typing import Any

import litellm

from swarm_memory.core import config
from swarm_memory.core.utils import timed

logger = logging.getLogger(__name__)


class LLMService:
    """
    Provides a standardized interface for interacting with LLMs.
    Configured to use LiteLLM for broad provider support (e.g., OpenRouter).
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.LLM_MODEL

    async def generate_json_async(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Calls the LLM with JSON mode enabled and parses the result.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.

        Returns:
            A parsed JSON dictionary.
        """
        with timed("llm.generate_json"):
            try:
                response = await litellm.acompletion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=200,
                    temperature=0.0,
                )

                content = response.choices[0].message.content or ""
                logger.debug(f"LLM Raw Response: {content}")

                # Clean markdown formatting if model didn't strictly follow JSON mode
                text = content.strip()
                if not text:
                    logger.error("LLM returned an empty response. Cannot parse JSON.")
                    return None
                    
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text.strip())

                return json.loads(text)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON output: {e} | Content: {content}")
                return None
            except Exception as e:
                logger.error(f"LLM API Error: {e}")
                return None
