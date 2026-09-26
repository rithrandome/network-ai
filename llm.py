"""Chat model selection: local Ollama (default) or Claude.

Pipeline steps ask for a model by role, not by name:
  extract - reading profiles, briefs, context suggestions (cheap, structured)
  compose - writing outreach drafts (the product - quality matters most)

LLM_PROVIDER in .env picks the backend:
  ollama    - OLLAMA_MODEL for both roles; free, runs locally. (default)
  anthropic - EXTRACT_MODEL (Haiku) / COMPOSE_MODEL (Sonnet); needs
              ANTHROPIC_API_KEY with credit on the account.

structured() is the one way steps get typed output. method="json_schema" on
both backends means constrained decoding - Ollama's format parameter, Claude's
native structured outputs - so the schema is enforced while generating rather
than requested in the prompt.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

load_dotenv()

Role = Literal["extract", "compose"]

CLAUDE_MODELS: dict[Role, str] = {
    "extract": os.environ.get("EXTRACT_MODEL", "claude-haiku-4-5-20251001"),
    "compose": os.environ.get("COMPOSE_MODEL", "claude-sonnet-5"),
}

# Ollama's default context window is 2048 tokens and it truncates silently -
# a full profile runs 3-4k tokens, so without this the model never sees
# most of the Experience section.
OLLAMA_CONTEXT_TOKENS = 8192


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "ollama")


def get_model(role: Role) -> BaseChatModel:
    if provider() == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set in .env")
        # No temperature: Sonnet 5 rejects sampling parameters with a 400.
        return ChatAnthropic(model=CLAUDE_MODELS[role])

    from langchain_ollama import ChatOllama

    return ChatOllama(model=os.environ.get("OLLAMA_MODEL", "gemma3:1b"),
                      temperature=0, num_ctx=OLLAMA_CONTEXT_TOKENS)


def structured(role: Role, schema: type[BaseModel]) -> Runnable:
    return get_model(role).with_structured_output(schema, method="json_schema")


def describe() -> str:
    """Which model is answering, for display."""
    if provider() == "anthropic":
        return f"Claude ({CLAUDE_MODELS['extract']} / {CLAUDE_MODELS['compose']})"
    return f"Ollama ({os.environ.get('OLLAMA_MODEL', 'gemma3:1b')}, local)"
