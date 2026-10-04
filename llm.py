"""Chat model selection: local Ollama (default), Claude, or OpenAI GPT.

Pipeline steps ask for a model by role, not by name:
  extract - reading resumes and profiles, briefs, suggestions (cheap, structured)
  compose - writing outreach drafts (the product - quality matters most)

Each role picks its own provider, so writing can use a strong cloud model
while reading your resume stays local and free:
  EXTRACT_PROVIDER / COMPOSE_PROVIDER   ollama | anthropic | openai
  LLM_PROVIDER                          the default for both (default: ollama)

  ollama    - OLLAMA_MODEL; free, runs locally
  anthropic - CLAUDE_EXTRACT_MODEL / CLAUDE_COMPOSE_MODEL; needs ANTHROPIC_API_KEY
  openai    - OPENAI_EXTRACT_MODEL / OPENAI_COMPOSE_MODEL; needs OPENAI_API_KEY

structured() is the one way steps get typed output. method="json_schema" on
every backend means constrained decoding - Ollama's format parameter, Claude's
structured outputs, OpenAI's strict structured outputs (the SDK marks every
field required, defaults included) - so the schema is enforced while
generating rather than requested in the prompt.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

load_dotenv()

Role = Literal["extract", "compose"]
Provider = Literal["ollama", "anthropic", "openai"]
PROVIDERS = ("ollama", "anthropic", "openai")
VENDOR = {"anthropic": "Anthropic", "openai": "OpenAI"}

# EXTRACT_MODEL / COMPOSE_MODEL are the older names for the Claude models.
CLAUDE_MODELS: dict[Role, str] = {
    "extract": os.environ.get("CLAUDE_EXTRACT_MODEL") or os.environ.get("EXTRACT_MODEL") or "claude-haiku-4-5-20251001",
    "compose": os.environ.get("CLAUDE_COMPOSE_MODEL") or os.environ.get("COMPOSE_MODEL") or "claude-sonnet-5",
}
OPENAI_MODELS: dict[Role, str] = {
    "extract": os.environ.get("OPENAI_EXTRACT_MODEL") or "gpt-5-mini",
    "compose": os.environ.get("OPENAI_COMPOSE_MODEL") or "gpt-5",
}

# Ollama's default context window is 2048 tokens and it truncates silently -
# a full profile runs 3-4k tokens, so without this the model never sees
# most of the Experience section.
OLLAMA_CONTEXT_TOKENS = 8192


def provider(role: Role = "extract") -> str:
    name = (os.environ.get(f"{role.upper()}_PROVIDER") or os.environ.get("LLM_PROVIDER") or "ollama").lower()
    if name not in PROVIDERS:
        raise RuntimeError(f"Unknown provider {name!r} for {role} - use one of {', '.join(PROVIDERS)}")
    return name


def _model_name(role: Role) -> str:
    return {"anthropic": CLAUDE_MODELS[role], "openai": OPENAI_MODELS[role]}.get(
        provider(role), os.environ.get("OLLAMA_MODEL", "gemma3:1b"))


def _require_key(name: str, role: Role) -> None:
    if not os.environ.get(name):
        raise RuntimeError(f"The {role} step uses {provider(role)} but {name} is not set in .env")


def get_model(role: Role) -> BaseChatModel:
    name = provider(role)
    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        _require_key("ANTHROPIC_API_KEY", role)
        # No temperature: Sonnet 5 rejects sampling parameters with a 400.
        return ChatAnthropic(model=CLAUDE_MODELS[role])

    if name == "openai":
        from langchain_openai import ChatOpenAI

        _require_key("OPENAI_API_KEY", role)
        model = OPENAI_MODELS[role]
        # GPT-5-family models are reasoning models: they reject temperature, and
        # their default effort spends seconds thinking about a 300-character note.
        effort = os.environ.get("OPENAI_REASONING_EFFORT", "low")
        reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
        return ChatOpenAI(model=model, **({"reasoning_effort": effort} if reasoning else {"temperature": 0}))

    from langchain_ollama import ChatOllama

    return ChatOllama(model=os.environ.get("OLLAMA_MODEL", "gemma3:1b"),
                      temperature=0, num_ctx=OLLAMA_CONTEXT_TOKENS)


def structured(role: Role, schema: type[BaseModel]) -> Runnable:
    return get_model(role).with_structured_output(schema, method="json_schema")


def is_local(role: Role) -> bool:
    return provider(role) == "ollama"


def vendor(role: Role) -> str:
    return VENDOR.get(provider(role), "this machine")


def describe(role: Role | None = None) -> str:
    """Which model is answering, for display: one role, or both."""
    if role:
        label = {"ollama": "Ollama", "anthropic": "Claude", "openai": "OpenAI"}[provider(role)]
        return f"{label} ({_model_name(role)}{', local' if is_local(role) else ''})"
    extract, compose = describe("extract"), describe("compose")
    return extract if extract == compose else f"reading: {extract} · writing: {compose}"
