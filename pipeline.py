"""
network-ai pipeline: describe a person -> pick the right candidate -> brief.

  find_candidates(query)
      free-text description -> Exa people search -> candidate profiles (the
      user picks one; same-name matches are normal)
  write_brief(candidate)
      full profile markdown -> local Ollama model -> Brief (summary, facts,
      icebreakers), schema-constrained so the UI can render each part

Runs against a local Ollama model (default: gemma3:1b) - set OLLAMA_MODEL in
.env to point at a different one.

Run:  EXA_MODE=mock uv run python pipeline.py
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from schemas import Brief, Candidate
from sources.exa_client import find_person

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")

# Ollama's default context window is 2048 tokens and it truncates silently -
# a full profile runs 3-4k tokens, so without this the model never sees
# most of the Experience section.
CONTEXT_TOKENS = 8192

BRIEF_TEMPLATE = """
Below is a person's professional profile. Produce:
1. A short, plain-language summary of who they are
2. Two specific, interesting facts about them
3. Two conversation openers that reference something specific in their profile

Use only the information in the profile - do not invent facts that aren't there.

Profile:
{profile}
"""


def find_candidates(query: str, mode: str | None = None) -> list[Candidate]:
    """query is free text, e.g. "Ashutosh Mishra, Tampa, Capgemini, Citi".
    mode overrides EXA_MODE: live | offline | mock."""
    return find_person(query, mode=mode)


def write_brief(candidate: Candidate) -> Brief:
    prompt = PromptTemplate(input_variables=["profile"], template=BRIEF_TEMPLATE)
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=CONTEXT_TOKENS)
    # json_schema = Ollama's constrained decoding: the schema is enforced at
    # generation time. Format instructions in the prompt aren't enough for a
    # 1B model - it echoes the schema's shape instead of filling it in.
    chain = prompt | llm.with_structured_output(Brief, method="json_schema")
    return chain.invoke({"profile": candidate.profile_text})


if __name__ == "__main__":
    candidates = find_candidates("Jane Doe, Northwind Analytics")
    chosen = candidates[0]
    print(f"Chose: {chosen.name} - {chosen.headline}\n")
    print(write_brief(chosen).model_dump_json(indent=2))
