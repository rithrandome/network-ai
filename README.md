# network-ai

A networking assistant for job seekers. Describe someone you want to reach out
to in one line - a name plus whatever you know ("Ashutosh Mishra, Tampa,
Capgemini, Citi") - pick the right profile from the candidates, say what the reach-out is for,
and get a draft message built from the context you chose - then keep tweaking
it in a playground that's saved until you delete it.

You start by uploading your resume or LinkedIn PDF; it becomes your profile,
and its bullets become what drafts can cite about you.

Each person is an **instance**, like a chat session: the person, a brief, the
knobs (goal, who they are to you, channel, tone, what goes in), and every draft
version. A tracker comes next.

## Pipeline

```
once: resume / LinkedIn PDF -> sections split in code -> one small extraction per
      section -> bullets verified against the file -> review screen -> profile
      (stories added later, when a draft is about to use an entry)

one free-text description
  --> Exa people search        -> up to 10 candidate profiles (full profile markdown
                                  + structured work history), via exa-py
  --> you pick the right one    (same-name matches are normal)
  --> goal first                 (decides whether status and the job apply)
  --> job link or pasted JD     -> Exa contents fetch -> JobPosting (extract model)
  --> suggest (one model call)  -> role guess, common ground, hooks, asks;
                                   filtered by code, then knobs seeded from the goal
  --> you adjust the knobs      settings + ingredients at Off/Hint/Mention/Lead
  --> plan -> compose -> check  numbered facts + a LEAD/MENTION/HINT plan ->
                                   DraftOutput{subject, body, used} -> validators ->
                                   retry with the problems (at most 2)
  --> playground                versions, regenerate, refine, copy - saved in SQLite
```

## Why this, not that

**Exa people search, not a search engine + a profile scraper.** One call
returns the candidate profiles themselves - headline, About, full experience -
as markdown, plus a structured entity. That replaces two vendors (one to find
the URL, one to fetch the profile), and scraping LinkedIn directly is exactly
what got profile-scraping APIs shut down. Picking from candidates instead of
taking the first hit fixes the common-name problem.

**One search box, not name/company/location fields.** Exa's search is
semantic, so the description goes in as the query as-is; splitting it into
fields first would add nothing.

**People search with full text, not Exa's default query + highlights.**
`category="people"` limits results to profiles - without it a name query
returns any page mentioning the name. Full `text` because the brief needs the
whole profile, and highlights are only query-relevant excerpts; measured, text
adds nothing to the $0.007 search cost.

**Search, not Exa's Agent API.** Resolving a known person is one call, and the
user needs to see the candidates to pick the right one. An agent run is slower,
costs more, and picks for you. It fits a later "find people matching criteria"
feature (list-building), not this.

**The profile text, not the structured entity, feeds the brief.** The entity
is incomplete - exa-py keeps only name, location and work history, dropping the
education the API returns - so it's only used for the one-line candidate label.

**Cache-first Exa client.** Every response is cached by request, so the same
lookup never spends credit twice. `EXA_MODE=offline` guarantees no spend while
developing; `EXA_MODE=mock` uses a fictional fixture with no key at all.

**Models by role, local by default.** Steps ask `llm.py` for an `extract` or
`compose` model, not a model name. `LLM_PROVIDER=ollama` (default) runs
everything on a local model at no cost; `anthropic` uses Haiku for extraction
and Sonnet for writing drafts, where quality is the product. Both backends use
constrained decoding (`method="json_schema"`: Ollama's format parameter,
Claude's native structured outputs), so the schema is enforced while
generating - a 1B model given format instructions in the prompt echoes the
schema instead of filling it in. No temperature is passed to Claude: Sonnet 5
rejects sampling parameters. Ollama's `num_ctx` is raised to 8192 because its
default 2048-token window silently truncates a full profile.

**Knobs as structured choices, not an "add context" box.** Context boxes go
unused, and output quality depends on context. So every knob is a pick from a
short list with custom input always available. The goal comes first and
narrows the rest (`knobs.GOAL_RULES`): it decides whether your application
status matters, the default channel, and the starting level of each ingredient.

**Ingredients with a level, not on/off.** Each piece of content - your
credibility, the role, the ask, common ground, a hook, how you found them,
something only you know - is Off, Hint (alluded to), Mention (stated) or Lead
(the opening is built on it; only one leads). Implicit vs explicit is a real
choice in outreach: common ground often works better hinted at than announced.

**Upload first, depth later.** A setup form asking for your whole career up
front is where people quit, so the profile is built from a file you already
have and is useful immediately. Depth comes later and is optional: each
job or project has a "story" (the problem, the decision, the result - what a
resume has no room for). The app asks for a story at the moment it pays off:
when a draft's credibility item comes from an entry without one. A strength
meter starts around 60% after an upload and names the next story worth
adding. The frameworks behind this: time-to-value, progressive profiling, Fogg's
B=MAP (ask when motivation is highest), and endowed progress.

**Code reads the structure; the model fills small schemas.** A 1B model asked
for a whole resume as JSON drops jobs. So code splits the document by its
headings, the model extracts one section at a time against a small schema with a
worked example, and code checks the result: every bullet must be found in the
file (fuzzy match), or it's flagged for you to keep or drop. If some source
lines aren't covered by anything extracted, that region is read again starting
at the missed item (at most twice). A title that isn't in the file (a bullet
put in the title slot) is repaired from its company's line. A certification
name that isn't in the file is dropped, not flagged - a made-up certification
is worse than a missing one, and the Me page has a table for adding any. A re-upload
replaces what came from the file and keeps your stories and edits.

**The profile reaches drafts as facts, not as a resume.** Your bullets are
ranked by word overlap with the job's requirements (weighted double) and the
recipient's profile, and the top one is preselected. A picked bullet goes to
the compose model with where it's from and, if you wrote one, its story as
background - "say why it matters, in a phrase, never retell it". Shared
employers, schools and city are matched in code and listed ahead of the
model's common ground.

**Suggestions that don't push.** One model call after the goal guesses who the
person is to you and offers common ground, hooks and ask wordings, all marked
✦. Open suggestions start unselected. The raw output isn't trusted: on a 1B
model, "common ground" came back as the JD's requirements, so code keeps only
common ground found in both people's text and hooks found in their profile.

**A rendered plan, not a template with blanks.** The system prompt is fixed:
what the levels mean and the rules. The knobs become the user message, built
in code: settings, the selected items as numbered facts (F1..Fn), and a plan
saying which facts lead, get mentioned or get hinted. Off items are never sent,
so they can't leak. The compose model never sees the raw profile, resume or JD
requirements - given requirements, a small model wrote them up as the sender's
own experience, and no check could catch it because the words were in the
prompt.

**Code checks, then retry - never truncate.** The output schema is narrowed
per message (fact IDs limited to the plan's, subject required on email), so
constrained decoding enforces what it can. Code checks the rest: length per
channel, the lead fact in the opening, mentions actually stated, off items not
leaking, and no numbers or names that weren't in the prompt. Problems go back
to the model as a retry turn. A draft still failing after two retries is kept
with its problems shown - a truncated message reads worse than a long one.
Whether a hint was subtle or the tone landed is a judgment call, left to an
LLM judge in evaluation rather than the live loop.

**SQLite, one JSON row per instance.** Local, single-user, and the models are
still changing - a new field needs no migration. It holds real people's data,
so it lives in the gitignored `.data/`.

On gemma3:1b the whole loop runs, but drafts often still fail a check (it
won't reliably state the role it was told to lead with); the checks report it
rather than hide it. Drafting is the step a bigger model is for:
`LLM_PROVIDER=anthropic` or a larger `OLLAMA_MODEL`.

## Run it

Requires [Ollama](https://ollama.com) running locally:

```bash
ollama pull gemma3:1b
uv sync
cp .env.example .env        # add EXA_API_KEY to look up real people

EXA_MODE=mock uv run python pipeline.py      # CLI brief, fictional data, no keys
EXA_MODE=mock uv run python compose.py       # CLI draft: prints the rendered plan and the checked draft
uv run streamlit run app.py                  # app; live search if EXA_API_KEY is set, else mock
uv run python -m sources.exa_client "Name, company, city"   # candidate lookup only
```

## Files

- `app.py` - Streamlit: instance sidebar, + New flow, playground
- `me_page.py` - upload -> review -> deepen; strength meter
- `knob_ui.py` - the knob widgets, shared by the New flow and the playground sidebar
- `knobs.py` - knob registry: settings, ingredients, levels, channels, goal rules
- `pipeline.py` - `find_candidates()`, `write_brief()`, knob seeding and goal changes
- `suggest.py` - the suggestions call and the code filter on its output
- `profile_match.py` - credibility ranking and code-matched common ground
- `prompts.py` - fixed system prompt, `MessagePlan`, `render()`
- `compose.py` - knobs -> plan -> draft, with the validate/retry loop
- `validators.py` - the deterministic checks
- `store.py` - SQLite persistence (instances, your profile)
- `llm.py` - model per role, Ollama or Claude, structured output
- `sources/exa_client.py` - Exa people search and page fetch (exa-py), cache/offline/mock modes
- `sources/job_posting.py` - job link or pasted JD -> `JobPosting`
- `sources/resume.py` - resume / LinkedIn PDF, DOCX or text -> `Me` (`uv run python -m sources.resume file.pdf`)
- `schemas.py` - the data models
- `mock_data/` - fictional Exa response (two Jane Does), a fictional job posting, and a
  fictional resume and LinkedIn profile (Alex Tester) - "Use the sample resume" on the Me page
