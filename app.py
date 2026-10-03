"""Streamlit front end for network-ai.

Each person you reach out to is an instance - saved until you delete it.
  Sidebar     your instances (newest first), + New, Me; under them, the open
              instance's knobs
  + New       search -> pick the person -> goal (-> status -> the role) ->
              suggestions -> knobs -> first draft
  Playground  the brief, draft versions, regenerate, refine, copy, delete
  Me          upload a resume or LinkedIn PDF -> review -> add stories over
              time (me_page.py); shared by every instance

Run:  uv run streamlit run app.py
"""

import os

import streamlit as st

from compose import compose
from knob_ui import goal_and_status, knob_panel, read_role
from knobs import CHANNELS, ROLE_STATUSES
from llm import describe
from me_page import me_page
from pipeline import find_candidates, seed_knobs, write_brief
from schemas import Me
from store import delete_instance, list_instances, load_instance, load_me, new_instance, save_instance, save_me
from suggest import suggest

st.set_page_config(page_title="network-ai", page_icon="🤝")

state = st.session_state
state.setdefault("view", "new")
state.setdefault("stage", "search")
state.setdefault("dirty", False)


def go(view: str) -> None:
    state.view = view
    state.inst = load_instance(view) if view not in ("new", "me") else None
    if view == "new":
        state.stage = "search"


def exa_mode() -> str | None:
    return "mock" if state.get("use_mock") else None


# Widget callbacks run before the script - save their edits before anything renders.
if state.dirty and state.view not in ("new", "me") and state.get("inst"):
    save_instance(state.inst)
state.dirty = False
if state.get("me_dirty"):
    save_me(state.me)
    state.me_dirty = False


# --- Sidebar -----------------------------------------------------------------------

with st.sidebar:
    st.title("network-ai 🤝")
    new_col, me_col = st.columns(2)
    new_col.button("+ New", on_click=go, args=("new",), width="stretch",
                   type="primary" if state.view == "new" else "secondary")
    me_col.button("Me", on_click=go, args=("me",), width="stretch",
                  type="primary" if state.view == "me" else "secondary")
    for saved in list_instances():
        st.button(saved.title, key=f"open:{saved.id}", on_click=go, args=(saved.id,), width="stretch",
                  type="primary" if state.view == saved.id else "tertiary")
    st.caption(f"Model: {describe()}")


# --- + New ---------------------------------------------------------------------------

def search_stage() -> None:
    has_key = bool(os.environ.get("EXA_API_KEY"))
    state.setdefault("use_mock", not has_key)
    # Kept outside the widget's own key: Streamlit drops a widget's state once
    # it stops rendering, and the playground still needs the mode.
    st.toggle("Use mock data (fictional profiles, no search credit)", key="mock_toggle",
              value=state.use_mock, disabled=not has_key,
              on_change=lambda: state.update(use_mock=state.mock_toggle))
    with st.form("lookup"):
        query = st.text_input("Who are you reaching out to?", label_visibility="collapsed",
                              placeholder="Name plus anything you know - e.g. Ashutosh Mishra, Tampa, Capgemini, Citi")
        if st.form_submit_button("Find", type="primary") and query.strip():
            with st.spinner("Searching..."):
                try:
                    state.candidates = find_candidates(query, mode=exa_mode())
                except Exception as exc:
                    st.error(f"Lookup failed: {exc}")

    candidates = state.get("candidates")
    if candidates == []:
        st.warning("No profiles found - try adding where they work or live.")
    if not candidates:
        return
    labels = []
    for c in candidates:
        role = c.current_role
        current = f"{role.title} @ {role.company}" if role and role.company else (c.headline or "")
        labels.append(f"**{c.name}** - {current} · {c.location or 'location unknown'}")
    choice = st.radio("Which one?", range(len(candidates)), format_func=lambda i: labels[i])
    st.caption(candidates[choice].url)
    if st.button("Reach out to this person", type="primary"):
        state.inst = new_instance(candidates[choice])
        state.stage = "goal"
        st.rerun()


def goal_stage(me: Me) -> None:
    inst = state.inst
    prefix = f"new:{inst.id}"
    goal_and_status(prefix, exa_mode(), job_button=False)
    if not inst.settings.get("goal"):
        return
    if not me.entries and not me.highlights:
        st.info("Tip: upload your resume on the Me page first - its bullets become credibility items.")
    if st.button("Next: suggest the rest", type="primary"):
        job_text = state.get(f"{prefix}:job", "").strip()
        wants_job = inst.settings.get("status") in ROLE_STATUSES and job_text
        if wants_job and (not inst.job or inst.job.input != job_text) and not read_role(job_text, exa_mode()):
            return
        try:
            with st.spinner("Reading their profile and suggesting knobs..."):
                inst.suggestions = suggest(inst, me)
                inst.brief = write_brief(inst.person)
        except Exception as exc:
            st.error(f"The model call failed: {exc}")
            return
        seed_knobs(inst, me)
        state.stage = "knobs"
        st.rerun()


def knobs_stage(me: Me) -> None:
    inst = state.inst
    knob_panel(me, f"new:{inst.id}", exa_mode())
    if st.button("Create and write the first draft", type="primary"):
        try:
            with st.spinner("Writing the first draft..."):
                inst.drafts.append(compose(inst, me))
        except Exception as exc:
            st.error(f"The model call failed: {exc}")
            return
        save_instance(inst)
        state.view = inst.id
        st.rerun()


def new_page(me: Me) -> None:
    st.header("New reach-out")
    inst = state.get("inst")
    if state.stage != "search" and inst:
        person = inst.person
        st.markdown(f"**{person.name}** - {person.headline or ''}")
        if st.button("← Pick someone else", type="tertiary"):
            state.stage, state.inst = "search", None
            st.rerun()
    if state.stage == "search" or not inst:
        search_stage()
    elif state.stage == "goal":
        goal_stage(me)
    else:
        knobs_stage(me)


# --- Playground --------------------------------------------------------------------

def generate(me: Me, refine: str = "", base=None) -> None:
    inst = state.inst
    try:
        with st.spinner("Writing..."):
            inst.drafts.append(compose(inst, me, refine=refine, base=base))
    except Exception as exc:
        st.error(f"The model call failed: {exc}")
        return
    save_instance(inst)
    state[f"{inst.id}:ver"] = len(inst.drafts) - 1
    st.rerun()


def playground(me: Me) -> None:
    inst = state.inst
    with st.sidebar:
        st.divider()
        knob_panel(me, inst.id, exa_mode())

    person = inst.person
    st.header(person.name)
    st.caption(" · ".join(p for p in (person.headline, person.url) if p))
    if inst.brief:
        with st.expander("Brief"):
            st.write(inst.brief.summary)
            for fact in inst.brief.facts:
                st.write(f"- {fact}")

    drafts = inst.drafts
    if drafts:
        vkey = f"{inst.id}:ver"
        index = min(state.get(vkey, len(drafts) - 1), len(drafts) - 1)
        prev_col, label_col, next_col = st.columns([1, 4, 1])
        if prev_col.button("◀", disabled=index == 0, width="stretch"):
            state[vkey] = index - 1
            st.rerun()
        label_col.markdown(f"<div style='text-align:center'>Version {index + 1} of {len(drafts)}</div>",
                           unsafe_allow_html=True)
        if next_col.button("▶", disabled=index == len(drafts) - 1, width="stretch"):
            state[vkey] = index + 1
            st.rerun()

        draft = drafts[index]
        channel = CHANNELS.get(draft.settings.get("channel", ""), CHANNELS["LinkedIn note"])
        if draft.subject:
            st.code(draft.subject, language=None, wrap_lines=True)
        st.code(draft.body, language=None, wrap_lines=True)
        st.caption(f"{len(draft.body)}/{channel.max_chars} characters · {draft.settings.get('goal', '')} · "
                   f"{channel.name} · {draft.model} · {draft.attempts} attempt(s)")
        if draft.refine:
            st.caption(f"Refined with: {draft.refine}")
        if draft.problems:
            st.warning("Checks still failing after retries - edit before sending:\n\n"
                       + "\n".join(f"- {p}" for p in draft.problems))
    else:
        draft = None
        st.info("No drafts yet.")

    regen_col, del_col = st.columns([3, 1])
    if regen_col.button("Regenerate with current knobs", type="primary"):
        generate(me)
    with del_col.popover("Delete"):
        st.write(f"Delete {person.name} and every draft?")
        if st.button("Delete for good", type="primary"):
            delete_instance(inst.id)
            go("new")
            st.rerun()

    if draft:
        with st.form("refine", clear_on_submit=True):
            refine = st.text_input("Refine this version", placeholder="e.g. shorter, and less formal")
            if st.form_submit_button("Refine") and refine.strip():
                generate(me, refine=refine, base=draft)


# --- Route ---------------------------------------------------------------------------

me = state.me if "me" in state else load_me()
if state.view == "me":
    me_page(bool(state.get("use_mock", not os.environ.get("EXA_API_KEY"))))
elif state.view == "new":
    new_page(me)
elif state.get("inst"):
    playground(me)
else:
    go("new")
    st.rerun()
