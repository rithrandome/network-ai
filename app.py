"""Streamlit front end for network-ai.

Each person you reach out to is an instance - saved until you delete it.
  /           + New and the playground; the sidebar lists your instances
              (newest first) and, under them, the open instance's knobs
  /profile    the person icon, top right
  + New       search -> pick the person -> goal (-> status -> the role) ->
              suggestions -> knobs -> first draft
  Playground  the brief, draft versions, regenerate, refine, copy, delete
  Profile     upload a resume or LinkedIn PDF -> review -> add stories over
              time (me_page.py); shared by every instance
  /settings   the gear icon, top right: which models read and write, and
              where your data goes

Run:  uv run streamlit run app.py
"""

import os

import streamlit as st

from compose import compose
from knob_ui import goal_and_status, knob_panel, read_role
from knobs import CHANNELS, ROLE_STATUSES
from llm import describe, is_local, vendor
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
    """Open + New or a saved instance - both live on the home page."""
    state.view = view
    state.inst = load_instance(view) if view != "new" else None
    if view == "new":
        state.stage = "search"
    state.go_home = True


def exa_mode() -> str | None:
    return "mock" if state.get("use_mock") else None


# Widget callbacks run before the script - save their edits before anything renders.
if state.dirty and state.view != "new" and state.get("inst"):
    save_instance(state.inst)
state.dirty = False
if state.get("me_dirty"):
    save_me(state.me)
    state.me_dirty = False


# --- Sidebar -----------------------------------------------------------------------

def sidebar(on_home: bool) -> None:
    """The open instance is highlighted only on the home page."""
    with st.sidebar:
        st.title("network-ai 🤝")
        st.button("+ New", on_click=go, args=("new",), width="stretch",
                  type="primary" if on_home and state.view == "new" else "secondary")
        for saved in list_instances():
            st.button(saved.title, key=f"open:{saved.id}", on_click=go, args=(saved.id,), width="stretch",
                      type="primary" if on_home and state.view == saved.id else "tertiary")


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
        st.info("Tip: upload your resume on your profile (top right) first - its bullets become credibility items.")
    if st.button("Next: suggest the rest", type="primary"):
        job_text = state.get(f"{prefix}:job", "").strip()
        wants_job = inst.settings.get("status") in ROLE_STATUSES and job_text
        if wants_job and (not inst.job or inst.job.input != job_text) and not read_role(job_text, exa_mode()):
            return
        with st.spinner("Reading their profile and suggesting knobs..."):
            # Both are suggestions, not requirements: if one fails, the knobs
            # still work with code-matched items and whatever the user picks.
            failed = []
            try:
                inst.suggestions = suggest(inst, me)
            except Exception as exc:
                failed.append(f"suggestions ({type(exc).__name__})")
            try:
                inst.brief = write_brief(inst.person)
            except Exception as exc:
                failed.append(f"profile summary ({type(exc).__name__})")
        if failed:
            state.suggest_warning = (f"Couldn't get {' or '.join(failed)} from {describe('extract')}. "
                                     "The knobs still work - pick what fits.")
        seed_knobs(inst, me)
        state.stage = "knobs"
        st.rerun()


def knobs_stage(me: Me) -> None:
    inst = state.inst
    if warning := state.pop("suggest_warning", None):
        st.warning(warning)
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


# --- Settings ------------------------------------------------------------------------

def settings_page() -> None:
    st.header("Settings")
    st.subheader("Models")
    for role, label, used_for in (
            ("extract", "Reading", "your resume, job postings, the person's profile, knob suggestions"),
            ("compose", "Writing", "every draft, including retries")):
        where = "runs on this machine" if is_local(role) else f"text is sent to {vendor(role)}'s API"
        st.markdown(f"**{label}:** {describe(role)}  \n{used_for} - {where}")
    st.caption("Set in `.env` - `EXTRACT_PROVIDER` / `COMPOSE_PROVIDER` (ollama, anthropic or openai) and the "
               "model names for each. Restart the app after editing it; `.env` is read at startup.")


# --- Route ---------------------------------------------------------------------------

def home() -> None:
    me = state.me if "me" in state else load_me()
    if state.view == "new":
        new_page(me)
    elif state.get("inst"):
        playground(me)
    else:
        go("new")
        st.rerun()


def profile() -> None:
    me_page(bool(state.get("use_mock", not os.environ.get("EXA_API_KEY"))))


HOME = st.Page(home, title="network-ai", icon="🤝", default=True)
PROFILE = st.Page(profile, title="Profile", icon=":material/account_circle:", url_path="profile")
SETTINGS = st.Page(settings_page, title="Settings", icon=":material/settings:", url_path="settings")
# Hidden: the sidebar is the instance list, and the icons below are the way to
# the other two pages.
page = st.navigation([HOME, PROFILE, SETTINGS], position="hidden")

# A sidebar click opens an instance, which lives on the home page.
if state.pop("go_home", False) and page.url_path != HOME.url_path:
    st.switch_page(HOME)

sidebar(on_home=page.url_path == HOME.url_path)
with st.container(horizontal=True, horizontal_alignment="right"):
    st.page_link(PROFILE, label="", icon=":material/account_circle:", help="Profile")
    st.page_link(SETTINGS, label="", icon=":material/settings:", help="Settings")

page.run()
