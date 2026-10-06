"""Knob widgets, shared by the creation flow and the playground sidebar.

The Instance in st.session_state.inst is the source of truth. Every widget is
set from it just before rendering and writes back through on_change. That
way a change made in code - a goal switch moving ingredient levels, a new
Lead demoting the old one - shows in the widgets in the same run, and every
edit marks the instance dirty so app.py saves it.

Suggestions are marked ✦ and never forced: open suggestions start unselected,
settings take custom text, and every item list accepts your own entries.
"""

import streamlit as st

from knobs import (CHANNEL, CUSTOM_OK, GOAL, INGREDIENTS, LEVEL_LABELS, LEVELS, ROLE_STATUSES, STATUS,
                   THEIR_ROLE, TONE, Ingredient, Setting, rule_for)
from pipeline import (ingredient_options, knob_warnings, role_item, set_job, set_level, set_setting,
                      story_entry)
from schemas import IngredientState, Instance, Me
from sources.exa_client import PageUnreadable
from sources.job_posting import read_job
from store import save_me

SUGGESTED = " ✦"
JOB_PLACEHOLDER = "Paste the job link, or the description itself if the link is login-walled"


def _inst() -> Instance:
    return st.session_state.inst


def _touch() -> None:
    st.session_state.dirty = True


# --- Settings ------------------------------------------------------------------

def _set_setting(key: str, value: str) -> None:
    set_setting(_inst(), key, value)
    _touch()


def _on_pill(key: str, wkey: str) -> None:
    value = st.session_state[wkey]
    if value is not None:
        _set_setting(key, value)
    elif not st.session_state.get(f"{wkey}:custom", "").strip():
        _set_setting(key, "")


def _on_custom(key: str, wkey: str) -> None:
    value = st.session_state[wkey].strip()
    if value:
        _set_setting(key, value)


def setting_widget(setting: Setting, prefix: str) -> None:
    inst = _inst()
    value = inst.settings.get(setting.key, "")
    wkey = f"{prefix}:{setting.key}"
    st.session_state[wkey] = value if value in setting.options else None

    guess = inst.suggestions.role_guess if (setting.key == "their_role" and inst.suggestions) else None
    st.pills(setting.label, setting.options, key=wkey, on_change=_on_pill, args=(setting.key, wkey),
             format_func=lambda o: o + SUGGESTED if o == guess else o)
    if setting.key in CUSTOM_OK:
        ckey = f"{wkey}:custom"
        st.session_state[ckey] = "" if value in setting.options else value
        st.text_input(f"{setting.label} (custom)", key=ckey, on_change=_on_custom, args=(setting.key, ckey),
                      placeholder="...or type your own", label_visibility="collapsed")
    if guess:
        st.caption(f"✦ suggested from their profile: {inst.suggestions.role_reason}")


# --- The role --------------------------------------------------------------------

def read_role(value: str, mode: str | None) -> bool:
    """Read a job link or pasted JD into the instance; the role ingredient's
    item follows it. False (with an error shown) if the page couldn't be read."""
    inst = _inst()
    try:
        with st.spinner("Reading the job..."):
            job = read_job(value, mode=mode)
    except PageUnreadable as exc:
        st.error(str(exc))
        return False
    set_job(inst, job)
    _touch()
    return True


def job_widget(prefix: str, mode: str | None, button: bool = True) -> None:
    inst = _inst()
    wkey = f"{prefix}:job"
    if wkey not in st.session_state:
        st.session_state[wkey] = inst.job.input if inst.job else ""
    st.text_area("The role - job link or description", key=wkey, height=90, placeholder=JOB_PLACEHOLDER)
    if inst.job:
        st.caption(f"Read: {role_item(inst.job.posting)}")
    value = st.session_state[wkey].strip()
    if button and value and (not inst.job or inst.job.input != value):
        if st.button("Read job", key=f"{wkey}:read") and read_role(value, mode):
            st.rerun()


# --- Ingredients -------------------------------------------------------------------

def _on_level(key: str, wkey: str) -> None:
    set_level(_inst(), key, st.session_state[wkey])
    _touch()


def _on_items(key: str, wkey: str) -> None:
    _inst().ingredients[key].items = list(st.session_state[wkey])
    _touch()


def story_nudge(me: Me, prefix: str) -> None:
    """Ask for a story at the moment it pays off: a draft is about to use an
    entry that has none. One entry at a time, never blocking."""
    entry = story_entry(_inst(), me)
    if not entry:
        return
    with st.popover(f"✍️ Add the story behind {entry.label}?", width="stretch"):
        st.caption("The resume says what happened; the story says why it mattered. Drafts that use "
                   "this entry get it as background. Saved to your profile for every reach-out.")
        skey = f"{prefix}:story:{entry.key}"
        st.text_area("Story", key=skey, height=140, label_visibility="collapsed",
                     placeholder="The problem: ...\nWhat you built: ...\nThe hard decision: ...\nThe result: ...")
        if st.button("Save story", key=f"{skey}:save", type="primary") and st.session_state[skey].strip():
            entry.story = st.session_state[skey].strip()
            save_me(me)
            st.session_state.me = me
            st.rerun()


def ingredient_widget(ing: Ingredient, prefix: str, options: list[str], suggested: set[str],
                      me: Me | None = None) -> None:
    state = _inst().ingredients.setdefault(ing.key, IngredientState())
    ikey, lkey = f"{prefix}:{ing.key}:items", f"{prefix}:{ing.key}:level"
    st.session_state[ikey] = list(state.items)
    st.session_state[lkey] = state.level

    count = f" ({len(state.items)})" if state.items else ""
    with st.expander(f"{ing.label} · {LEVEL_LABELS[state.level]}{count}", expanded=state.level != "off"):
        st.multiselect(ing.label, list(dict.fromkeys([*state.items, *options])), key=ikey,
                       on_change=_on_items, args=(ing.key, ikey), accept_new_options=True,
                       placeholder=ing.placeholder, help=ing.help, label_visibility="collapsed",
                       format_func=lambda o: o + SUGGESTED if o in suggested else o)
        st.select_slider("Level", LEVELS, key=lkey, on_change=_on_level, args=(ing.key, lkey),
                         format_func=LEVEL_LABELS.get, label_visibility="collapsed")
        if state.level != "off" and not state.items and ing.key != "ask":
            st.caption("Nothing selected - the draft won't use this.")
        if ing.key == "credibility" and me:
            story_nudge(me, prefix)


def show_warnings(inst: Instance) -> None:
    for note in knob_warnings(inst):
        st.warning(note, icon="⚠️")


# --- The whole panel ---------------------------------------------------------------

def goal_and_status(prefix: str, mode: str | None, job_button: bool = True) -> None:
    """The first knobs: the goal decides whether status (and then the job) apply."""
    inst = _inst()
    setting_widget(GOAL, prefix)
    if not inst.settings.get("goal"):
        return
    if rule_for(inst.settings["goal"]).show_status:
        setting_widget(STATUS, prefix)
        if inst.settings.get("status") in ROLE_STATUSES:
            job_widget(prefix, mode, button=job_button)


def knob_panel(me: Me, prefix: str, mode: str | None) -> None:
    inst = _inst()
    goal_and_status(prefix, mode)
    for setting in (THEIR_ROLE, CHANNEL, TONE):
        setting_widget(setting, prefix)

    st.markdown("**What goes in**")
    st.caption("Off · Hint · Mention · Lead - only one thing leads. ✦ = suggested.")
    options, suggested = ingredient_options(inst, me)
    for ing in INGREDIENTS:
        ingredient_widget(ing, prefix, options.get(ing.key, []), suggested, me)
    show_warnings(inst)
