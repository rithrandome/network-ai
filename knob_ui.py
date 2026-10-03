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

from knobs import (CHANNEL, CHANNELS, GOAL, INGREDIENTS, LEVEL_LABELS, LEVELS, ROLE_STATUSES, STATUS,
                   THEIR_ROLE, TONE, Ingredient, Setting, rule_for, warnings_for)
from pipeline import change_goal, role_item
from schemas import IngredientState, Instance, Me
from sources.exa_client import PageUnreadable
from sources.job_posting import read_job

CUSTOM_OK = {"goal", "their_role", "tone"}
SUGGESTED = " ✦"
JOB_PLACEHOLDER = "Paste the job link, or the description itself if the link is login-walled"


def _inst() -> Instance:
    return st.session_state.inst


def _touch() -> None:
    st.session_state.dirty = True


# --- Settings ------------------------------------------------------------------

def _set_setting(key: str, value: str) -> None:
    inst = _inst()
    if key == "goal":
        change_goal(inst, value)
    else:
        inst.settings[key] = value
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
    old = role_item(inst.job.posting) if inst.job else None
    inst.job = job
    role = inst.ingredients.get("role")
    if role is not None:
        role.items = [role_item(job.posting), *(i for i in role.items if i != old)]
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

def ingredient_options(inst: Instance, me: Me) -> tuple[dict[str, list[str]], set[str]]:
    """Pickable items per ingredient, and which of them are model suggestions."""
    s, brief = inst.suggestions, inst.brief
    suggested = {
        "ask": s.ask_wording if s else [],
        "common_ground": s.common_ground if s else [],
        # The suggestion filter often leaves no hooks; the brief's facts are
        # grounded in the profile too.
        "hook": [*(s.hooks if s else []), *(brief.facts if brief else [])],
    }
    options = {**suggested,
               "credibility": list(me.highlights),
               "role": [role_item(inst.job.posting)] if inst.job else []}
    return options, {item for items in suggested.values() for item in items}


def _on_level(key: str, wkey: str) -> None:
    inst, level = _inst(), st.session_state[wkey]
    if level == "lead":
        for other, state in inst.ingredients.items():
            if other != key and state.level == "lead":
                state.level = "mention"
    inst.ingredients[key].level = level
    _touch()


def _on_items(key: str, wkey: str) -> None:
    _inst().ingredients[key].items = list(st.session_state[wkey])
    _touch()


def ingredient_widget(ing: Ingredient, prefix: str, options: list[str], suggested: set[str]) -> None:
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


def knob_warnings(inst: Instance) -> None:
    notes = warnings_for(inst.settings)
    channel = inst.settings.get("channel")
    if channel == "LinkedIn note":
        loud = [k for k, s in inst.ingredients.items() if s.level in ("mention", "lead") and s.items]
        if len(loud) > 2:
            notes.append(f"A {CHANNELS[channel].max_chars}-character note fits about two things at "
                         f"Mention or Lead - you have {len(loud)}.")
    for note in notes:
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
        ingredient_widget(ing, prefix, options.get(ing.key, []), suggested)
    knob_warnings(inst)
