"""Instance routes: one person you're reaching out to, from creation to drafts.

The flow a front end follows, mirroring the Streamlit app:
  POST   /instances                 the picked candidate -> a saved instance
  PATCH  /instances/{id}/settings   goal first (it narrows the rest), then any knob
  PUT    /instances/{id}/job        a job link or pasted JD, when status calls for one
  POST   /instances/{id}/suggest    suggestions + brief, then knobs seeded from the goal
  PATCH  /instances/{id}/ingredients/{key}
  POST   /instances/{id}/drafts     compose, or refine a version

Every change returns an InstanceView - the instance plus what the UI shows
alongside it - so the client never re-derives options or warnings.

Routes are plain `def`, not `async def`: the model and search calls block,
and FastAPI runs sync routes in a thread pool so they don't stall the server.
"""

from fastapi import APIRouter, HTTPException, Response

from api.errors import upstream
from api.meta import search_mode
from api.models import (CreateInstance, DraftRequest, IngredientPatch, InstanceSummary, InstanceView, JobRequest,
                        SettingsPatch, StoryPrompt)
from compose import compose
from knobs import INGREDIENT_KEYS
from llm import describe
from pipeline import (ingredient_options, knob_warnings, seed_knobs, set_job, set_level, set_setting, story_entry,
                      write_brief)
from schemas import IngredientState, Instance, Me
from sources.job_posting import read_job
from store import delete_instance, list_instances, load_instance, load_me, new_instance, save_instance
from suggest import suggest

router = APIRouter(prefix="/instances", tags=["instances"])


def _load(instance_id: str) -> Instance:
    instance = load_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"No instance {instance_id!r}")
    return instance


def view(instance: Instance, me: Me | None = None, notice: str | None = None) -> InstanceView:
    me = me or load_me()
    options, suggested = ingredient_options(instance, me)
    entry = story_entry(instance, me)
    return InstanceView(instance=instance, options=options, suggested=sorted(suggested),
                        warnings=knob_warnings(instance),
                        story_prompt=StoryPrompt(entry_key=entry.key, label=entry.label) if entry else None,
                        notice=notice)


@router.get("")
def list_reachouts() -> list[InstanceSummary]:
    """Most recently touched first."""
    return [InstanceSummary(id=i.id, title=i.title, updated_at=i.updated_at, drafts=len(i.drafts))
            for i in list_instances()]


@router.post("", status_code=201)
def create_instance(request: CreateInstance) -> InstanceView:
    return view(save_instance(new_instance(request.person)))


@router.get("/{instance_id}")
def get_instance(instance_id: str) -> InstanceView:
    return view(_load(instance_id))


@router.delete("/{instance_id}", status_code=204)
def delete_reachout(instance_id: str) -> Response:
    _load(instance_id)
    delete_instance(instance_id)
    return Response(status_code=204)


@router.patch("/{instance_id}/settings")
def update_settings(instance_id: str, patch: SettingsPatch) -> InstanceView:
    instance = _load(instance_id)
    changes = patch.model_dump(exclude_unset=True)
    # Goal first: switching it moves defaults that a knob in the same patch may then override.
    if "goal" in changes:
        set_setting(instance, "goal", changes.pop("goal") or "")
    for key, value in changes.items():
        set_setting(instance, key, value or "")
    return view(save_instance(instance))


@router.patch("/{instance_id}/ingredients/{key}")
def update_ingredient(instance_id: str, key: str, patch: IngredientPatch) -> InstanceView:
    if key not in INGREDIENT_KEYS:
        raise HTTPException(404, f"No ingredient {key!r} - one of {', '.join(INGREDIENT_KEYS)}")
    instance = _load(instance_id)
    state = instance.ingredients.setdefault(key, IngredientState())
    if patch.items is not None:
        state.items = list(dict.fromkeys(i.strip() for i in patch.items if i.strip()))
    if patch.level is not None:
        set_level(instance, key, patch.level)
    return view(save_instance(instance))


@router.put("/{instance_id}/job")
def update_job(instance_id: str, request: JobRequest) -> InstanceView:
    instance = _load(instance_id)
    value = request.input.strip()
    if not instance.job or instance.job.input != value:  # a JD is never read twice
        with upstream("Reading the job"):
            set_job(instance, read_job(value, mode=search_mode(request.mode)))
    return view(save_instance(instance))


@router.post("/{instance_id}/suggest")
def run_suggest(instance_id: str) -> InstanceView:
    """Suggestions and the brief are both optional: if either fails, the knobs
    are still seeded and the failure comes back as a notice, not an error."""
    instance = _load(instance_id)
    if not instance.settings.get("goal"):
        raise HTTPException(409, "Set a goal first - suggestions depend on it")
    me = load_me()
    failed = []
    try:
        instance.suggestions = suggest(instance, me)
    except Exception as exc:
        failed.append(f"suggestions ({type(exc).__name__})")
    try:
        instance.brief = write_brief(instance.person)
    except Exception as exc:
        failed.append(f"profile summary ({type(exc).__name__})")
    seed_knobs(instance, me)
    notice = (f"Couldn't get {' or '.join(failed)} from {describe('extract')}. The knobs still work - "
              "pick what fits.") if failed else None
    return view(save_instance(instance), me, notice)


@router.post("/{instance_id}/drafts", status_code=201)
def create_draft(instance_id: str, request: DraftRequest) -> InstanceView:
    """A new version from the current knobs. With `refine`, it rewrites a
    version (`base`, default the latest). Checks that still fail after retries
    are on the version's `problems`."""
    instance = _load(instance_id)
    if request.base is not None and not 0 <= request.base < len(instance.drafts):
        raise HTTPException(404, f"No version {request.base} - there are {len(instance.drafts)}")
    base = instance.drafts[request.base] if request.base is not None else None
    me = load_me()
    with upstream("Writing the draft"):
        instance.drafts.append(compose(instance, me, refine=request.refine, base=base))
    return view(save_instance(instance), me)
