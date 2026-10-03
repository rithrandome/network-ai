"""The knobs that shape a draft - defined once, rendered in both the creation
form and the playground sidebar, and read by the prompt builder.

Two kinds:
  Settings     describe the message (goal, who they are, channel, tone).
               Pick one template option, or type a custom value.
  Ingredients  content the message could use (your credibility, the role,
               the ask, common ground, ...). Each has items and a level:
                 off     - not sent to the model at all
                 hint    - referenced indirectly
                 mention - stated plainly
                 lead    - the opening is built on it (only one ingredient)

The goal comes first and narrows the rest: it decides which settings apply
and the default level of each ingredient (GOAL_RULES).
"""

from dataclasses import dataclass, field

LEVELS = ["off", "hint", "mention", "lead"]
LEVEL_LABELS = {"off": "Off", "hint": "Hint", "mention": "Mention", "lead": "Lead"}

# Statuses where there's a specific role in play - the JD field appears.
ROLE_STATUSES = {"Planning to apply", "Applied", "Interviewing"}


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    options: list[str]
    help: str = ""


@dataclass(frozen=True)
class Ingredient:
    key: str
    label: str
    help: str
    # Free-text placeholder for adding your own item.
    placeholder: str = ""


@dataclass(frozen=True)
class Channel:
    name: str
    max_chars: int
    has_subject: bool
    guide: str


GOAL = Setting("goal", "What's this reach-out for?",
               ["Referral", "Coffee chat", "Get on their radar", "Follow up", "Just connect"])
THEIR_ROLE = Setting("their_role", "Who they are to you",
                     ["Recruiter", "Hiring manager", "Potential teammate", "Leader", "Founder", "Alum/peer"])
STATUS = Setting("status", "Your status at their company",
                 ["No role in mind", "Planning to apply", "Applied", "Interviewing"])
CHANNEL = Setting("channel", "Channel", ["LinkedIn note", "InMail", "Email"])
TONE = Setting("tone", "Tone", ["Warm", "Professional", "Concise"])

SETTINGS = [GOAL, THEIR_ROLE, STATUS, CHANNEL, TONE]

CHANNELS = {
    # Free LinkedIn accounts get 200 characters for a connection note.
    "LinkedIn note": Channel("LinkedIn note", 200, False,
                             "A LinkedIn connection request note. No greeting line beyond 'Hi <name>,' and no sign-off."),
    "InMail": Channel("InMail", 1900, True,
                      "A LinkedIn InMail. Short subject line; 3-5 sentence body."),
    "Email": Channel("Email", 1500, True,
                     "A cold email. Specific subject line; short paragraphs; sign off with the sender's first name."),
}

INGREDIENTS = [
    Ingredient("credibility", "Your credibility",
               "Achievements from your profile that make you worth replying to.",
               "e.g. Shipped a RAG system used by 200 analysts"),
    Ingredient("role", "The role",
               "The specific job - from the JD you added.",
               "e.g. GenAI Developer, job ID R-48213"),
    Ingredient("ask", "The ask",
               "What you want them to do.",
               "e.g. Ask who the hiring manager is"),
    Ingredient("common_ground", "Common ground",
               "Things you share - school, employer, city, community.",
               "e.g. Both spoke at PyData Tampa"),
    Ingredient("hook", "Hook",
               "Something specific about them or their work.",
               "e.g. Her post on hiring GenAI engineers"),
    Ingredient("how_found", "How you found them",
               "A mutual connection, an event, a referral source.",
               "e.g. Ravi Kumar suggested I reach out"),
    Ingredient("personal_note", "Something only you know",
               "Context nobody else could add.",
               "e.g. I used their LangGraph talk in my project"),
]
INGREDIENT_KEYS = [i.key for i in INGREDIENTS]


@dataclass(frozen=True)
class GoalRule:
    show_status: bool
    channel: str
    levels: dict[str, str] = field(default_factory=dict)  # ingredient -> default level


GOAL_RULES = {
    "Referral": GoalRule(True, "Email", {"role": "lead", "credibility": "mention", "ask": "mention"}),
    "Coffee chat": GoalRule(False, "LinkedIn note", {"credibility": "hint", "ask": "mention"}),
    "Get on their radar": GoalRule(True, "LinkedIn note", {"credibility": "lead", "role": "mention", "ask": "hint"}),
    "Follow up": GoalRule(True, "Email", {"role": "lead", "ask": "mention"}),
    "Just connect": GoalRule(False, "LinkedIn note", {}),
}
DEFAULT_RULE = GoalRule(True, "LinkedIn note", {"credibility": "mention", "ask": "hint"})

DEFAULT_SETTINGS = {"their_role": "", "status": "No role in mind", "tone": "Professional"}


def rule_for(goal: str) -> GoalRule:
    return GOAL_RULES.get(goal, DEFAULT_RULE)


def default_levels(goal: str) -> dict[str, str]:
    """Every ingredient's starting level for this goal; anything unnamed is off."""
    levels = rule_for(goal).levels
    return {key: levels.get(key, "off") for key in INGREDIENT_KEYS}


def warnings_for(settings: dict[str, str]) -> list[str]:
    """Combinations worth questioning - shown, never enforced."""
    goal, role = settings.get("goal"), settings.get("their_role")
    notes = []
    if goal == "Referral" and role == "Recruiter":
        notes.append("Referrals usually come from teammates or leaders - a recruiter typically can't refer you.")
    if goal == "Follow up" and settings.get("status") == "No role in mind":
        notes.append("A follow-up usually refers to an application or an earlier conversation.")
    return notes
