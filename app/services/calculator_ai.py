"""AI helpers for the public build-cost calculator on combit-construction.com/calculator/.

Two entry points, both engine-agnostic (no pricing is read here):

* ``parse_project`` — turn a free-text project description into the wizard's own
  fields so the client can review a pre-filled form instead of clicking through.
* ``explain_estimate`` — a short, category-level explanation of the ballpark
  figure. Hard rule: no £-per-line-item, no work breakdown, no hours.

Both degrade to a keyword/template mock when OPENAI_API_KEY is absent.
"""

from __future__ import annotations

import logging
import re
from time import perf_counter

from app.config import settings
from app.schemas import (
    CalculatorAskRequest,
    CalculatorAskResponse,
    CalculatorExplainRequest,
    CalculatorExplainResponse,
    CalculatorParseResponse,
    LLMCalculatorAskOutput,
    LLMCalculatorExplainOutput,
    LLMCalculatorParseOutput,
)

logger = logging.getLogger(__name__)

# ---- the wizard's vocabulary (calc-popup.php) -------------------------------

# Leaf project types a client would actually describe. The "Extension" / "Loft
# conversion" / "Garage conversion" group toggles are set automatically by the
# wizard when a child is selected, so they are not offered to the model.
PROJECT_TYPES: tuple[str, ...] = (
    "House refurbishment",
    "Flat refurbishment",
    "New build",
    "Back extension",
    "Side extension",
    "1st floor extension",
    "Roof light loft conversion",
    "Dormer loft conversion",
    "Hip to gable loft conversion",
    "L-shaped loft conversion",
    "Mansard loft conversion",
    "Internal garage conversion",
    "Attached garage conversion",
    "Free standing garage conversion",
    "Outbuilding",
    "Porch",
)

# floor-area input name per project type (calc-popup.php)
AREA_FIELD_BY_TYPE: dict[str, str] = {
    "House refurbishment": "hr_ground_floor_area",
    "Flat refurbishment": "flat_refurbishment_area",
    "New build": "new_build_total_floor_area",
    "Back extension": "back_extension_area",
    "Side extension": "side_extension_area",
    "1st floor extension": "first_floor_extension_area",
    "Roof light loft conversion": "roof_light_loft_conversion_area",
    "Dormer loft conversion": "dormer_loft_conversion_area",
    "Hip to gable loft conversion": "hip_to_gable_loft_conversion_area",
    "L-shaped loft conversion": "l_shaped_loft_conversion_area",
    "Mansard loft conversion": "mansard_loft_conversion_area",
    "Internal garage conversion": "internal_garage_conversion_area",
    "Attached garage conversion": "attached_garage_conversion_area",
    "Free standing garage conversion": "free_standing_garage_conversion_area",
    "Outbuilding": "outbuilding_area",
    "Porch": "porch_area",
}
AREA_FIELDS: frozenset[str] = frozenset(AREA_FIELD_BY_TYPE.values()) | {
    "hr_first_floor_area",
}

# Yes/No option toggles the client can meaningfully mention up front.
OPTION_FIELDS: tuple[str, ...] = (
    "is_kitchen_fitting",
    "is_bathrooms",
    "is_remove_chimney",
    "is_boiler",
    "is_ufh_water",
    "is_ufh_electric",
    "is_driveway",
    "is_patio",
    "is_fences",
    "is_old_extension_demolition",
    "is_structural_alterations",
    "is_removing_non_load_bearing_walls",
    "is_full_house_rewiring",
    "is_radiator_replacement",
    "is_external_redecoration",
    "is_live_in_during_the_project",
)

LOCATION_FIELDS: dict[str, tuple[str, ...]] = {
    "location": ("Outer London", "Inner London"),
    "area": ("Non Conservation", "Conservation"),
    "listed_building": ("No", "Yes"),
    "material_quality": ("Regular", "Premium"),
    "type_of_property": (
        "Semi-detached house",
        "Detached house",
        "Bungalow",
        "Terraced house",
    ),
}

_DISCLAIMER = (
    "This is a ballpark figure based on £/m² rates, not a quote. The real cost "
    "depends on your drawings, specification and site conditions."
)

# ---- guardrails ----------------------------------------------------------------

_MONEY_PER_ITEM = re.compile(r"£\s?\d")
_HOURS = re.compile(r"\b\d+(\.\d+)?\s?(hrs?|hours?)\b", re.IGNORECASE)


def _clean_bullet(text: str) -> str | None:
    t = " ".join(str(text).split())
    if not t or len(t) > 160:
        t = t[:157].rstrip() + "…" if t else ""
    if not t:
        return None
    # never leak a per-item price or an hours figure into the client-facing copy
    if _MONEY_PER_ITEM.search(t) or _HOURS.search(t):
        return None
    return t


def _sanitise_explain(bullets: list[str], limit: int = 6) -> list[str]:
    out: list[str] = []
    for b in bullets:
        c = _clean_bullet(b)
        if c and c not in out:
            out.append(c)
        if len(out) >= limit:
            break
    return out


# ---- parse_project -----------------------------------------------------------

_PARSE_SYSTEM = f"""You help a UK homeowner fill in a building-cost calculator wizard for a London \
refurbishment contractor. Convert their free-text description into the wizard's own fields. \
Reply in English. Do NOT invent anything the text does not state.

Allowed project types (use the exact strings, pick every one that applies):
{chr(10).join("- " + p for p in PROJECT_TYPES)}

Area fields (square metres, only when the text gives a size or clear dimensions):
{chr(10).join("- " + f for f in sorted(AREA_FIELDS))}

Option toggles (list only the ones the text clearly asks for):
{chr(10).join("- " + o for o in OPTION_FIELDS)}

Location fields (only if stated): {", ".join(f"{k} in {v}" for k, v in LOCATION_FIELDS.items())}

Rules:
- If the text is NOT a building / renovation project description (a greeting, an
  unrelated question, random text, spam), set `understood` to false and leave every
  other field empty. Otherwise set `understood` to true.
- If a loft/extension type is ambiguous, pick the most likely and add a line to `uncertain`.
- Never guess an area from the number of bedrooms; leave it out.
- `summary`: one plain sentence describing what you understood.
"""


class CalculatorAIUnavailable(RuntimeError):
    pass


def _openai_client():
    from openai import OpenAI

    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=2,
    )


def _parse_mock(text: str) -> CalculatorParseResponse:
    low = text.lower()
    types: list[str] = []

    def add(t: str) -> None:
        if t not in types:
            types.append(t)

    if "loft" in low or "dormer" in low:
        add(
            "Dormer loft conversion"
            if "dormer" in low
            else "Roof light loft conversion"
        )
    if (
        "back extension" in low
        or "rear extension" in low
        or ("extension" in low and "side" not in low)
    ):
        add("Back extension")
    if "side extension" in low or "side return" in low:
        add("Side extension")
    if "porch" in low:
        add("Porch")
    if "outbuilding" in low or "garden room" in low or "garden office" in low:
        add("Outbuilding")
    if "garage" in low:
        add("Internal garage conversion")
    if "new build" in low or "newbuild" in low:
        add("New build")
    if "flat" in low and ("refurb" in low or "renovat" in low or "do up" in low):
        add("Flat refurbishment")
    if not types and (
        "refurb" in low or "renovat" in low or "whole house" in low or "do up" in low
    ):
        add("House refurbishment")

    options: dict[str, bool] = {}
    if "kitchen" in low:
        options["is_kitchen_fitting"] = True
    if "bathroom" in low or "en-suite" in low or "ensuite" in low:
        options["is_bathrooms"] = True
    if "underfloor" in low and "electric" in low:
        options["is_ufh_electric"] = True
    elif "underfloor" in low:
        options["is_ufh_water"] = True
    if "chimney" in low:
        options["is_remove_chimney"] = True
    if "rewir" in low:
        options["is_full_house_rewiring"] = True
    if "driveway" in low:
        options["is_driveway"] = True
    if "patio" in low:
        options["is_patio"] = True

    location: dict[str, str] = {}
    if "conservation" in low:
        location["area"] = "Conservation"
    if "listed" in low:
        location["listed_building"] = "Yes"
    if "inner london" in low or "central london" in low:
        location["location"] = "Inner London"

    understood = bool(types or options or location)
    return CalculatorParseResponse(
        project_types=types,
        options=options,
        location=location,
        summary=("Understood: " + ", ".join(types).lower()) if types else "",
        uncertain=[]
        if types
        else (["Could not identify a project type."] if understood else []),
        understood=understood,
        service_mode="mock",
    )


def parse_project(text: str) -> CalculatorParseResponse:
    text = text.strip()
    if not settings.openai_api_key or not settings.openai_model:
        return _parse_mock(text)
    try:
        client = _openai_client()
        started = perf_counter()
        completion = client.beta.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _PARSE_SYSTEM},
                {"role": "user", "content": text},
            ],
            response_format=LLMCalculatorParseOutput,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise CalculatorAIUnavailable("no structured output")
        logger.info(
            "calculator parse_project ok in %dms",
            int((perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 - degrade to mock, never 500 the wizard
        logger.warning("calculator parse_project failed (%s) — using mock", exc)
        return _parse_mock(text)

    types = [t for t in parsed.project_types if t in PROJECT_TYPES]
    areas = {
        a.field: round(float(a.sqm), 1)
        for a in parsed.areas
        if a.field in AREA_FIELDS and a.sqm and a.sqm > 0
    }
    options = {o: True for o in parsed.options if o in OPTION_FIELDS}
    location = {
        loc.field: loc.value
        for loc in parsed.location
        if loc.field in LOCATION_FIELDS and loc.value in LOCATION_FIELDS[loc.field]
    }
    understood = bool(parsed.understood) and bool(types or areas or options or location)
    if not understood:
        return CalculatorParseResponse(
            understood=False, summary="", service_mode="real"
        )
    return CalculatorParseResponse(
        project_types=types,
        areas=areas,
        options=options,
        location=location,
        uncertain=[u for u in parsed.uncertain if u][:4],
        summary=parsed.summary.strip()[:280],
        understood=True,
        service_mode="real",
    )


# ---- explain_estimate ------------------------------------------------------------

_EXPLAIN_SYSTEM = """You explain a building-cost BALLPARK to a UK homeowner, for a London \
refurbishment contractor. Reply in English, plain and reassuring.

Return three short lists:
- `included`: what a project of this kind broadly covers at this £/m² rate (structure, \
first & second fix, standard finishes, etc.) — 3 to 5 bullets.
- `excluded`: what is NOT in the figure (design & planning fees, structural sign-off, VAT, \
a contingency, upgrades beyond a standard spec, anything client-supplied) — 3 to 5 bullets.
- `cost_drivers`: 2 to 3 things that would move THIS client's number, based on their answers \
(inner vs outer London, conservation area, listed building, living in during the works, \
premium materials, unusually small or large floor area).

Hard rules:
- No pounds figures. No per-item prices. No work breakdown by trade line. No hour counts.
- Each bullet is one short sentence. Keep it general, category-level.
"""


def _explain_mock(req: CalculatorExplainRequest) -> CalculatorExplainResponse:
    types = ", ".join(req.project_types).lower() or "the works"
    included = [
        f"The main building work for {types}: structure, first and second fix, and standard finishes.",
        "Standard electrics, plumbing and heating connections within the scope shown.",
        "Basic making-good and decoration to a standard specification.",
    ]
    excluded = [
        "Design, planning and building-control fees.",
        "Structural engineer sign-off and party-wall costs.",
        "VAT and a contingency for the unknowns.",
        "Kitchens, bathrooms, flooring and fittings beyond a standard allowance.",
    ]
    drivers: list[str] = []
    loc = req.location
    if loc.get("location") == "Inner London":
        drivers.append("Inner London labour and access typically push the rate up.")
    if loc.get("area") == "Conservation":
        drivers.append("A conservation area adds cost for materials and detailing.")
    if loc.get("listed_building") == "Yes":
        drivers.append(
            "A listed building carries a premium for specialist work and consents."
        )
    if req.options.get("is_live_in_during_the_project"):
        drivers.append("Living in during the works slows the programme and adds cost.")
    if not drivers:
        drivers.append(
            "Floor area, specification level and site access are the biggest swing factors."
        )
    return CalculatorExplainResponse(
        included=_sanitise_explain(included),
        excluded=_sanitise_explain(excluded),
        cost_drivers=_sanitise_explain(drivers, limit=3),
        disclaimer=_DISCLAIMER,
        service_mode="mock",
    )


def explain_estimate(req: CalculatorExplainRequest) -> CalculatorExplainResponse:
    if not settings.openai_api_key or not settings.openai_model:
        return _explain_mock(req)
    context = {
        "project_types": req.project_types,
        "location": req.location,
        "options_selected": [k for k, v in req.options.items() if v],
        "floor_areas_sqm": req.floor_areas,
    }
    try:
        client = _openai_client()
        started = perf_counter()
        completion = client.beta.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _EXPLAIN_SYSTEM},
                {"role": "user", "content": _dump_context(context)},
            ],
            response_format=LLMCalculatorExplainOutput,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise CalculatorAIUnavailable("no structured output")
        logger.info(
            "calculator explain_estimate ok in %dms",
            int((perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("calculator explain_estimate failed (%s) — using mock", exc)
        return _explain_mock(req)

    included = _sanitise_explain(parsed.included)
    excluded = _sanitise_explain(parsed.excluded)
    drivers = _sanitise_explain(parsed.cost_drivers, limit=3)
    # if guardrails stripped everything, fall back to the safe template
    if not included or not excluded:
        return _explain_mock(req)
    return CalculatorExplainResponse(
        included=included,
        excluded=excluded,
        cost_drivers=drivers,
        disclaimer=_DISCLAIMER,
        service_mode="real",
    )


def _dump_context(context: dict) -> str:
    import json

    return json.dumps(context, ensure_ascii=False, indent=2)


# ---- ask_project --------------------------------------------------------------

_ASK_SYSTEM = """You answer ONE quick question from a UK homeowner using a free building-cost
calculator, on behalf of Combit — a London refurbishment contractor. Reply in English.

Keep it to 2-3 short sentences, max ~55 words, plain and friendly. Use their project
details for relevance. Draw on general UK renovation knowledge (planning permission,
permitted development, party wall, building control, typical timelines, sequencing,
what "conservation area" or "listed" means in practice, VAT, what a ballpark rate
usually covers).

Questions about timelines, how long the work takes, planning permission,
permitted development, building control, party wall, the order of works, VAT,
conservation-area or listed rules, and what a ballpark rate typically covers are
ALL on-topic — answer them normally, do NOT deflect them.

Hard limits:
- NEVER give a pounds figure, a price, or any cost breakdown.
- NEVER produce a scope of works, a task list, or "your project needs: 1)... 2)...".
- NEVER give design dimensions or tell them how to design anything.
- ONLY set `deflected` to true when they explicitly ask for a written quote, an
  itemised or line-by-line estimate, or a full scope of works, OR the question is
  genuinely not about a building/renovation project. Then answer with ONE line:
  that's best covered on a call with Combit — leave your details below and the team
  will come back to you.
- Otherwise set `deflected` to false, answer the question, and end with a short nudge
  to book a call or send their details to Combit for a firm answer.
"""

_ASK_DEFLECT = (
    "That's best covered on a call with Combit — leave your details below and the "
    "team will get back to you."
)

# a numbered/bulleted work list is a scope of works — not allowed out of this endpoint
_WORK_LIST = re.compile(r"(^|\n)\s*(\d+[.)]\s|[-*•]\s)", re.MULTILINE)


def _sanitise_answer(text: str) -> str:
    """Collapse whitespace, cap length, and refuse anything that leaks a price or a
    scope-of-works list. Returns "" when the answer must be replaced with a deflect."""
    t = " ".join(str(text).split())
    if not t:
        return ""
    if _MONEY_PER_ITEM.search(t) or _HOURS.search(t) or _WORK_LIST.search(str(text)):
        return ""
    words = t.split()
    if len(words) > 70:  # hard cap regardless of the prompt
        t = " ".join(words[:70]).rstrip(",.;:") + "…"
    return t


def _ask_mock(req: CalculatorAskRequest) -> CalculatorAskResponse:
    q = req.question.lower()
    detailed = any(
        w in q
        for w in (
            "detailed quote",
            "itemised",
            "itemized",
            "scope of works",
            "full estimate",
            "breakdown",
            "line by line",
        )
    )
    if detailed:
        return CalculatorAskResponse(
            answer=_ASK_DEFLECT, deflected=True, service_mode="mock"
        )

    if "planning" in q or "permitted development" in q:
        a = (
            "Many extensions and loft conversions fall under permitted development, but a "
            "conservation area or a listed building usually needs a full application. Combit "
            "can confirm what yours needs — send your details below."
        )
    elif "how long" in q or "timeline" in q or "take" in q:
        a = (
            "A typical single-storey extension or loft runs roughly 10-16 weeks on site once "
            "you have drawings and approvals; design and planning add time before that. Book a "
            "call with Combit for a schedule for your project."
        )
    elif "vat" in q:
        a = (
            "The calculator figure is shown excluding VAT. Most refurbishment work is charged "
            "at 20%, though some conversions can qualify for a reduced rate — Combit can advise "
            "for your case."
        )
    elif "conservation" in q:
        a = (
            "A conservation area mainly restricts what's visible from the street — materials, "
            "window styles, roof form — and tends to add cost and a planning application. "
            "Combit can talk you through it."
        )
    elif "listed" in q:
        a = (
            "A listed building needs listed building consent for most changes and specialist "
            "work, so budgets and timelines are higher. Combit handles listed properties — get "
            "in touch to discuss yours."
        )
    else:
        a = (
            "Good question — the calculator gives a ballpark only, so the reliable answer is a "
            "quick call with Combit. Leave your details below and the team will come back to you."
        )
    return CalculatorAskResponse(answer=a, deflected=False, service_mode="mock")


def ask_project(req: CalculatorAskRequest) -> CalculatorAskResponse:
    if not settings.openai_api_key or not settings.openai_model:
        return _ask_mock(req)

    context = {
        "project_types": req.project_types,
        "location": req.location,
        "options_selected": [k for k, v in req.options.items() if v],
        "ballpark_total_shown": req.totals.get("total"),
    }
    try:
        client = _openai_client()
        started = perf_counter()
        completion = client.beta.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _ASK_SYSTEM},
                {
                    "role": "user",
                    "content": "Project context:\n"
                    + _dump_context(context)
                    + "\n\nQuestion:\n"
                    + req.question.strip(),
                },
            ],
            response_format=LLMCalculatorAskOutput,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise CalculatorAIUnavailable("no structured output")
        logger.info(
            "calculator ask_project ok in %dms", int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("calculator ask_project failed (%s) — using mock", exc)
        return _ask_mock(req)

    answer = _sanitise_answer(parsed.answer)
    if not answer:
        # a guardrail tripped (price / hours / work list) — deflect rather than mock,
        # so we never ship a stripped or misleading reply
        return CalculatorAskResponse(
            answer=_ASK_DEFLECT, deflected=True, service_mode="real"
        )
    return CalculatorAskResponse(
        answer=answer, deflected=bool(parsed.deflected), service_mode="real"
    )
