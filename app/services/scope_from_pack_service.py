from __future__ import annotations

import json
import logging
from time import perf_counter

from app.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior Quantity Surveyor assistant for a UK construction company.

You receive a structured context pack extracted from architectural drawings and schedules.
Your task: produce a concise, accurate Scope of Works optimised for AI-assisted work item
matching against a construction pricing catalogue.

RULES:
- Group work by trade section, not by room (the matcher is trade-based).
- Each section maps to one trade: Demolition, Structure, Plastering, Electrics, Plumbing,
  Heating, Flooring, Decorating, Joinery, External, Roofing.
- Within each section, list specific work items as bullet lines with room context.
  Example: "- Strip plaster to walls: Kitchen, Living Room"
- Use standard UK QS language: first fix / second fix, strip out, make good, MF ceiling,
  screed, tanking, etc.
- Do NOT invent work not supported by the context pack data.
- Where drawings show a wet room without explicit work items, infer "strip and refurbish
  bathroom" as a reasonable assumption and mark it "(assumed)".
- Include Prelims as a section if prelim requirements are present.

OUTPUT: Return valid JSON only, in this exact schema:
{
  "scope_text": "<plain text version of the full scope — suitable as user-facing summary>",
  "sections": [
    {
      "key": "<lowercase_trade_key e.g. electrics>",
      "title": "<human title e.g. Electrics>",
      "lines": ["<work item line>", ...]
    }
  ]
}

Valid section keys: demolition, structure, plastering, electrics, plumbing, heating,
flooring, decorating, joinery, external, roofing, prelims, general.
"""


def generate_scope_from_context_pack(context_pack: dict) -> dict:
    if not settings.openai_api_key:
        return {"scope_text": _fallback_scope(context_pack), "model_name": "fallback"}

    model = settings.openai_intake_model or "gpt-4o"

    user_content = _build_user_content(context_pack)

    try:
        import json as _json

        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=2,
        )

        started_at = perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2000,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        raw = (response.choices[0].message.content or "").strip()
        usage = response.usage
        logger.info(
            "generate_scope_from_pack model=%s latency=%dms prompt=%d completion=%d",
            model,
            latency_ms,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
        parsed = _json.loads(raw)
        return {
            "scope_text": parsed.get("scope_text", ""),
            "sections": parsed.get("sections", []),
            "model_name": model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            } if usage else None,
        }
    except Exception:
        logger.exception("generate_scope_from_pack failed — using fallback")
        fallback = _fallback_scope(context_pack)
        return {"scope_text": fallback, "sections": [], "model_name": "fallback"}


def _build_user_content(pack: dict) -> str:
    parts: list[str] = []

    address = pack.get("property_address") or ""
    if address:
        parts.append(f"Property: {address}")

    rooms = pack.get("rooms") or []
    if rooms:
        parts.append("\nROOMS FROM DRAWINGS:")
        for r in rooms:
            area = r.get("area_m2")
            floor = r.get("floor_level") or ""
            area_str = f"{area} m²" if area else ""
            meta = ", ".join(filter(None, [floor, area_str]))
            parts.append(f"  {r.get('name', 'Room')}" + (f" ({meta})" if meta else ""))

    work_items = pack.get("work_items") or []
    if work_items:
        parts.append("\nWORK ITEMS FROM DRAWINGS:")
        for w in work_items:
            parts.append(f"  [{w.get('trade', 'General')}] {w.get('description', '')}")

    structural_notes = pack.get("structural_notes") or []
    if structural_notes:
        parts.append("\nSTRUCTURAL NOTES:")
        for n in structural_notes:
            parts.append(f"  {n.get('text', '')}")

    prelims = pack.get("prelim_requirements") or []
    if prelims:
        parts.append("\nPRELIM REQUIREMENTS:")
        for p in prelims:
            parts.append(f"  [{p.get('category', '')}] {p.get('text', '')}")

    electrical = pack.get("electrical_fixtures") or []
    if electrical:
        parts.append(f"\nELECTRICAL FIXTURES: {len(electrical)} items found in drawings")

    uncertainties = pack.get("uncertainties") or []
    if uncertainties:
        parts.append("\nUNCERTAINTIES / ISSUES:")
        for u in uncertainties[:5]:
            parts.append(f"  {u.get('text', '')}")

    return "\n".join(parts)


def _fallback_scope(pack: dict) -> str:
    lines = ["SCOPE OF WORK:", ""]
    rooms = pack.get("rooms") or []
    if rooms:
        lines.append("AREAS:")
        for r in rooms:
            area = r.get("area_m2")
            floor = r.get("floor_level") or ""
            area_str = f" — {area} m²" if area else ""
            floor_str = f" ({floor})" if floor else ""
            lines.append(f"{r.get('name', 'Room')}{floor_str}{area_str}")
        lines.append("")

    work_items = pack.get("work_items") or []
    if work_items:
        from collections import defaultdict
        by_trade: dict[str, list[str]] = defaultdict(list)
        for w in work_items:
            by_trade[(w.get("trade") or "General").upper()].append(w.get("description", ""))
        for trade, items in by_trade.items():
            lines.append(f"{trade}:")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")
    else:
        lines.extend(["DEMOLITION:", "", "STRUCTURE:", "", "ELECTRICAL:", "", "PLUMBING:", "", "FLOORING:", "", "DECORATING:", ""])

    structural_notes = pack.get("structural_notes") or []
    if structural_notes:
        lines.append("STRUCTURAL:")
        for n in structural_notes:
            lines.append(f"- {n.get('text', '')}")
        lines.append("")

    prelims = pack.get("prelim_requirements") or []
    if prelims:
        lines.append("PRELIMS:")
        for p in prelims:
            lines.append(f"- {p.get('text', '')}")
        lines.append("")

    return "\n".join(lines)
