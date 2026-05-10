from __future__ import annotations

import logging
from time import perf_counter

from app.config import settings

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a quantity surveyor's assistant. You receive per-page transcriptions from construction
drawings (PDFs). Each page entry may include a vision description, programmatic text, and table
summaries extracted from that page.

Your task: produce a single clean, readable document that an estimator can use directly for pricing.
You are cleaning and reorganising the raw data — not just copying it verbatim.

━━━ WHAT TO INCLUDE ━━━
• Exact member sizes, grades, and specifications (UC 203×203×71, S275, C16/20, etc.)
• Room names and dimensions grouped by floor level
• Structural requirements written as complete readable sentences
• Work items (construction tasks) grouped by trade
• Key schedule data: steel beams, lintels, pads/spreaders — formatted as clean numbered lists
• Conflicts or gaps under Uncertainties

━━━ WHAT TO SKIP ━━━
• Fragmented OCR text: partial sentences, lines starting mid-word, garbled sequences of numbers
  with no clear meaning (e.g. "5d 4d4d 4d4d Minimum", "BBeeddrroooomm 11")
• Title block contents: company name, phone number, logo text, drawing number references, scale bars
• Revision history blocks
• Tables where the content is clearly scrambled OCR (random letters, mirrored/rotated text)
• Any line under ~25 characters that is not a standalone meaningful fact

━━━ OUTPUT STRUCTURE ━━━
(Omit any section with no useful content. Do NOT use markdown # headings — use plain = underlines.)

PROJECT CONTEXT PACK
====================

DOCUMENT
========
Project name:
Address:
Document type:
Structural engineer (if named):
Drawing set / reference:
Revision and date:
Status (e.g. Preliminary / Not for Construction):
Important notes from title block (do not scale, print in colour, etc.):

ROOM / AREA SCHEDULE
====================

Rooms with dimensions — grouped by floor:
[Floor name]
- [Room name]: W × L = area m², sources: p__

Rooms without dimensions — grouped by floor:
[Floor name]
- [Room name]

Notes: [any conflicts or caveats about dimensions]

GENERAL STRUCTURAL NOTES
========================
[Numbered list of clear, complete structural requirements. Group by topic if more than one topic
is present: e.g. Masonry, Timber, Fire Protection, General. Write each note as a proper sentence.
Consolidate very similar notes into one. Skip fragments. Skip anything that reads like OCR noise.]

WORK ITEMS FROM DRAWINGS
========================
[Group by trade. Each item:
- Description.
  Source: page N]

STEEL BEAM SCHEDULE
===================
[If found. Numbered list:]
1. Section description
   Grade: ...
   Finish / note: ...

LINTEL SCHEDULE NOTES
=====================
[Plain text summary of lintels found. Include section, grade, fire protection rating.]

FOUNDATION / PAD NOTES
=======================
[Plain text. Note concrete spec (C16/20 etc.), key sizes where clearly readable.]

UNCERTAINTIES & GAPS
====================
[Numbered list. Specific items only: rooms without dimensions, OCR gaps, conflicts, missing data.]

SCOPE SUMMARY FOR AI FILL
=========================
[5-10 bullet points summarising the key construction scope. Plain English. Suitable for feeding
into an AI estimating tool. Focus on what work is actually being done on site.]
"""


def _build_input_text(page_bundles: list[dict]) -> str:
    """Format page bundles into a readable text block for the synthesis prompt."""
    parts: list[str] = []
    for bundle in page_bundles:
        src = bundle.get("source_file", "")
        pn = bundle.get("page_number", "?")
        pt = bundle.get("page_type", "other")
        header = f"=== Page {pn} of {src} ({pt}) ==="
        lines = [header]

        vision_desc = (bundle.get("vision_description") or "").strip()
        if vision_desc:
            lines.append(f"[Vision — complete transcription]\n{vision_desc}")

        prog_text = (bundle.get("programmatic_text") or "").strip()
        if prog_text:
            # If vision is present, programmatic text supplements it — trim more aggressively.
            # If no vision, programmatic text is the primary source — keep more of it.
            limit = 800 if vision_desc else 2000
            if len(prog_text) > limit:
                prog_text = prog_text[:limit] + "…"
            lines.append(f"[Programmatic text]\n{prog_text}")

        tables_text = (bundle.get("tables_text") or "").strip()
        if tables_text:
            lines.append(f"[Tables on this page]\n{tables_text}")

        # Only include pages that have at least some content
        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n\n".join(parts)


def synthesize_document_to_markdown(*, page_bundles: list[dict]) -> dict:
    """
    Take per-page bundles (programmatic + vision) and synthesize into clean readable Markdown.

    Returns:
        {"markdown": str, "model_name": str}
    """
    if not settings.openai_api_key:
        return {"markdown": "", "model_name": "", "error": "OpenAI not configured."}

    if not page_bundles:
        return {"markdown": "", "model_name": ""}

    model = settings.openai_intake_model or "gpt-4o"

    input_text = _build_input_text(page_bundles)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=3,
        )

        started_at = perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
            max_tokens=16000,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)
        logger.info(
            "Document synthesis completed — model=%s pages=%d latency=%dms",
            model, len(page_bundles), latency_ms,
        )

        markdown = (response.choices[0].message.content or "").strip()
        return {"markdown": markdown, "model_name": model}

    except Exception as exc:
        logger.warning("Document synthesis failed: %s", exc)
        return {"markdown": "", "model_name": "", "error": str(exc)}
