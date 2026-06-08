from __future__ import annotations

import json
import logging
import re
from time import perf_counter

from app.config import settings

logger = logging.getLogger(__name__)

_MAX_RECENT_TURNS = 15  # keep this many turns verbatim
_SUMMARISE_THRESHOLD = 20  # trigger compression when history exceeds this

_SYSTEM_PROMPT = """\
You are a construction document analyst assistant.
You have already extracted a structured context pack from the user's PDF drawings and XLSX schedules.
The context pack is provided below as markdown.

Your job:
1. Answer questions about the extracted data clearly and concisely.
2. When the user makes corrections or additions ("the kitchen is 4m x 3m", "add corridor 6m²"),
   update the context pack markdown accordingly and return the full updated version.
3. When the user asks you to clarify, expand, or restructure the markdown for copy-pasting into AI Fill,
   do so and return the updated markdown.
4. When you only answer a question without modifying data, set updated_markdown and room_patches to null.

Rules:
- Do NOT invent data not present in the original documents or stated by the user.
- Keep the markdown structure (headers, tables, bullet lists) consistent.
- Be concise in your assistant_message — one to three sentences.
- Whenever you add or update room dimensions, populate room_patches with ONLY the changed/added rooms.
  Each patch must include name and at least one of: width_m, length_m, area_m2 (use null for unknown).
  floor_level is optional — omit if unchanged or unknown.

Return ONLY valid JSON, no markdown fences:
{
  "assistant_message": "Done — I've updated the kitchen dimensions to 4.0m × 3.0m.",
  "updated_markdown": "..." | null,
  "room_patches": [{"name": "Kitchen", "floor_level": "Ground Floor", "width_m": 4.0, "length_m": 3.0, "area_m2": 12.0}] | null
}
"""


_FENCE_OPEN = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE | re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(content: str) -> dict:
    fallback = {"assistant_message": content.strip(), "updated_markdown": None, "room_patches": None}
    for candidate in [
        content,
        _FENCE_OPEN.sub("", content.strip(), count=1),
    ]:
        candidate = _FENCE_CLOSE.sub("", candidate.strip())
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    m = _JSON_BLOCK.search(content)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return fallback


def _summarise_turns(turns: list[dict], client: object, model: str) -> str:
    """Summarise a slice of chat history into 3-5 bullet points."""
    text = "\n".join(
        f"{t.get('role', 'user').upper()}: {t.get('message', '')}"
        for t in turns
        if t.get("role") in ("user", "assistant") and t.get("message")
    )
    if not text.strip():
        return ""
    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarise this construction document chat in 3-5 concise bullet points. Include only factual corrections and additions the user made. Be brief.",
                },
                {"role": "user", "content": text},
            ],
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Chat history summarisation failed: %s", exc)
        return ""


def chat_about_document(
    *,
    context_markdown: str,
    turns: list[dict],
    message: str,
) -> dict:
    """
    Chat with AI about an extracted document context pack.

    Returns:
        {
            "assistant_message": str,
            "updated_markdown": str | None,
            "model_name": str,
            "summary_used": bool,
            "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int} | None,
        }
    """
    if not settings.openai_api_key:
        return {
            "assistant_message": "AI service is not configured (no OpenAI key).",
            "updated_markdown": None,
            "model_name": "",
            "summary_used": False,
            "usage": None,
        }

    model = settings.openai_intake_fast_model or "gpt-4o-mini"

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=3,
        )

        messages: list[dict] = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT
                + "\n\n---\n\n## Current context pack\n\n"
                + context_markdown,
            }
        ]

        summary_used = False
        if len(turns) > _SUMMARISE_THRESHOLD:
            old_turns = turns[:-_MAX_RECENT_TURNS]
            recent_turns = turns[-_MAX_RECENT_TURNS:]
            summary = _summarise_turns(old_turns, client, model)
            if summary:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Earlier conversation summary (first {len(old_turns)} turns):\n{summary}",
                    }
                )
                summary_used = True
                logger.info(
                    "Chat history compressed: %d old turns summarised, %d recent turns kept",
                    len(old_turns),
                    len(recent_turns),
                )
            else:
                recent_turns = turns[-_MAX_RECENT_TURNS:]
        else:
            recent_turns = turns

        for turn in recent_turns:
            role = turn.get("role", "user")
            content = turn.get("message", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        started_at = perf_counter()
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=4000,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)

        raw_usage = response.usage
        usage = (
            {
                "prompt_tokens": raw_usage.prompt_tokens,
                "completion_tokens": raw_usage.completion_tokens,
                "total_tokens": raw_usage.total_tokens,
            }
            if raw_usage
            else None
        )
        logger.info(
            "Document chat completed — model=%s latency=%dms prompt=%d completion=%d",
            model,
            latency_ms,
            usage["prompt_tokens"] if usage else 0,
            usage["completion_tokens"] if usage else 0,
        )

        content = response.choices[0].message.content or "{}"
        data = _parse_json(content)
        data["model_name"] = model
        data["summary_used"] = summary_used
        data["usage"] = usage
        return data

    except Exception as exc:
        logger.warning("Document chat failed: %s", exc)
        return {
            "assistant_message": f"AI request failed: {exc}",
            "updated_markdown": None,
            "model_name": "",
            "summary_used": False,
            "usage": None,
        }
