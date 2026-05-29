from __future__ import annotations

import json
import logging
import re
from time import perf_counter

from app.config import settings

logger = logging.getLogger(__name__)

_MAX_CHAT_TURNS = (
    20  # keep last 10 exchanges; prevents context overflow on long sessions
)

_SYSTEM_PROMPT = """\
You are a construction document analyst assistant.
You have already extracted a structured context pack from the user's PDF drawings and XLSX schedules.
The context pack is provided below as markdown.

Your job:
1. Answer questions about the extracted data clearly and concisely.
2. When the user makes corrections or additions ("the kitchen is 4m x 3m", "add 2 downlights in bathroom"),
   update the context pack markdown accordingly and return the full updated version.
3. When the user asks you to clarify, expand, or restructure the markdown for copy-pasting into AI Fill,
   do so and return the updated markdown.
4. When you only answer a question without modifying data, set updated_markdown to null.

Rules:
- Do NOT invent data not present in the original documents or stated by the user.
- Keep the markdown structure (headers, tables, bullet lists) consistent.
- Be concise in your assistant_message — one to three sentences.

Return ONLY valid JSON, no markdown fences:
{
  "assistant_message": "Done — I've updated the kitchen dimensions to 4.0m × 3.0m.",
  "updated_markdown": "# Project Context Pack\\n..." | null
}
"""


_FENCE_OPEN = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE | re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(content: str) -> dict:
    fallback = {"assistant_message": content.strip(), "updated_markdown": None}
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
        }
    """
    if not settings.openai_api_key:
        return {
            "assistant_message": "AI service is not configured (no OpenAI key).",
            "updated_markdown": None,
            "model_name": "",
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

        recent_turns = (
            turns[-_MAX_CHAT_TURNS:] if len(turns) > _MAX_CHAT_TURNS else turns
        )
        if len(turns) > _MAX_CHAT_TURNS:
            logger.info(
                "Chat history truncated from %d to %d turns to prevent context overflow.",
                len(turns),
                _MAX_CHAT_TURNS,
            )
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
        logger.info(
            "Document chat completed — model=%s latency=%dms", model, latency_ms
        )

        content = response.choices[0].message.content or "{}"
        data = _parse_json(content)
        data["model_name"] = model
        return data

    except Exception as exc:
        logger.warning("Document chat failed: %s", exc)
        return {
            "assistant_message": f"AI request failed: {exc}",
            "updated_markdown": None,
            "model_name": "",
        }
