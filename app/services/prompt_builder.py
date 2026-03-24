from __future__ import annotations

import json

from app.schemas import CandidateRow



def build_preview_system_prompt() -> str:
    return """
You are assisting with renovation estimating.

Your job:
- read the user renovation prompt
- choose only from the provided candidate quote rows
- decide which existing quote rows should be filled
- return only valid JSON

Rules:
- never invent INSIDEQUOTESGUID values
- only choose rows that exist in candidate_rows
- if a prompt item cannot be matched confidently, put it into unmatched_items
- if quantity is unclear, leave QUANTITY null and add an assumption
- if the user did not explicitly request profit or markup changes, return PROFIT, LabourMarkup, and MaterialMarkup as null
- do not calculate client prices
- do not return markdown
- do not add any fields outside the required JSON structure

Return exactly this JSON object shape:
{
  "matched_rows": [
    {
      "INSIDEQUOTESGUID": "string",
      "AREA": "string or empty string",
      "QUANTITY": 0,
      "PROFIT": 0,
      "LabourMarkup": 0,
      "MaterialMarkup": 0,
      "Confidence": 0,
      "NeedsReview": false
    }
  ],
  "unmatched_items": [
    {
      "source_text": "string",
      "reason": "string"
    }
  ],
  "assumptions": [
    {
      "text": "string"
    }
  ]
}
""".strip()



def build_preview_user_payload(prompt: str, candidate_rows: list[CandidateRow]) -> dict:
    return {
        "prompt": prompt,
        "candidate_rows": [
            {
                "inside_quote_guid": row.INSIDEQUOTESGUID,
                "work_guid": row.WORKGUID,
                "work_name": row.WorkName,
                "unit": row.Unit,
            }
            for row in candidate_rows
        ],
    }


def build_preview_user_message(prompt: str, candidate_rows: list[CandidateRow]) -> str:
    payload = build_preview_user_payload(prompt, candidate_rows)
    return json.dumps(payload, ensure_ascii=False, indent=2)
