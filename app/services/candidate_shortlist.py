from __future__ import annotations

from app.schemas import CandidateRow
from app.services.normalizer import normalize_unit, tokenize



def shortlist_candidates(prompt: str, candidate_rows: list[CandidateRow], limit: int = 25) -> list[CandidateRow]:
    prompt_tokens = tokenize(prompt)

    scored: list[tuple[int, CandidateRow]] = []
    for row in candidate_rows:
        work_tokens = tokenize(row.WorkName)
        overlap = len(prompt_tokens & work_tokens)

        unit_bonus = 0
        normalized_unit = normalize_unit(row.Unit)
        if normalized_unit and normalized_unit in prompt.lower():
            unit_bonus = 1

        scored.append((overlap + unit_bonus, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    shortlisted = [row for score, row in scored if score > 0][:limit]
    return shortlisted or candidate_rows[:limit]
