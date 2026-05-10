from __future__ import annotations

from collections import Counter, defaultdict
from math import log

from app.schemas import CandidateRow
from app.services.normalizer import expand_tokens, normalize_unit, split_prompt_segments, tokenize, tokenize_terms
from app.services.scope_parsing import infer_section_keys_from_label, normalize_heading

_SECTION_GROUP_HINTS: dict[str, set[str]] = {
    "demolition": {"demolition", "remove", "strip", "waste"},
    "groundworks": {"ground", "groundwork", "excavate", "foundation", "concrete", "trench", "patio", "driveway"},
    "steelworks": {"steel", "beam", "column", "post", "junction", "connection"},
    "structure": {"structure", "structural", "roof", "brick", "block", "masonry", "dormer"},
    "carpentry": {"carpentry", "timber", "stud", "frame", "joist", "chipboard"},
    "ceilings_and_insulation": {"ceiling", "plasterboard", "board", "insulation", "pir", "acoustic", "rafter"},
    "doors": {"door", "frame", "pocket", "fire"},
    "joinery": {"joinery", "cabinet", "cupboard", "storage", "wardrobe", "worktop", "stair"},
    "plastering": {"plastering", "plaster", "plasterboard", "skim", "render", "wall", "ceiling"},
    "tiling": {"tiling", "tile", "grout", "splashback"},
    "plumbing": {"plumbing", "plumb", "sanitary", "basin", "bath", "shower", "sink", "toilet", "wc", "radiator"},
    "heating": {"heating", "heat", "boiler", "cylinder", "megaflo", "gas", "ufh", "radiator"},
    "electrical": {"electrical", "electric", "socket", "switch", "downlight", "light", "rewire", "consumer", "extractor", "outlet", "fitting", "thermostat", "appliance", "oven", "hob"},
    "decorating": {"decorating", "painting", "decorate", "paint", "mist", "coat"},
    "flooring": {"flooring", "floor", "carpet", "vinyl", "laminate", "timber", "engineered", "wood"},
    "windows": {"window", "glazing", "velux", "rooflight", "skylight"},
    "woodwork_painting": {"woodwork", "skirting", "architrave", "stair", "door"},
    "preliminaries": {"prelim", "protection", "toilet", "welfare", "management", "delivery", "site", "wait", "load"},
}
_ACTIVE_SECTION_GROUP_LIMIT = 35
_MIN_ROWS_PER_ACTIVE_SECTION = 8
def _combined_candidate_text(row: CandidateRow) -> str:
    return " ".join(
        filter(
            None,
            [
                row.WorkName,
                row.WorkGroupName or "",
                row.WorkItemCode or "",
                row.GuardrailStatus or "",
            ],
        )
    )


def _build_bigrams(terms: list[str]) -> set[str]:
    return {f"{left}_{right}" for left, right in zip(terms, terms[1:])}


def _tokenize_candidate_document(row: CandidateRow) -> list[str]:
    return tokenize_terms(_combined_candidate_text(row))


def _build_document_frequencies(candidate_rows: list[CandidateRow]) -> tuple[dict[str, int], float]:
    document_frequency: Counter[str] = Counter()
    total_terms = 0

    for row in candidate_rows:
        terms = _tokenize_candidate_document(row)
        total_terms += max(len(terms), 1)
        document_frequency.update(set(terms))

    average_document_length = total_terms / max(len(candidate_rows), 1)
    return dict(document_frequency), average_document_length


def _bm25_score(
    query_terms: list[str],
    document_terms: list[str],
    *,
    document_frequency: dict[str, int],
    document_count: int,
    average_document_length: float,
) -> float:
    if not query_terms or not document_terms or document_count <= 0:
        return 0.0

    term_frequency = Counter(document_terms)
    document_length = max(len(document_terms), 1)
    k1 = 1.2
    b = 0.75
    score = 0.0

    for term in set(query_terms):
        frequency = term_frequency.get(term, 0)
        if frequency <= 0:
            continue
        df = document_frequency.get(term, 0)
        idf = log(1 + (document_count - df + 0.5) / (df + 0.5))
        numerator = frequency * (k1 + 1)
        denominator = frequency + k1 * (1 - b + b * (document_length / max(average_document_length, 1.0)))
        score += idf * (numerator / denominator)

    return score


def _score_candidate(
    prompt_text: str,
    prompt_tokens: set[str],
    prompt_terms: list[str],
    prompt_bigrams: set[str],
    row: CandidateRow,
    *,
    document_frequency: dict[str, int],
    document_count: int,
    average_document_length: float,
) -> float:
    lowered_prompt = prompt_text.lower()
    work_tokens = expand_tokens(tokenize(row.WorkName))
    context_tokens = expand_tokens(tokenize(_combined_candidate_text(row)))
    group_tokens = expand_tokens(tokenize(row.WorkGroupName or ""))
    overlap = len(prompt_tokens & work_tokens) * 3 + len(prompt_tokens & context_tokens) + len(prompt_tokens & group_tokens) * 2
    document_terms = _tokenize_candidate_document(row)
    document_bigrams = _build_bigrams(document_terms)
    phrase_bonus = len(prompt_bigrams & document_bigrams) * 2.5
    bm25 = _bm25_score(
        prompt_terms,
        document_terms,
        document_frequency=document_frequency,
        document_count=document_count,
        average_document_length=average_document_length,
    )

    unit_bonus = 0
    normalized_unit = normalize_unit(row.Unit)
    if overlap > 0 and normalized_unit and normalized_unit in lowered_prompt:
        unit_bonus = 1
    return overlap + phrase_bonus + bm25 * 2 + unit_bonus


def _group_key(row: CandidateRow) -> str:
    return (row.WorkGroupName or "__ungrouped__").strip().lower() or "__ungrouped__"


def _normalize_section_heading(line: str) -> str:
    return normalize_heading(line)


def _extract_active_sections(prompt: str) -> list[tuple[str, set[str]]]:
    matches: list[tuple[int, str, set[str]]] = []
    seen_sections: set[str] = set()
    for index, line in enumerate(prompt.splitlines()):
        normalized = _normalize_section_heading(line)
        hints = _SECTION_GROUP_HINTS.get(normalized)
        if not hints or normalized in seen_sections:
            continue
        seen_sections.add(normalized)
        matches.append((index, normalized, hints))
    return [(section, hints) for _, section, hints in matches]


def _candidate_section_keys(row: CandidateRow) -> set[str]:
    keys: set[str] = set()
    for value in (row.WorkGroupName or "", row.WorkName, row.WorkItemCode or ""):
        keys.update(infer_section_keys_from_label(value))
    return keys


def _row_matches_section_hints(row: CandidateRow, section_key: str, hints: set[str]) -> bool:
    if section_key in _candidate_section_keys(row):
        return True
    haystack = _combined_candidate_text(row).lower()
    return any(hint in haystack for hint in hints)


def _active_section_alignment_bonus(row: CandidateRow, active_section_keys: set[str]) -> float:
    if not active_section_keys:
        return 0.0
    candidate_sections = _candidate_section_keys(row)
    if not candidate_sections:
        return 0.0
    if candidate_sections & active_section_keys:
        return 4.0
    return -0.75


def _quality_adjustment(prompt_text: str, row: CandidateRow) -> float:
    lowered_prompt = prompt_text.lower()
    work_name = (row.WorkName or "").strip().lower()
    group_name = (row.WorkGroupName or "").strip().lower()
    unit = normalize_unit(row.Unit)
    adjustment = 0.0

    if work_name.startswith("building control fee") and not any(
        keyword in lowered_prompt for keyword in ("building control", "control fee", "certificate", "inspection")
    ):
        adjustment -= 6.0

    if work_name.startswith("general electrics") and any(
        keyword in lowered_prompt
        for keyword in ("socket", "switch", "spotlight", "spotlights", "downlight", "light", "extractor", "appliance")
    ):
        adjustment -= 4.5

    if work_name.startswith("general plumbing") and any(
        keyword in lowered_prompt for keyword in ("radiator", "basin", "bath", "shower", "toilet", "wc", "sanitary")
    ):
        adjustment -= 4.5

    if work_name == "temporary floor protection":
        if any(keyword in lowered_prompt for keyword in ("site setup", "site protection", "correx", "temporary protection")):
            adjustment += 4.0
        else:
            adjustment -= 1.5

    if "light fitting" in work_name and any(keyword in lowered_prompt for keyword in ("spotlight", "spotlights", "downlight", "downlights")):
        adjustment += 4.0

    if "outlet" in work_name and any(keyword in lowered_prompt for keyword in ("socket", "sockets", "double socket", "double sockets", "outlet")):
        adjustment += 4.0

    if ("skimming walls" in work_name or "plastering walls" in work_name) and any(
        keyword in lowered_prompt for keyword in ("skim", "skimming", "wall", "walls", "plaster")
    ):
        adjustment += 3.5

    if ("skimming ceiling" in work_name or "plastering ceiling" in work_name or "fixing plasterboard to ceiling" in work_name) and any(
        keyword in lowered_prompt for keyword in ("ceiling", "ceilings", "plasterboard", "skim", "skimming")
    ):
        adjustment += 3.5

    if unit == "hour" and ("general" in work_name or "building control fee" in work_name):
        adjustment -= 1.5

    if unit == "hour" and "preliminaries" in group_name and not any(
        keyword in lowered_prompt for keyword in ("prelim", "site", "management", "delivery", "protection", "toilet", "welfare")
    ):
        adjustment -= 2.0

    if "internal wall" in lowered_prompt or "stud partition" in lowered_prompt or "partition wall" in lowered_prompt:
        if "stud partition" in work_name or ("partition" in work_name and "remove" in work_name):
            adjustment += 16.0
        if "kitchen wall and  floor units" in work_name or "kitchen wall and floor units" in work_name:
            adjustment -= 7.0

    if "boiler" in lowered_prompt and any(
        keyword in lowered_prompt for keyword in ("client supply", "client supplied", "supply only", "install")
    ):
        if "install boiler" in work_name:
            adjustment += 8.0
        if work_name.startswith("remove boiler"):
            adjustment -= 6.0

    if any(keyword in lowered_prompt for keyword in ("wall tiling", "floor tiling", "tile", "tiling")):
        if any(keyword in work_name for keyword in ("wall tile", "floor tile", "tiling", "tile")):
            adjustment += 4.0

    if any(keyword in lowered_prompt for keyword in ("electric underfloor heating", "electric ufh", "ufh")):
        if "electric underfloor heating" in work_name or ("underfloor heating" in work_name and "electric" in group_name):
            adjustment += 4.5
        if "remove underfloor heating" in work_name or ("remove" in work_name and "underfloor heating" in work_name):
            adjustment -= 5.0

    if any(keyword in lowered_prompt for keyword in ("remove", "strip out", "strip-out", "demolition")) and (
        work_name.startswith("install ") or " install " in work_name or "1st fix" in work_name or "first fix" in work_name or "second fix" in work_name
    ):
        adjustment -= 3.5

    if unit == "hour":
        if "radiator" in lowered_prompt and "radiator" not in work_name:
            adjustment -= 3.5
        if any(keyword in lowered_prompt for keyword in ("skirting", "architrave")) and not any(
            keyword in work_name for keyword in ("skirting", "architrave")
        ):
            adjustment -= 3.5
        if any(keyword in lowered_prompt for keyword in ("insulation", "rockwool", "pir")) and not any(
            keyword in work_name for keyword in ("insulation", "rockwool", "pir")
        ):
            adjustment -= 3.5
        if any(keyword in lowered_prompt for keyword in ("boiler", "underfloor heating", "ufh")) and not any(
            keyword in work_name for keyword in ("boiler", "underfloor heating", "ufh", "radiator")
        ):
            adjustment -= 4.0

    return adjustment


def _build_active_group_limit_overrides(
    scored_rows: list[tuple[float, CandidateRow]],
    *,
    active_sections: list[tuple[str, set[str]]],
    default_limit: int,
    active_limit: int,
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if not active_sections:
        return overrides

    for _, row in scored_rows:
        if any(_row_matches_section_hints(row, section_key, hints) for section_key, hints in active_sections):
            overrides[_group_key(row)] = max(default_limit, active_limit)
    return overrides


def _build_section_priority_selection(
    scored_rows: list[tuple[float, CandidateRow]],
    *,
    active_sections: list[tuple[str, set[str]]],
    limit: int,
    min_per_section: int,
) -> tuple[list[CandidateRow], set[str]]:
    if not active_sections or limit <= 0:
        return [], set()

    selected: list[CandidateRow] = []
    seen_ids: set[str] = set()
    capped_minimum = max(min_per_section, 0)

    for section_key, hints in active_sections:
        section_count = 0
        for _, row in scored_rows:
            if row.INSIDEQUOTESGUID in seen_ids:
                continue
            if not _row_matches_section_hints(row, section_key, hints):
                continue
            selected.append(row)
            seen_ids.add(row.INSIDEQUOTESGUID)
            section_count += 1
            if len(selected) >= limit or section_count >= capped_minimum:
                break
        if len(selected) >= limit:
            break

    return selected, seen_ids


def _build_group_balanced_selection(
    scored_rows: list[tuple[float, CandidateRow]],
    *,
    limit: int,
    per_group_limit: int,
    per_group_limit_overrides: dict[str, int] | None = None,
) -> list[CandidateRow]:
    rows_by_group: dict[str, list[tuple[int, CandidateRow]]] = defaultdict(list)
    for score, row in scored_rows:
        rows_by_group[_group_key(row)].append((score, row))

    group_order = sorted(
        rows_by_group,
        key=lambda key: (
            rows_by_group[key][0][0],
            len(rows_by_group[key]),
            key,
        ),
        reverse=True,
    )
    group_positions = {key: 0 for key in group_order}
    group_counts = {key: 0 for key in group_order}
    selected: list[CandidateRow] = []
    seen_ids: set[str] = set()

    while len(selected) < limit:
        added_in_round = False
        for key in group_order:
            group_limit = per_group_limit_overrides.get(key, per_group_limit) if per_group_limit_overrides else per_group_limit
            if group_counts[key] >= group_limit:
                continue

            group_rows = rows_by_group[key]
            while group_positions[key] < len(group_rows):
                _, row = group_rows[group_positions[key]]
                group_positions[key] += 1
                if row.INSIDEQUOTESGUID in seen_ids:
                    continue
                seen_ids.add(row.INSIDEQUOTESGUID)
                selected.append(row)
                group_counts[key] += 1
                added_in_round = True
                break

            if len(selected) >= limit:
                break

        if not added_in_round:
            break

    if len(selected) >= limit:
        return selected[:limit]

    for _, row in scored_rows:
        if row.INSIDEQUOTESGUID in seen_ids:
            continue
        seen_ids.add(row.INSIDEQUOTESGUID)
        selected.append(row)
        if len(selected) >= limit:
            break

    return selected[:limit]


def shortlist_candidates(
    prompt: str,
    candidate_rows: list[CandidateRow],
    limit: int = 150,
    per_segment_limit: int = 20,
    per_group_limit: int = 20,
) -> list[CandidateRow]:
    prompt_terms = tokenize_terms(prompt)
    prompt_tokens = expand_tokens(set(prompt_terms))
    prompt_bigrams = _build_bigrams(prompt_terms)
    segments = split_prompt_segments(prompt)
    best_scores: dict[str, tuple[float, CandidateRow]] = {}
    document_frequency, average_document_length = _build_document_frequencies(candidate_rows)
    document_count = len(candidate_rows)
    active_sections = _extract_active_sections(prompt)
    active_section_keys = {section_key for section_key, _ in active_sections}

    overall_scored: list[tuple[float, CandidateRow]] = []
    for row in candidate_rows:
        score = _score_candidate(
            prompt,
            prompt_tokens,
            prompt_terms,
            prompt_bigrams,
            row,
            document_frequency=document_frequency,
            document_count=document_count,
            average_document_length=average_document_length,
        )
        score += _active_section_alignment_bonus(row, active_section_keys)
        score += _quality_adjustment(prompt, row)
        overall_scored.append((score, row))

    for score, row in sorted(overall_scored, key=lambda item: item[0], reverse=True):
        if score <= 0:
            continue
        best_scores[row.INSIDEQUOTESGUID] = (score, row)

    for segment in segments:
        segment_terms = tokenize_terms(segment)
        segment_tokens = expand_tokens(set(segment_terms))
        segment_bigrams = _build_bigrams(segment_terms)
        scored_for_segment: list[tuple[float, CandidateRow]] = []
        for row in candidate_rows:
            score = _score_candidate(
                segment,
                segment_tokens,
                segment_terms,
                segment_bigrams,
                row,
                document_frequency=document_frequency,
                document_count=document_count,
                average_document_length=average_document_length,
            )
            score += _active_section_alignment_bonus(row, active_section_keys)
            score += _quality_adjustment(segment, row)
            if score > 0:
                scored_for_segment.append((score, row))

        for score, row in sorted(scored_for_segment, key=lambda item: item[0], reverse=True)[:per_segment_limit]:
            current = best_scores.get(row.INSIDEQUOTESGUID)
            if current is None or score > current[0]:
                best_scores[row.INSIDEQUOTESGUID] = (score, row)

    scored_rows = sorted(best_scores.values(), key=lambda item: item[0], reverse=True)
    if not scored_rows:
        return candidate_rows[:limit]

    base_group_limit = max(per_group_limit, per_segment_limit)
    active_group_limit_overrides = _build_active_group_limit_overrides(
        scored_rows,
        active_sections=active_sections,
        default_limit=base_group_limit,
        active_limit=max(base_group_limit, _ACTIVE_SECTION_GROUP_LIMIT),
    )
    section_priority_rows, section_priority_ids = _build_section_priority_selection(
        scored_rows,
        active_sections=active_sections,
        limit=limit,
        min_per_section=_MIN_ROWS_PER_ACTIVE_SECTION,
    )
    remaining_scored_rows = [
        (score, row)
        for score, row in scored_rows
        if row.INSIDEQUOTESGUID not in section_priority_ids
    ]
    shortlisted = section_priority_rows + _build_group_balanced_selection(
        remaining_scored_rows,
        limit=max(limit - len(section_priority_rows), 0),
        per_group_limit=base_group_limit,
        per_group_limit_overrides=active_group_limit_overrides,
    )
    return shortlisted[:limit] or candidate_rows[:limit]
