from __future__ import annotations

from app.schemas import (
    ExtractedCountItem,
    ExtractedPropertyContext,
    ExtractedSection,
    ScopeExtractionResponse,
    StructuredTakeoffSummary,
)
from app.services.matcher import (
    _find_source_snippet,
    _looks_like_valid_source_snippet,
    _resolve_source_snippet,
)


def _scope(*, sections=None) -> ScopeExtractionResponse:
    return ScopeExtractionResponse(
        property_context=ExtractedPropertyContext(),
        rooms=[],
        room_takeoff=[],
        sections=sections or [],
        takeoff_summary=StructuredTakeoffSummary(),
        assumptions=[],
    )


class TestFindSourceSnippetFromExtractedScope:
    def test_picks_the_line_with_the_most_token_overlap(self):
        scope = _scope(
            sections=[
                ExtractedSection(
                    key="electrical",
                    title="Electrical",
                    lines=[
                        "Install new gas boiler",
                        "Kitchen: 8 downlights",
                        "Kitchen: 6 double sockets",
                    ],
                ),
            ]
        )

        result = _find_source_snippet(
            "Install LED downlights",
            "Kitchen",
            scope_text="",
            extracted_scope=scope,
        )

        assert result == "Kitchen: 8 downlights"

    def test_reads_raw_text_off_structured_count_items(self):
        scope = _scope(
            sections=[
                ExtractedSection(
                    key="electrical",
                    title="Electrical",
                    count_items=[
                        ExtractedCountItem(
                            section_key="electrical",
                            name="downlight",
                            quantity=8,
                            raw_text="8x downlights to kitchen ceiling",
                        ),
                    ],
                ),
            ]
        )

        result = _find_source_snippet(
            "Install LED downlight",
            "Kitchen",
            scope_text="",
            extracted_scope=scope,
        )

        assert result == "8x downlights to kitchen ceiling"

    def test_returns_none_when_nothing_scores_well_enough(self):
        scope = _scope(
            sections=[
                ExtractedSection(
                    key="decorating",
                    title="Decorating",
                    lines=["Paint the hallway ceiling white"],
                ),
            ]
        )

        result = _find_source_snippet(
            "Supply and fit radiator",
            "Throughout",
            scope_text="",
            extracted_scope=scope,
        )

        assert result is None

    def test_returns_none_for_row_name_with_no_significant_tokens(self):
        scope = _scope(sections=[ExtractedSection(key="a", title="A", lines=["x"])])

        result = _find_source_snippet("the a", "", scope_text="", extracted_scope=scope)

        assert result is None


class TestFindSourceSnippetFromRawPrompt:
    def test_falls_back_to_splitting_raw_prompt_when_no_extracted_scope(self):
        prompt = "Strip out existing kitchen units.\nInstall new radiators throughout."

        result = _find_source_snippet(
            "Strip out kitchen",
            "",
            scope_text=prompt,
            extracted_scope=None,
        )

        assert result == "Strip out existing kitchen units"

    def test_returns_none_on_no_overlap(self):
        prompt = "Paint the hallway ceiling white."

        result = _find_source_snippet(
            "Install boiler", "", scope_text=prompt, extracted_scope=None
        )

        assert result is None

    def test_returns_none_rather_than_quoting_the_entire_prompt(self):
        # Regression: when room extraction fails and the prompt is one long
        # run-on sentence with no '.', ';' or newline to split on, the raw-prompt
        # fallback used to produce exactly one "line" - the whole prompt - which
        # trivially had the highest token overlap and was returned as the
        # "citation", defeating the point of a short quote in the most common
        # degraded case (no room sizes found).
        prompt = (
            "Strip out the existing kitchen units and worktops throughout the "
            "ground floor and install new radiators in every room including the "
            "hallway and the lounge and the dining room before redecorating "
            "and also replastering every wall that gets disturbed along the way"
        )

        result = _find_source_snippet(
            "Strip out kitchen", "", scope_text=prompt, extracted_scope=None
        )

        assert result is None


class TestLooksLikeValidSourceSnippet:
    def test_accepts_a_snippet_that_substantially_overlaps_the_prompt(self):
        prompt = "ELECTRICAL:\nPartial rewire to ground floor only"

        assert (
            _looks_like_valid_source_snippet(
                "Partial rewire to ground floor only", prompt
            )
            is True
        )

    def test_rejects_a_snippet_unrelated_to_the_prompt(self):
        prompt = "ELECTRICAL:\nPartial rewire to ground floor only"

        assert (
            _looks_like_valid_source_snippet(
                "Install a swimming pool with a retractable roof", prompt
            )
            is False
        )

    def test_rejects_blank_snippet(self):
        assert _looks_like_valid_source_snippet(None, "anything") is False
        assert _looks_like_valid_source_snippet("   ", "anything") is False

    def test_rejects_a_sentence_stitched_from_scattered_real_words(self):
        # Regression: a bag-of-words overlap check can be fooled by a sentence
        # built out of real words that appear somewhere in the prompt but were
        # never actually adjacent - it isn't a quote just because every word in
        # it happens to exist somewhere in the source text.
        prompt = (
            "ELECTRICAL: Partial rewire to ground floor only. "
            "PLUMBING: Replace the boiler in the loft. "
            "DECORATING: Paint the hallway ceiling white."
        )
        fabricated = "Paint the boiler ground floor ceiling only ground rewire"

        assert _looks_like_valid_source_snippet(fabricated, prompt) is False

    def test_rejects_candidate_longer_than_the_max_snippet_length(self):
        prompt = "A" * 500
        assert _looks_like_valid_source_snippet("A" * 300, prompt) is False


class TestResolveSourceSnippet:
    def test_prefers_valid_llm_reported_snippet_over_finder(self):
        result = _resolve_source_snippet(
            llm_reported="Partial rewire to ground floor only",
            row_name="Full property rewire",
            area_hint="Ground Floor",
            scope_text="ELECTRICAL:\nPartial rewire to ground floor only",
            extracted_scope=None,
        )

        assert result == "Partial rewire to ground floor only"

    def test_falls_back_to_finder_when_llm_reported_snippet_is_hallucinated(self):
        result = _resolve_source_snippet(
            llm_reported="Install a helicopter pad",
            row_name="Full property rewire",
            area_hint="Ground Floor",
            scope_text="ELECTRICAL:\nPartial rewire to ground floor only",
            extracted_scope=None,
        )

        assert result == "Partial rewire to ground floor only"
