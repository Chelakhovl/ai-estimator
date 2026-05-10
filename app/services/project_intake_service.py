from __future__ import annotations

import re

from app.config import settings
from app.schemas import (
    IntakeAssetExcerpt,
    IntakeChatRequest,
    IntakeChatResponse,
    IntakeFinalizeRequest,
    IntakeFinalizeResponse,
    IntakeQuestion,
    ProjectBriefFact,
    ProjectBriefSection,
    ProjectBriefV1,
)
from app.services.project_intake_llm_client import IntakeLLMClient
from app.services.project_intake_renderer import render_project_brief_to_structured_scope
from app.services.scope_extractor import extract_scope


def _empty_section(title: str) -> ProjectBriefSection:
    return ProjectBriefSection(title=title, summary="", facts=[])


def _empty_project_brief() -> ProjectBriefV1:
    return ProjectBriefV1(
        project_overview=_empty_section("Project Overview"),
        existing_layout=_empty_section("Existing Layout"),
        proposed_layout_scope=_empty_section("Proposed Layout / Scope"),
        areas_dimensions=_empty_section("Areas & Dimensions"),
        structural=_empty_section("Structural"),
        extension_roof_external=_empty_section("Extension / Roof / External"),
        internal_build=_empty_section("Internal Build"),
        mep=_empty_section("MEP"),
        finishes=_empty_section("Finishes"),
        site_conditions=_empty_section("Site Conditions"),
        client_supply_vs_combit_supply=_empty_section("Client Supply vs Combit Supply"),
        missing_rfi=_empty_section("Missing / RFI"),
        source_documents=[],
        confidence_summary={},
    )


def _append_fact(section: ProjectBriefSection, text: str, *, source: str = "ai", confidence: str = "medium", uncertain: bool = False) -> None:
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        return
    if any(existing.text == normalized for existing in section.facts):
        return
    section.facts.append(ProjectBriefFact(text=normalized, source=source, confidence=confidence, uncertain=uncertain))


def _extract_assets_text(assets: list[IntakeAssetExcerpt]) -> str:
    return "\n\n".join(asset.document_excerpt for asset in assets if asset.document_excerpt.strip())


def _find_source_lines(assets: list[IntakeAssetExcerpt], turns: list, latest_message: str) -> list[str]:
    lines: list[str] = []
    for asset in assets:
        if asset.document_excerpt.strip():
            lines.extend(part.strip() for part in asset.document_excerpt.splitlines() if part.strip())
    for turn in turns:
        if turn.message.strip():
            lines.extend(part.strip() for part in turn.message.splitlines() if part.strip())
    if latest_message.strip():
        lines.extend(part.strip() for part in latest_message.splitlines() if part.strip())
    return lines


def _build_brief_from_sources(payload: IntakeChatRequest | IntakeFinalizeRequest) -> ProjectBriefV1:
    brief = _empty_project_brief()
    lines = _find_source_lines(payload.assets, payload.turns, getattr(payload, "message", ""))
    combined_text = "\n".join(lines).strip()
    extracted_scope = extract_scope(combined_text or getattr(payload, "message", "") or "Property scope not yet provided")

    if extracted_scope.property_context.property_type:
        brief.project_overview.summary = extracted_scope.property_context.property_type
        _append_fact(brief.project_overview, extracted_scope.property_context.property_type, source="document", confidence="high")
    if extracted_scope.property_context.location:
        _append_fact(brief.project_overview, extracted_scope.property_context.location, source="document", confidence="medium")
    for scope_label in extracted_scope.property_context.project_scopes:
        _append_fact(brief.proposed_layout_scope, scope_label, source="document", confidence="medium")

    for room in extracted_scope.rooms:
        _append_fact(
            brief.areas_dimensions,
            f"{room.level} - {room.name}: {room.width_m} x {room.length_m} m ({room.area_m2} m2)",
            source="document",
            confidence="high",
        )

    for section in extracted_scope.sections:
        target = None
        if section.key in {"structure", "steelworks", "groundworks"}:
            target = brief.structural
        elif section.key in {"demolition", "carpentry", "doors", "joinery"}:
            target = brief.internal_build
        elif section.key in {"electrical", "plumbing", "heating"}:
            target = brief.mep
        elif section.key in {"plastering", "tiling", "decorating", "flooring", "woodwork_painting"}:
            target = brief.finishes
        elif section.key in {"windows"}:
            target = brief.extension_roof_external
        elif section.key in {"preliminaries"}:
            target = brief.site_conditions
        else:
            target = brief.proposed_layout_scope
        for line in section.lines:
            _append_fact(target, line, source="document", confidence="medium")
        for item in section.count_items:
            _append_fact(target, f"{item.name}: {item.quantity}", source="document", confidence="medium")
        for item in section.measure_items:
            unit = f" {item.unit}" if item.unit else ""
            _append_fact(target, f"{item.name}: {item.value}{unit}", source="document", confidence="medium")
        for item in section.dimension_items:
            dims = [value for value in [item.length_m, item.width_m, item.height_m] if value]
            dims_label = " x ".join(str(value) for value in dims)
            _append_fact(target, f"{item.name}: {dims_label}", source="document", confidence="medium")

    supply_markers = ("client supply", "client supplied", "supply and fit", "by others", "appliances supplied")
    for line in lines:
        lowered = line.lower()
        room_match = re.match(
            r"^(?P<name>.+?):\s*(?P<width>\d+(?:\.\d+)?)\s*[x×]\s*(?P<length>\d+(?:\.\d+)?)",
            line,
            re.IGNORECASE,
        )
        if room_match:
            _append_fact(
                brief.areas_dimensions,
                f"{room_match.group('name').strip()}: {room_match.group('width')} x {room_match.group('length')} m",
                source="document",
                confidence="high",
            )
        if lowered.startswith("property:"):
            _append_fact(brief.project_overview, line.split(":", 1)[1].strip(), source="document", confidence="high")
        if any(marker in lowered for marker in ("refurb", "renovation", "strip out", "install", "layout")):
            _append_fact(brief.proposed_layout_scope, line, source="document", confidence="medium")
        if any(marker in lowered for marker in ("downlight", "socket", "consumer unit", "electrical", "plumbing", "wc", "basin", "shower", "bath")):
            _append_fact(brief.mep, line, source="document", confidence="medium")
        if any(marker in lowered for marker in ("paint", "decorate", "plaster", "skim", "tile", "flooring")):
            _append_fact(brief.finishes, line, source="document", confidence="medium")
        if any(marker in lowered for marker in supply_markers):
            _append_fact(brief.client_supply_vs_combit_supply, line, source="user", confidence="medium")
        if any(marker in lowered for marker in ("access", "parking", "occupied", "permit", "scaffold", "skip", "wait and load", "wait & load")):
            _append_fact(brief.site_conditions, line, source="user", confidence="medium")
        if any(marker in lowered for marker in ("existing", "current layout", "existing layout", "retain")):
            _append_fact(brief.existing_layout, line, source="user", confidence="medium")

    brief.source_documents = [
        {
            "filename": asset.filename,
            "asset_type": asset.asset_type,
            "extraction_status": asset.extraction_status,
            "warnings": asset.warnings,
        }
        for asset in payload.assets
    ]
    brief.confidence_summary = {
        "asset_count": len(payload.assets),
        "room_count": len(extracted_scope.rooms),
        "parsed_section_count": len(extracted_scope.sections),
        "document_warning_count": sum(len(asset.warnings) for asset in payload.assets),
    }
    return brief


def _build_missing_items(project_brief: ProjectBriefV1) -> tuple[list[str], list[IntakeQuestion], list[str]]:
    missing_items: list[str] = []
    questions: list[IntakeQuestion] = []
    warnings: list[str] = []

    if not project_brief.areas_dimensions.facts:
        missing_items.append("Room dimensions or area schedule are still missing.")
        questions.append(
            IntakeQuestion(
                id="areas-dimensions",
                label="Add the room schedule or the key dimensions from the drawings.",
                reason="AI Fill produces a much stronger first-pass draft when rooms and dimensions are explicit.",
            )
        )
    if not project_brief.site_conditions.facts:
        missing_items.append("Site access, scaffold, permits, or occupancy constraints are not clear yet.")
        questions.append(
            IntakeQuestion(
                id="site-conditions",
                label="Clarify access, scaffold, parking, permits, and whether the property is occupied.",
                reason="These project conditions change scope handling and later prelim logic.",
            )
        )
    if not project_brief.client_supply_vs_combit_supply.facts:
        missing_items.append("Client-supply vs Combit-supply split is not defined.")
        questions.append(
            IntakeQuestion(
                id="supply-split",
                label="Clarify which sanitaryware, appliances, finishes, or fixtures are client-supplied.",
                reason="The draft scope should separate supply responsibility before handoff into quote generation.",
            )
        )
    if not project_brief.mep.facts:
        warnings.append("MEP scope is still thin. Review whether plumbing, heating, or electrical works are fully captured.")

    if project_brief.confidence_summary.get("document_warning_count", 0):
        warnings.append("At least one uploaded document was extraction-poor or needed review.")

    project_brief.missing_rfi.summary = "Resolve these items before final handoff where possible."
    project_brief.missing_rfi.facts = [
        ProjectBriefFact(text=item, source="ai", confidence="medium", uncertain=True)
        for item in missing_items
    ]
    return missing_items, questions, warnings


def _merge_current_brief(current_project_brief_json: dict[str, object], generated_brief: ProjectBriefV1) -> ProjectBriefV1:
    if not current_project_brief_json:
        return generated_brief
    try:
        current = ProjectBriefV1.model_validate(current_project_brief_json)
    except Exception:
        return generated_brief

    for field_name in ProjectBriefV1.model_fields:
        current_value = getattr(current, field_name)
        generated_value = getattr(generated_brief, field_name)
        if isinstance(current_value, ProjectBriefSection):
            if current_value.summary and not generated_value.summary:
                generated_value.summary = current_value.summary
            for fact in current_value.facts:
                _append_fact(generated_value, fact.text, source=fact.source, confidence=fact.confidence, uncertain=fact.uncertain)
        elif field_name == "source_documents":
            generated_brief.source_documents = generated_brief.source_documents or current.source_documents
        elif field_name == "confidence_summary":
            combined = dict(current.confidence_summary)
            combined.update(generated_brief.confidence_summary)
            generated_brief.confidence_summary = combined
    return generated_brief


def chat_project_intake(payload: IntakeChatRequest) -> IntakeChatResponse:
    client = IntakeLLMClient()
    if client.is_enabled():
        try:
            llm_response = client.chat(payload)
            return IntakeChatResponse(
                assistant_message=llm_response.assistant_message,
                updated_project_brief_json=llm_response.updated_project_brief_json,
                questions=llm_response.questions,
                missing_items=llm_response.missing_items,
                warnings=llm_response.warnings,
                ready_to_finalize=llm_response.ready_to_finalize,
                model_name=str(client.last_metadata.get("model_name") or settings.openai_intake_fast_model),
            )
        except Exception:
            if not settings.allow_mock_fallback:
                raise

    generated_brief = _merge_current_brief(payload.current_project_brief_json, _build_brief_from_sources(payload))
    missing_items, questions, warnings = _build_missing_items(generated_brief)
    ready_to_finalize = bool(generated_brief.proposed_layout_scope.facts or generated_brief.areas_dimensions.facts) and len(missing_items) <= 1
    assistant_message = (
        "Received. Continue."
        if not payload.turns
        else "I updated the structured project brief. Review the missing items and answer the remaining intake questions before final handoff."
    )
    return IntakeChatResponse(
        assistant_message=assistant_message,
        updated_project_brief_json=generated_brief,
        questions=questions,
        missing_items=missing_items,
        warnings=warnings,
        ready_to_finalize=ready_to_finalize,
        model_name="fallback-intake",
    )


def finalize_project_intake(payload: IntakeFinalizeRequest) -> IntakeFinalizeResponse:
    client = IntakeLLMClient()
    if client.is_enabled():
        try:
            llm_response = client.finalize(payload)
            return IntakeFinalizeResponse(
                project_brief_json=llm_response.project_brief_json,
                structured_scope_markdown=llm_response.structured_scope_markdown,
                final_missing_items=llm_response.final_missing_items,
                final_warnings=llm_response.final_warnings,
                handoff_readiness=llm_response.handoff_readiness,
                model_name=str(client.last_metadata.get("model_name") or settings.openai_intake_model),
            )
        except Exception:
            if not settings.allow_mock_fallback:
                raise

    generated_brief = _merge_current_brief(payload.current_project_brief_json, _build_brief_from_sources(payload))
    missing_items, _questions, warnings = _build_missing_items(generated_brief)
    structured_scope_markdown = render_project_brief_to_structured_scope(generated_brief)
    handoff_readiness = bool(structured_scope_markdown.strip()) and bool(generated_brief.proposed_layout_scope.facts or generated_brief.areas_dimensions.facts)
    return IntakeFinalizeResponse(
        project_brief_json=generated_brief,
        structured_scope_markdown=structured_scope_markdown,
        final_missing_items=missing_items,
        final_warnings=warnings,
        handoff_readiness=handoff_readiness,
        model_name="fallback-intake",
    )
