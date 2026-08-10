from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CandidateRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    INSIDEQUOTESGUID: str
    WORKGUID: str | None = None
    WorkItemCode: str | None = None
    WorkGroupGUID: str | None = None
    WorkGroupName: str | None = None
    GuardrailCode: str | None = None
    GuardrailStatus: str | None = None
    GuardrailDetail: str | None = None
    WorkName: str
    Unit: str | None = None
    WorkLabourCost: float = 0
    WorkMatCost: float = 0
    WorkOtherCost: float = 0
    WorkQTYforNorm: float = Field(default=1, gt=0)
    WorkToolsHire: float = 0
    WorkDays: float = 0
    Work8Skips: float = 0
    Work12Skips: float = 0
    AREA: str | None = ""
    QUANTITY: float | None = 0
    PROFIT: float = 0
    LabourMarkup: float = 0
    MaterialMarkup: float = 0
    IsFixedRate: bool = False
    FixedClientRate: float | None = None


class PreviewRulesContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_version: str = ""
    canonical_source: str = ""
    runtime_source: str = ""
    published_version: str = ""
    published_at: str | None = None


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote_guid: str
    prompt: str = Field(min_length=1)
    candidate_rows: list[CandidateRow] = Field(min_length=1)
    accepted_examples: list["PreviewAcceptedExample"] = Field(default_factory=list)
    rules_context: PreviewRulesContext | None = None
    normalized_prompt_markdown: str | None = None
    normalized_scope: ScopeExtractionResponse | None = None
    clarification_answers: list["PromptNormalizationAnswer"] = Field(
        default_factory=list
    )
    intake_confidence: float | None = Field(default=None, ge=0, le=1)
    document_context: str | None = (
        None  # AI-synthesized project context from Document Analysis
    )
    catalog_context: "CatalogContext | None" = (
        None  # enables batch-pricing of unmatched items
    )


class ScopeExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class PromptNormalizationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    value: str | float | bool | list[str]


class ClarificationQuestionOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    label: str


class ClarificationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    reason: str
    required: bool = True
    answer_type: str
    options: list[ClarificationQuestionOption] = Field(default_factory=list)
    applies_to_section: str | None = None


class PromptBlocker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    title: str
    message: str
    severity: str = "warning"
    scope_label: str | None = None
    recommended_action: str = ""


class PromptNormalizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    quote_guid: str | None = None
    project_type_id: str | int | None = None
    site_id: str | int | None = None
    template_id: str | None = None
    existing_quote_context: dict[str, object] | None = None
    clarification_answers: list[PromptNormalizationAnswer] = Field(default_factory=list)


class PreviewMatchedRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    INSIDEQUOTESGUID: str
    WORKGUID: str | None = None
    WorkName: str
    Unit: str | None = None
    AREA: str = ""
    QUANTITY: float = Field(gt=0)
    PROFIT: float = Field(ge=0, le=200)
    LabourMarkup: float = Field(ge=0, le=200)
    MaterialMarkup: float = Field(ge=0, le=100)
    WorkLabourCost: float = 0
    WorkMatCost: float = 0
    WorkOtherCost: float = 0
    WorkQTYforNorm: float = Field(gt=0)
    ClientCostPerUnit: float = Field(ge=0)
    ClientTotalCost: float = Field(ge=0)
    Confidence: float = Field(ge=0, le=1)
    NeedsReview: bool = False
    ReviewReason: str | None = None
    MatchedSectionKey: str | None = None
    MatchedSectionTitle: str | None = None
    MatchedSectionOrder: int | None = None


class PreviewUnmatchedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_text: str
    reason: str


class PreviewAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    kind: str = "general"
    severity: str = "info"


class PreviewReviewQueueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    title: str
    message: str
    severity: str = "warning"
    kind: str = "manual_review"
    scope_label: str | None = None
    recommended_action: str = ""
    suggested_prompt: str = ""


class PreviewCoveragePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    title: str
    message: str
    severity: str = "warning"
    scope_label: str | None = None
    suggested_prompt: str = ""


class PreviewAcceptedExampleRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_name: str
    unit: str | None = None
    area: str | None = ""
    quantity: float | None = None
    section_title: str | None = None


class PreviewAcceptedExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    prompt_template_title: str | None = ""
    accepted_rows: list[PreviewAcceptedExampleRow] = Field(default_factory=list)


class ExtractedPropertyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_type: str = ""
    location: str = ""
    project_scopes: list[str] = Field(default_factory=list)


class ExtractedRoom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str
    name: str
    width_m: float = Field(gt=0)
    length_m: float = Field(gt=0)
    area_m2: float = Field(gt=0)


class ExtractedRoomTakeoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str
    name: str
    room_type: str = ""
    area_m2: float = Field(gt=0)
    estimated_wall_finish_area_m2: float = Field(ge=0)
    estimated_ceiling_finish_area_m2: float = Field(ge=0)
    estimated_skirting_lm: float = Field(ge=0)
    estimated_wall_tiling_area_m2: float = Field(ge=0)
    estimated_floor_finish_area_m2: float = Field(ge=0)
    estimated_plumbing_points: float = Field(ge=0, default=0)
    estimated_wc_count: float = Field(ge=0, default=0)
    estimated_basin_count: float = Field(ge=0, default=0)
    estimated_bath_count: float = Field(ge=0, default=0)
    estimated_shower_count: float = Field(ge=0, default=0)
    estimated_sink_count: float = Field(ge=0, default=0)
    estimated_sanitary_set_count: float = Field(ge=0, default=0)
    estimated_vanity_unit_count: float = Field(ge=0, default=0)
    estimated_concealed_cistern_count: float = Field(ge=0, default=0)
    estimated_appliance_connection_count: float = Field(ge=0, default=0)
    estimated_socket_count: float = Field(ge=0, default=0)
    estimated_switch_count: float = Field(ge=0, default=0)
    estimated_lighting_point_count: float = Field(ge=0, default=0)
    estimated_downlight_count: float = Field(ge=0, default=0)
    estimated_extractor_fan_count: float = Field(ge=0, default=0)
    estimated_ducted_extractor_count: float = Field(ge=0, default=0)
    estimated_hardwired_appliance_count: float = Field(ge=0, default=0)
    estimated_electrical_second_fix_points: float = Field(ge=0, default=0)
    estimated_plumbing_second_fix_points: float = Field(ge=0, default=0)


class ExtractedCountItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: str
    name: str
    quantity: float = Field(gt=0)
    raw_text: str


class ExtractedMeasureItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: str
    name: str
    value: float
    unit: str | None = None
    raw_text: str


class ExtractedDimensionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: str
    name: str
    raw_text: str
    count: float = Field(default=1, gt=0)
    length_m: float | None = Field(default=None, gt=0)
    width_m: float | None = Field(default=None, gt=0)
    height_m: float | None = Field(default=None, gt=0)
    area_m2: float | None = Field(default=None, gt=0)
    volume_m3: float | None = Field(default=None, gt=0)


class ExtractedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    lines: list[str] = Field(default_factory=list)
    count_items: list[ExtractedCountItem] = Field(default_factory=list)
    measure_items: list[ExtractedMeasureItem] = Field(default_factory=list)
    dimension_items: list[ExtractedDimensionItem] = Field(default_factory=list)


class StructuredTakeoffSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_count: int = 0
    basement_area_m2: float = 0
    ground_floor_area_m2: float = 0
    first_floor_area_m2: float = 0
    second_floor_area_m2: float = 0
    loft_area_m2: float = 0
    total_internal_floor_area_m2: float = 0
    estimated_wall_finish_area_m2: float = 0
    estimated_ceiling_finish_area_m2: float = 0
    estimated_skirting_lm: float = 0
    estimated_hard_floor_scope_area_m2: float = 0
    estimated_carpet_scope_area_m2: float = 0
    estimated_boarding_coverage_m2: float = 0
    estimated_insulation_coverage_m2: float = 0
    estimated_extension_slab_area_m2: float = 0
    estimated_extension_flat_roof_area_m2: float = 0
    estimated_extension_external_wall_area_m2: float = 0
    estimated_loft_roof_finish_area_m2: float = 0
    estimated_roof_removal_area_m2: float = 0
    estimated_dormer_build_area_m2: float = 0
    wet_room_count: int = 0
    estimated_wet_room_wall_tiling_area_m2: float = 0
    estimated_wet_room_floor_tiling_area_m2: float = 0
    total_wc_count: float = 0
    total_basin_count: float = 0
    total_bath_count: float = 0
    total_shower_count: float = 0
    total_sink_count: float = 0
    total_sanitary_set_count: float = 0
    total_vanity_unit_count: float = 0
    total_concealed_cistern_count: float = 0
    total_appliance_connection_count: float = 0
    consumer_unit_count: float = 0
    downlight_count: float = 0
    ducted_extractor_count: float = 0
    hardwired_appliance_count: float = 0
    client_supply_sanitaryware: bool = False
    client_supply_appliances: bool = False
    estimated_electrical_second_fix_points: float = 0
    estimated_plumbing_second_fix_points: float = 0
    pad_foundation_volume_m3: float = 0
    trench_foundation_volume_m3: float = 0
    internal_foundation_volume_m3: float = 0
    total_foundation_concrete_m3: float = 0
    patio_area_m2: float = 0
    roof_area_m2: float = 0
    dormer_area_m2: float = 0
    skylight_count: int = 0
    skylight_area_m2: float = 0
    velux_count: int = 0
    upvc_window_count: int = 0
    steel_post_count: int = 0
    steel_junction_count: int = 0
    radiator_removal_count: int = 0
    radiator_install_count: int = 0
    socket_count: int = 0
    switch_count: int = 0
    extractor_fan_count: int = 0
    door_set_count: int = 0
    fire_rated_door_set_count: float = 0
    pocket_door_set_count: float = 0
    fire_rated_pocket_door_set_count: float = 0
    double_pocket_door_set_count: float = 0
    staircase_count: float = 0
    storage_door_set_count: float = 0
    architrave_set_count: float = 0
    estimated_architrave_lm: float = 0
    window_board_count: float = 0
    estimated_window_board_lm: float = 0
    door_supply_allowance_total: float = 0
    tree_count: int = 0
    fence_count: int = 0
    soil_volume_m3: float = 0


class ScopeExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser_mode: str = "rule_parser"
    property_context: ExtractedPropertyContext
    rooms: list[ExtractedRoom]
    room_takeoff: list[ExtractedRoomTakeoff] = Field(default_factory=list)
    sections: list[ExtractedSection]
    takeoff_summary: StructuredTakeoffSummary
    assumptions: list[PreviewAssumption]
    error_text: str = ""


class PromptNormalizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    confidence: float = Field(ge=0, le=1)
    normalized_prompt_markdown: str = ""
    normalized_scope: ScopeExtractionResponse | None = None
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    blockers: list[PromptBlocker] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    error_text: str = ""


class PreviewReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched_count: int
    ready_count: int
    review_count: int
    unmatched_count: int
    assumption_count: int


class PreviewTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_row_count: int = 0
    shortlist_size: int = 0
    accepted_example_count: int = 0
    room_count: int = 0
    llm_attempt_count: int = 0
    llm_latency_ms: int = 0
    parse_retry_used: bool = False
    structured_output_mode: str = ""
    fallback_reason: str = ""
    takeoff_heuristics_version: str = ""
    takeoff_heuristics_source: str = ""
    rule_version: str = ""
    canonical_source: str = ""
    runtime_source: str = ""
    published_version: str = ""
    published_at: str = ""


class PreviewCoverageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_count: int = 0
    section_count: int = 0
    matched_section_count: int = 0
    unmatched_item_count: int = 0
    normalized_scope_used: bool = False


class CustomPricedRow(BaseModel):
    """An unmatched scope item that was auto-priced via batch GPT call."""

    model_config = ConfigDict(extra="forbid")

    source_text: str  # original unmatched item description
    name: str
    unit: str
    labour_cost: float
    material_cost: float
    other_cost: float
    work_days: float
    qty_for_norm: float
    suggested_work_group_id: int | None = None
    suggested_work_group_name: str = ""
    market_justification: str = ""
    price_source_notes: str = ""
    confidence_level: float = 0.0


class LLMBatchPricedItem(BaseModel):
    """One item in the batch pricing GPT output."""

    source_text: str
    name: str
    unit: str
    labour_cost: float = Field(ge=0)
    material_cost: float = Field(ge=0)
    other_cost: float = Field(ge=0)
    work_days: float = Field(ge=0)
    qty_for_norm: float = Field(gt=0, default=1.0)
    suggested_work_group_id: int | None = None
    suggested_work_group_name: str = ""
    market_justification: str = ""
    price_source_notes: str = ""
    confidence_level: float = Field(ge=0, le=1, default=0.7)


class LLMBatchPriceOutput(BaseModel):
    items: list[LLMBatchPricedItem]


class PreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_text: str
    service_mode: str = "mock"
    input_mode: str = "raw"
    review_summary: PreviewReviewSummary
    matched_rows: list[PreviewMatchedRow]
    unmatched_items: list[PreviewUnmatchedItem]
    custom_priced_rows: list[CustomPricedRow] = Field(default_factory=list)
    assumptions: list[PreviewAssumption]
    review_queue: list[PreviewReviewQueueItem] = Field(default_factory=list)
    coverage_prompts: list[PreviewCoveragePrompt] = Field(default_factory=list)
    telemetry: PreviewTelemetry = Field(default_factory=PreviewTelemetry)
    warnings: list[str] = Field(default_factory=list)
    coverage_summary: PreviewCoverageSummary = Field(
        default_factory=PreviewCoverageSummary
    )
    error_text: str = ""


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str
    category: str
    prompt: str


class PromptTemplatesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    templates: list[PromptTemplate]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str
    mode: str


class LLMPreviewMatchedRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    INSIDEQUOTESGUID: str
    AREA: str | None = ""
    QUANTITY: float | None = None
    PROFIT: float | None = None
    LabourMarkup: float | None = None
    MaterialMarkup: float | None = None
    Confidence: float | None = None
    NeedsReview: bool | None = None
    ReviewReason: str | None = None


class LLMPreviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched_rows: list[LLMPreviewMatchedRow]
    unmatched_items: list[PreviewUnmatchedItem]
    assumptions: list[PreviewAssumption]


class IntakeAssetPagePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    text_excerpt: str = ""
    text_length: int = 0
    preview_url: str = ""


class IntakeAssetSheetExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    row_count_excerpt: int = 0
    recognized_schedule: bool = False
    rows: list[list[str]] = Field(default_factory=list)
    excerpt: str = ""


class IntakeAssetExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: int
    filename: str
    asset_type: str
    checksum: str
    extraction_status: str
    document_excerpt: str = ""
    warnings: list[str] = Field(default_factory=list)
    page_count: int = 0
    sheet_count: int = 0
    page_previews: list[IntakeAssetPagePreview] = Field(default_factory=list)
    sheets: list[IntakeAssetSheetExcerpt] = Field(default_factory=list)
    table_excerpt: str = ""
    pdf_base64: str | None = None


class IntakeTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    message: str = ""
    structured_delta: dict[str, object] = Field(default_factory=dict)
    created_at: str = ""


class ProjectBriefFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    source: str = "ai"
    confidence: str = "medium"
    uncertain: bool = False


class ProjectBriefSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str = ""
    facts: list[ProjectBriefFact] = Field(default_factory=list)


class IntakeQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    reason: str
    required: bool = True


class ProjectBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_overview: ProjectBriefSection
    existing_layout: ProjectBriefSection
    proposed_layout_scope: ProjectBriefSection
    areas_dimensions: ProjectBriefSection
    structural: ProjectBriefSection
    extension_roof_external: ProjectBriefSection
    internal_build: ProjectBriefSection
    mep: ProjectBriefSection
    finishes: ProjectBriefSection
    site_conditions: ProjectBriefSection
    client_supply_vs_combit_supply: ProjectBriefSection
    missing_rfi: ProjectBriefSection
    source_documents: list[dict[str, object]] = Field(default_factory=list)
    confidence_summary: dict[str, object] = Field(default_factory=dict)


class IntakeChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: int
    quote_id: int
    knowledge_version: str = ""
    current_project_brief_json: dict[str, object] = Field(default_factory=dict)
    assets: list[IntakeAssetExcerpt] = Field(default_factory=list)
    turns: list[IntakeTurn] = Field(default_factory=list)
    message: str = Field(min_length=1)


class IntakeChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_message: str
    updated_project_brief_json: ProjectBriefV1
    questions: list[IntakeQuestion] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    ready_to_finalize: bool = False
    model_name: str = ""


class IntakeFinalizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: int
    quote_id: int
    knowledge_version: str = ""
    current_project_brief_json: dict[str, object] = Field(default_factory=dict)
    assets: list[IntakeAssetExcerpt] = Field(default_factory=list)
    turns: list[IntakeTurn] = Field(default_factory=list)


class IntakeFinalizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_brief_json: ProjectBriefV1
    structured_scope_markdown: str
    final_missing_items: list[str] = Field(default_factory=list)
    final_warnings: list[str] = Field(default_factory=list)
    handoff_readiness: bool = False
    model_name: str = ""


class LLMIntakeChatOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_message: str
    updated_project_brief_json: ProjectBriefV1
    questions: list[IntakeQuestion] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    ready_to_finalize: bool = False


class LLMIntakeFinalizeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_brief_json: ProjectBriefV1
    structured_scope_markdown: str
    final_missing_items: list[str] = Field(default_factory=list)
    final_warnings: list[str] = Field(default_factory=list)
    handoff_readiness: bool = False


# ---------------------------------------------------------------------------
# Custom Work Creator chat schemas
# ---------------------------------------------------------------------------


class CustomWorkChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str  # "user" | "assistant"
    content: str


class CatalogWorkGroupContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str


class CatalogLabourRateContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    name: str
    net_rate: float


class CatalogContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_groups: list[CatalogWorkGroupContext] = Field(default_factory=list)
    labour_rates: list[CatalogLabourRateContext] = Field(default_factory=list)


class CustomWorkProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    unit: str  # m2 | lm | m3 | each | nr
    labour_cost: float
    material_cost: float
    other_cost: float
    work_days: float
    qty_for_norm: float
    suggested_work_group_id: int
    suggested_work_group_name: str
    market_justification: str
    price_source_notes: str  # explicit source attribution per cost line
    confidence_level: float  # 0.0–1.0


class CustomWorkChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[CustomWorkChatMessage] = Field(min_length=1)
    catalog_context: CatalogContext


class CustomWorkChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    status: str  # "chatting" | "proposing"
    work_proposal: CustomWorkProposal | None = None
    model_name: str = ""


class LLMCustomWorkOutput(BaseModel):
    """Internal structured output model used by responses.parse — not exposed in the API."""

    model_config = ConfigDict(extra="forbid")

    message: str
    status: str  # "chatting" | "proposing"
    work_proposal: CustomWorkProposal | None = None
    handoff_readiness: bool = False


# ── Batch Corrections ────────────────────────────────────────────────────────


class ItemContext(BaseModel):
    """A current quote item provided as context for the AI corrections parser."""

    model_config = ConfigDict(extra="forbid")

    id: int
    ref: str  # excel ref code, e.g. "1.1.4"
    display_index: int  # 1-based position in quote
    name: str  # effective display name
    area: str
    quantity: str


class BatchCorrectionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    items: list[ItemContext]


class CorrectionPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: int
    ref: str
    name: str  # current item name (for UI diff display)
    field: str  # which field is being changed
    old_value: str
    new_value: str


class BatchCorrectionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patches: list[CorrectionPatch]
    unresolved: list[str]  # correction clauses the AI could not map to any item
    service_mode: str  # "real" | "mock"


class LLMBatchCorrectionsOutput(BaseModel):
    """Internal structured output model for the corrections parser — not exposed in API."""

    model_config = ConfigDict(extra="forbid")

    patches: list[CorrectionPatch]
    unresolved: list[str]
