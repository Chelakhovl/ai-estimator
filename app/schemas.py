from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CandidateRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    INSIDEQUOTESGUID: str
    WORKGUID: str | None = None
    WorkGroupGUID: str | None = None
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


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote_guid: str
    prompt: str = Field(min_length=1)
    candidate_rows: list[CandidateRow] = Field(min_length=1)


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


class PreviewUnmatchedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_text: str
    reason: str


class PreviewAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class PreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_text: str
    matched_rows: list[PreviewMatchedRow]
    unmatched_items: list[PreviewUnmatchedItem]
    assumptions: list[PreviewAssumption]
    error_text: str = ""


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


class LLMPreviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched_rows: list[LLMPreviewMatchedRow]
    unmatched_items: list[PreviewUnmatchedItem]
    assumptions: list[PreviewAssumption]
