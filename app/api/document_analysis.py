from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.security import require_api_key
from app.services.document_chat_service import chat_about_document
from app.services.document_enrichment_service import enrich_page_with_vision
from app.services.document_synthesis_service import synthesize_document_to_markdown

router = APIRouter(prefix="/v1/document-analysis", tags=["document-analysis"])


class EnrichPageRequest(BaseModel):
    page_image_base64: str
    page_number: int
    source_file: str
    page_type: str = "other"
    existing_rooms: list[dict] = []
    existing_notes: list[str] = []


class EnrichedRoom(BaseModel):
    name: str
    floor_level: str | None = None
    width_m: float | None = None
    length_m: float | None = None


class EnrichedDimension(BaseModel):
    element: str
    width_m: float | None = None
    length_m: float | None = None


class EnrichedWorkItem(BaseModel):
    trade: str
    description: str


class EnrichPageResponse(BaseModel):
    page_type: str
    description: str = ""
    rooms: list[EnrichedRoom]
    dimensions: list[EnrichedDimension]
    work_items: list[EnrichedWorkItem]
    structural_notes: list[str]
    uncertainties: list[str]
    model_name: str
    used_vision: bool
    fallback_reason: str | None = None


def _coerce_rooms(raw: list) -> list[EnrichedRoom]:
    rooms: list[EnrichedRoom] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        rooms.append(EnrichedRoom(
            name=str(item.get("name", "")).strip(),
            floor_level=item.get("floor_level") or None,
            width_m=item.get("width_m") or None,
            length_m=item.get("length_m") or None,
        ))
    return [r for r in rooms if r.name]


def _coerce_dimensions(raw: list) -> list[EnrichedDimension]:
    dims: list[EnrichedDimension] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        dims.append(EnrichedDimension(
            element=str(item.get("element", "")).strip(),
            width_m=item.get("width_m") or None,
            length_m=item.get("length_m") or None,
        ))
    return [d for d in dims if d.element]


def _coerce_work_items(raw: list) -> list[EnrichedWorkItem]:
    items: list[EnrichedWorkItem] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        items.append(EnrichedWorkItem(
            trade=str(item.get("trade", "general")).strip(),
            description=str(item.get("description", "")).strip(),
        ))
    return [i for i in items if i.description]


class ChatTurn(BaseModel):
    role: str
    message: str


class ChatRequest(BaseModel):
    context_markdown: str
    turns: list[ChatTurn] = []
    message: str


class ChatResponse(BaseModel):
    assistant_message: str
    updated_markdown: str | None = None
    model_name: str


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> ChatResponse:
    result = chat_about_document(
        context_markdown=payload.context_markdown,
        turns=[t.model_dump() for t in payload.turns],
        message=payload.message,
    )
    return ChatResponse(
        assistant_message=result.get("assistant_message", ""),
        updated_markdown=result.get("updated_markdown"),
        model_name=result.get("model_name", ""),
    )


class PageBundle(BaseModel):
    page_number: int
    page_type: str = "other"
    source_file: str = ""
    vision_description: str = ""
    programmatic_text: str = ""
    tables_text: str = ""


class SynthesizeRequest(BaseModel):
    page_bundles: list[PageBundle]


class SynthesizeResponse(BaseModel):
    markdown: str
    model_name: str


@router.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(
    payload: SynthesizeRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> SynthesizeResponse:
    result = synthesize_document_to_markdown(
        page_bundles=[b.model_dump() for b in payload.page_bundles],
    )
    return SynthesizeResponse(
        markdown=result.get("markdown", ""),
        model_name=result.get("model_name", ""),
    )


@router.post("/enrich-page", response_model=EnrichPageResponse)
def enrich_page(
    payload: EnrichPageRequest,
    _authorized: Annotated[None, Depends(require_api_key)],
) -> EnrichPageResponse:
    raw = enrich_page_with_vision(
        page_image_base64=payload.page_image_base64,
        page_number=payload.page_number,
        source_file=payload.source_file,
        page_type=payload.page_type,
        existing_rooms=payload.existing_rooms,
        existing_notes=payload.existing_notes,
    )
    return EnrichPageResponse(
        page_type=raw.get("page_type", "other"),
        description=str(raw.get("description") or ""),
        rooms=_coerce_rooms(raw.get("rooms", [])),
        dimensions=_coerce_dimensions(raw.get("dimensions", [])),
        work_items=_coerce_work_items(raw.get("work_items", [])),
        structural_notes=[str(n) for n in raw.get("structural_notes", []) if n],
        uncertainties=[str(u) for u in raw.get("uncertainties", []) if u],
        model_name=raw.get("model_name", ""),
        used_vision=raw.get("used_vision", False),
        fallback_reason=raw.get("fallback_reason"),
    )
