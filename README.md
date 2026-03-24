# Combit Estimator AI Service

Python service for the future AI layer of the Combit Estimator Power Platform solution.

## Purpose

This service sits behind Power Automate and owns the AI-specific logic that should not live inside Canvas formulas or large Power Automate expressions.

Planned responsibilities:
- accept quote context and prompt from Power Automate
- shortlist candidate quote rows
- validate and normalize preview input
- call the model later
- calculate deterministic cost fields before rows are written back
- return a strict preview contract to Power Platform

## Integration model

Runtime flow:
1. Canvas App collects prompt and candidate quote rows
2. Power Automate flow `AI_CreateEstimatePreviewFromText` calls this service over HTTP
3. This service returns structured preview JSON
4. Canvas shows preview to the user
5. Approved rows are written through the existing Dataverse bulk-update flow

## Current status

This repository now contains a **working v1 preview service skeleton** with:
- `GET /health`
- `POST /v1/estimate/preview`
- strict request/response schemas
- deterministic price calculation helpers
- candidate shortlist logic
- mock matching fallback
- a real LLM integration point via `LLMClient`
- validation of model-selected quote rows before totals are calculated

Current runtime behavior:
- if `OPENAI_API_KEY` and `OPENAI_MODEL` are not configured, the service runs in `mock` mode
- if they are configured, the service attempts the LLM path first
- if the LLM path fails, the service falls back to mock preview mode and records the fallback in assumptions

## API contract v1

### `GET /health`

Response example:

```json
{
  "status": "ok",
  "service": "combit-estimator-ai-service",
  "mode": "mock"
}
```

### `POST /v1/estimate/preview`

Request example:

```json
{
  "quote_guid": "6f2d6d9d-1111-2222-3333-444444444444",
  "prompt": "Kitchen renovation, paint walls 120 m2, remove old tiles 20 m2",
  "candidate_rows": [
    {
      "INSIDEQUOTESGUID": "11111111-2222-3333-4444-555555555555",
      "WORKGUID": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
      "WorkName": "Paint walls",
      "Unit": "m2",
      "WorkLabourCost": 10,
      "WorkMatCost": 4,
      "WorkOtherCost": 1,
      "WorkQTYforNorm": 1,
      "PROFIT": 20,
      "LabourMarkup": 15,
      "MaterialMarkup": 10
    }
  ]
}
```

Response example:

```json
{
  "summary_text": "Matched 1 works, 0 items need review.",
  "matched_rows": [
    {
      "INSIDEQUOTESGUID": "11111111-2222-3333-4444-555555555555",
      "WORKGUID": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
      "WorkName": "Paint walls",
      "Unit": "m2",
      "AREA": "Kitchen",
      "QUANTITY": 120,
      "PROFIT": 20,
      "LabourMarkup": 15,
      "MaterialMarkup": 10,
      "WorkLabourCost": 10,
      "WorkMatCost": 4,
      "WorkOtherCost": 1,
      "WorkQTYforNorm": 1,
      "ClientCostPerUnit": 18.45,
      "ClientTotalCost": 2214.0,
      "Confidence": 0.95,
      "NeedsReview": false
    }
  ],
  "unmatched_items": [],
  "assumptions": [],
  "error_text": ""
}
```

## Project structure

- `app/main.py`: FastAPI entrypoint
- `app/config.py`: environment/config loading
- `app/schemas.py`: request/response schemas
- `app/api/preview.py`: preview endpoint
- `app/api/health.py`: health endpoint
- `app/services/calculator.py`: deterministic pricing logic
- `app/services/normalizer.py`: prompt normalization helpers
- `app/services/candidate_shortlist.py`: shortlist builder
- `app/services/matcher.py`: preview orchestration with mock + LLM fallback
- `app/services/llm_client.py`: OpenAI Responses API integration point
- `app/services/preview_validator.py`: validation and materialization of LLM-selected rows
- `app/tests/`: tests

## Local run

```bash
uvicorn app.main:app --reload
```

## Next step

Wire Power Automate HTTP calls to this service, then tighten the model prompt and validation loop using real estimating examples.
