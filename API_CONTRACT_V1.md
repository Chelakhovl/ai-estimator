# AI Service API Contract v1

This document fixes the runtime contract between:
- Power Automate flow `AI_CreateEstimatePreviewFromText`
- Python AI service `POST /v1/estimate/preview`
- Canvas App preview parsing in `Quote Manager Screen`

It is the source of truth for request/response payloads during the safe MVP phase.

## 1. Architecture boundary

### Canvas -> Flow
Canvas sends:
- `QuoteGuid`
- `Prompt`
- `CandidateRowsJson` as a JSON string

### Flow -> Python service
Flow sends:
- normal JSON object
- `candidate_rows` as a real array, not a string

### Python service -> Flow
Python service returns:
- normal JSON object
- `matched_rows`, `unmatched_items`, `assumptions` as real arrays

### Flow -> Canvas
Flow returns:
- `summary_text`
- `error_text`
- `matched_rows_json`
- `unmatched_items_json`
- `assumptions_json`

Important:
- Python service should return arrays
- Flow is responsible for converting arrays into JSON strings for Canvas
- Canvas already expects string fields and uses `ParseJSON(...)`

## 2. HTTP endpoint

### Method
`POST`

### URL
`/v1/estimate/preview`

### Headers
`Content-Type: application/json`

Optional for later:
`x-api-key: <SERVICE_API_KEY>`

## 3. Flow -> Python request contract

### JSON schema

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["quote_guid", "prompt", "candidate_rows"],
  "properties": {
    "quote_guid": {
      "type": "string"
    },
    "prompt": {
      "type": "string",
      "minLength": 1
    },
    "candidate_rows": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "INSIDEQUOTESGUID",
          "WorkName"
        ],
        "properties": {
          "INSIDEQUOTESGUID": { "type": "string" },
          "WORKGUID": { "type": ["string", "null"] },
          "WorkGroupGUID": { "type": ["string", "null"] },
          "WorkName": { "type": "string" },
          "Unit": { "type": ["string", "null"] },
          "WorkLabourCost": { "type": "number" },
          "WorkMatCost": { "type": "number" },
          "WorkOtherCost": { "type": "number" },
          "WorkQTYforNorm": { "type": "number", "exclusiveMinimum": 0 },
          "WorkToolsHire": { "type": "number" },
          "WorkDays": { "type": "number" },
          "Work8Skips": { "type": "number" },
          "Work12Skips": { "type": "number" },
          "AREA": { "type": ["string", "null"] },
          "QUANTITY": { "type": ["number", "null"] },
          "PROFIT": { "type": "number" },
          "LabourMarkup": { "type": "number" },
          "MaterialMarkup": { "type": "number" }
        }
      }
    }
  }
}
```

### Real request example

```json
{
  "quote_guid": "6f2d6d9d-1111-2222-3333-444444444444",
  "prompt": "Kitchen renovation, paint walls 120 m2, remove old tiles 20 m2, install skirting 30 m",
  "candidate_rows": [
    {
      "INSIDEQUOTESGUID": "11111111-2222-3333-4444-555555555555",
      "WORKGUID": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
      "WorkGroupGUID": "99999999-8888-7777-6666-555555555555",
      "WorkName": "Paint walls",
      "Unit": "m2",
      "WorkLabourCost": 10,
      "WorkMatCost": 4,
      "WorkOtherCost": 1,
      "WorkQTYforNorm": 1,
      "WorkToolsHire": 0,
      "WorkDays": 0.5,
      "Work8Skips": 0,
      "Work12Skips": 0,
      "AREA": "",
      "QUANTITY": 0,
      "PROFIT": 20,
      "LabourMarkup": 15,
      "MaterialMarkup": 10
    },
    {
      "INSIDEQUOTESGUID": "66666666-7777-8888-9999-aaaaaaaaaaaa",
      "WORKGUID": "bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
      "WorkGroupGUID": "99999999-8888-7777-6666-555555555555",
      "WorkName": "Remove old tiles",
      "Unit": "m2",
      "WorkLabourCost": 8,
      "WorkMatCost": 0,
      "WorkOtherCost": 2,
      "WorkQTYforNorm": 1,
      "WorkToolsHire": 0,
      "WorkDays": 0.4,
      "Work8Skips": 0,
      "Work12Skips": 0,
      "AREA": "",
      "QUANTITY": 0,
      "PROFIT": 20,
      "LabourMarkup": 15,
      "MaterialMarkup": 10
    }
  ]
}
```

## 4. Python service -> Flow response contract

### JSON schema

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "summary_text",
    "matched_rows",
    "unmatched_items",
    "assumptions",
    "error_text"
  ],
  "properties": {
    "summary_text": {
      "type": "string"
    },
    "matched_rows": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "INSIDEQUOTESGUID",
          "WorkName",
          "QUANTITY",
          "PROFIT",
          "LabourMarkup",
          "MaterialMarkup",
          "WorkQTYforNorm",
          "ClientCostPerUnit",
          "ClientTotalCost",
          "Confidence",
          "NeedsReview"
        ],
        "properties": {
          "INSIDEQUOTESGUID": { "type": "string" },
          "WORKGUID": { "type": ["string", "null"] },
          "WorkName": { "type": "string" },
          "Unit": { "type": ["string", "null"] },
          "AREA": { "type": "string" },
          "QUANTITY": { "type": "number", "exclusiveMinimum": 0 },
          "PROFIT": { "type": "number" },
          "LabourMarkup": { "type": "number" },
          "MaterialMarkup": { "type": "number" },
          "WorkLabourCost": { "type": "number" },
          "WorkMatCost": { "type": "number" },
          "WorkOtherCost": { "type": "number" },
          "WorkQTYforNorm": { "type": "number", "exclusiveMinimum": 0 },
          "ClientCostPerUnit": { "type": "number", "minimum": 0 },
          "ClientTotalCost": { "type": "number", "minimum": 0 },
          "Confidence": { "type": "number", "minimum": 0, "maximum": 1 },
          "NeedsReview": { "type": "boolean" }
        }
      }
    },
    "unmatched_items": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["source_text", "reason"],
        "properties": {
          "source_text": { "type": "string" },
          "reason": { "type": "string" }
        }
      }
    },
    "assumptions": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["text"],
        "properties": {
          "text": { "type": "string" }
        }
      }
    },
    "error_text": {
      "type": "string"
    }
  }
}
```

### Real response example

```json
{
  "summary_text": "Matched 2 works, 1 items need review.",
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
    },
    {
      "INSIDEQUOTESGUID": "66666666-7777-8888-9999-aaaaaaaaaaaa",
      "WORKGUID": "bbbbbbbb-cccc-dddd-eeee-ffffffffffff",
      "WorkName": "Remove old tiles",
      "Unit": "m2",
      "AREA": "Kitchen",
      "QUANTITY": 20,
      "PROFIT": 20,
      "LabourMarkup": 15,
      "MaterialMarkup": 10,
      "WorkLabourCost": 8,
      "WorkMatCost": 0,
      "WorkOtherCost": 2,
      "WorkQTYforNorm": 1,
      "ClientCostPerUnit": 13.44,
      "ClientTotalCost": 268.8,
      "Confidence": 0.81,
      "NeedsReview": false
    }
  ],
  "unmatched_items": [
    {
      "source_text": "install skirting 30 m",
      "reason": "No confident match found in candidate quote rows."
    }
  ],
  "assumptions": [
    {
      "text": "Wall painting interpreted as internal wall painting."
    }
  ],
  "error_text": ""
}
```

## 5. Flow -> Canvas response contract

Canvas currently expects these fields from the flow:

```json
{
  "summary_text": "Matched 2 works, 1 items need review.",
  "matched_rows_json": "[{...}]",
  "unmatched_items_json": "[{...}]",
  "assumptions_json": "[{...}]",
  "error_text": ""
}
```

Important:
- `matched_rows_json` must be a string containing serialized JSON array
- `unmatched_items_json` must be a string containing serialized JSON array
- `assumptions_json` must be a string containing serialized JSON array

This is required because Canvas currently parses these values using `ParseJSON(...)`.

## 6. Required field mapping to current Canvas parsing

Current Canvas preview parsing requires these matched row fields:
- `INSIDEQUOTESGUID`
- `WORKGUID`
- `WorkName`
- `Unit`
- `AREA`
- `QUANTITY`
- `PROFIT`
- `LabourMarkup`
- `MaterialMarkup`
- `WorkLabourCost`
- `WorkMatCost`
- `WorkOtherCost`
- `WorkQTYforNorm`
- `Confidence`
- `NeedsReview`

Optional but already useful:
- `ClientCostPerUnit`
- `ClientTotalCost`

Current Canvas unmatched items parsing requires:
- `source_text`
- `reason`

Current Canvas assumptions parsing requires:
- `text`

## 7. Deterministic calculation rule

These values must be calculated in Python code, not trusted to the model:

```text
ClientCostPerUnit =
((WorkLabourCost * (1 + LabourMarkup / 100) +
(WorkMatCost + WorkOtherCost) * (1 + MaterialMarkup / 100)) *
(1 + PROFIT / 100)) / WorkQTYforNorm

ClientTotalCost =
QUANTITY * ClientCostPerUnit
```

## 8. Validation rules

The Python service must enforce:
- `INSIDEQUOTESGUID` must exist in `candidate_rows`
- `QUANTITY > 0`
- `WorkQTYforNorm > 0`
- `PROFIT` within `0..200`
- `LabourMarkup` within `0..200`
- `MaterialMarkup` within `0..100`
- `AREA` length should stay within current Dataverse limit

## 9. MVP rule

For safe MVP:
- service may only match against provided `candidate_rows`
- service must not invent new `INSIDEQUOTESGUID`
- service must not create new work dictionary entries
- unmatched items stay in preview only

