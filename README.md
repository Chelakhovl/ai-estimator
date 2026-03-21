# Combit Estimator AI Service

Python service for the future AI layer of the Combit Estimator Power Platform solution.

## Purpose

This service will sit behind Power Automate and handle the AI-specific logic that we do not want to keep inside Canvas formulas or large Power Automate expressions.

Planned responsibilities:
- accept quote context and user prompt from Power Automate
- prepare prompt payloads for the model
- shortlist candidate works from the current quote/work dictionary
- validate model output against the allowed quote rows
- calculate deterministic cost fields before data is written back
- return a strict JSON preview contract to Power Platform

## Planned integration

Target runtime flow:
1. Canvas App collects prompt and candidate quote rows
2. Power Automate flow `AI_CreateEstimatePreviewFromText` calls this service over HTTP
3. This service returns structured preview JSON
4. Canvas shows preview to the user
5. Approved rows are written through existing Dataverse bulk-update logic

## Project structure

- `app/main.py`: FastAPI entrypoint
- `app/config.py`: environment/config loading
- `app/schemas.py`: request/response schemas
- `app/api/preview.py`: preview endpoint
- `app/api/health.py`: health endpoint
- `app/services/`: AI and calculation helpers
- `app/tests/`: service tests

## Local setup

Create a virtual environment and install dependencies manually after scaffold creation.

## Status

Scaffold only. Business logic has not been implemented yet.
