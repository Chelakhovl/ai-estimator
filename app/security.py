from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="x-api-key")) -> None:
    settings = get_settings()

    if not settings.service_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SERVICE_API_KEY is not configured on the server.",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.service_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
