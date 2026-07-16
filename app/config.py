from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str
    app_host: str
    app_port: int
    openai_api_key: str
    openai_model: str
    openai_intake_model: str
    openai_intake_fast_model: str
    openai_timeout_seconds: float
    openai_request_max_attempts: int
    service_api_key: str

    @property
    def preview_mode(self) -> str:
        return "llm" if self.openai_api_key and self.openai_model else "mock"

    @property
    def allow_mock_fallback(self) -> bool:
        return self.app_env.lower() in {"local", "test"}


def get_settings() -> Settings:
    return Settings(
        app_env=os.getenv("APP_ENV", "local"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8001")),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", ""),
        openai_intake_model=os.getenv("OPENAI_INTAKE_MODEL", "gpt-4o"),
        openai_intake_fast_model=os.getenv("OPENAI_INTAKE_FAST_MODEL", "gpt-4o"),
        openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45")),
        openai_request_max_attempts=max(
            int(os.getenv("OPENAI_REQUEST_MAX_ATTEMPTS", "2")), 1
        ),
        service_api_key=os.getenv("SERVICE_API_KEY", ""),
    )


settings = get_settings()
