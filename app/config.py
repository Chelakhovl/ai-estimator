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
    service_api_key: str

    @property
    def preview_mode(self) -> str:
        return "llm" if self.openai_api_key and self.openai_model else "mock"



def get_settings() -> Settings:
    return Settings(
        app_env=os.getenv("APP_ENV", "local"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", ""),
        service_api_key=os.getenv("SERVICE_API_KEY", ""),
    )


settings = get_settings()
