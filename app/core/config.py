from functools import lru_cache
from typing import List, Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VaaniFlow AI"
    app_env: str = "development"
    app_host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("APP_HOST", "HOST"))
    app_port: int = Field(default=8000, validation_alias=AliasChoices("PORT", "APP_PORT"))
    debug: bool = True

    # CORS configuration (allows configurable frontend origin for Render/production)
    allowed_origins: Optional[str] = None
    frontend_url: Optional[str] = None

    # Google Gemini API
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"

    # ElevenLabs API
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_default_voice_id: Optional[str] = None

    # SQLite Database
    database_path: str = "data/vaaniflow.db"

    @property
    def cors_origins(self) -> List[str]:
        default_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ]
        if self.allowed_origins and self.allowed_origins.strip() == "*":
            return ["*"]

        origins = list(default_origins)
        if self.frontend_url and self.frontend_url.strip():
            url = self.frontend_url.strip().rstrip("/")
            if url and url not in origins:
                origins.append(url)
        if self.allowed_origins and self.allowed_origins.strip():
            for item in self.allowed_origins.split(","):
                clean = item.strip().rstrip("/")
                if clean and clean not in origins:
                    origins.append(clean)
        return origins

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
