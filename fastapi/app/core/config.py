"""Configuration via pydantic-settings. All secrets from env / .env."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openai_api_key: str = Field(..., description="OpenAI API key")
    chat_model: str = Field(default="gpt-4o", description="Model for chat + vision")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Model for embeddings"
    )


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
