from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = Field(
        default="sk-proj-...",
        description="OpenAI API key for accessing GPT models",
    )

    openai_model_name: str = Field(
        default="gpt-4o-mini", description="OpenAI model name to use"
    )

    serper_api_key: str = Field(
        default="775...",
        description="Serper API key for web search functionality",
    )


settings = AppSettings()
