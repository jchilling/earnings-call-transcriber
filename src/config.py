"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/earnings_calls"
    database_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""

    # DeepSeek (for LLM speaker identification)
    deepseek_model: str = "deepseek-chat"
    deepseek_reasoner_model: str = "deepseek-reasoner"

    # HuggingFace (for pyannote speaker diarization)
    hf_token: str = ""

    # Whisper
    whisper_model: str = "large-v3"
    whisper_device: str = "cpu"
    whisper_use_local: bool = True

    # LLM
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Audio storage
    audio_storage_path: str = "./data/audio"
    transcript_storage_path: str = "./data/transcripts"

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
