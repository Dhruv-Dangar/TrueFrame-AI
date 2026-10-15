from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings validated via Pydantic."""

    app_name: str = "TrueFrame AI Backend"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    # Security & CORS
    allowed_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # File storage
    storage_path: Path = Path("storage/temp")
    max_image_size_mb: int = 15
    max_video_size_mb: int = 20
    max_video_duration_seconds: int = 15

    # Model settings
    model_dir: Path = Path("models/weights")
    active_model: str = "efficientnet_b0"
    inference_device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
