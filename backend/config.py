"""Configuration settings for SARVision AI."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application runtime configuration loaded from environment or .env."""

    model_config = SettingsConfigDict(
        env_prefix="SARVISION_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SARVision AI"
    env: str = "production"
    host: str = "127.0.0.1"
    port: int = 8000

    db_url: str = f"sqlite:///{(PROJECT_ROOT / 'sarvision.db').as_posix()}"

    model_dir: Path = PROJECT_ROOT / "models"
    onnx_model: str = "sarvision_micro_detector.onnx"
    tile_size: int = 640
    tile_overlap: float = 0.20
    conf_threshold: float = 0.45
    nms_iou_threshold: float = 0.40
    detector_mode: str = "fusion"

    default_agl_meters: float = 120.0
    default_gimbal_pitch_deg: float = -90.0
    default_hfov_deg: float = 64.0
    default_heading_deg: float = 0.0

    uploads_dir: Path = PROJECT_ROOT / "uploads"
    sample_data_dir: Path = PROJECT_ROOT / "sample_data"
    frontend_dist_dir: Path = PROJECT_ROOT / "frontend" / "dist"


def get_settings() -> Settings:
    """Return initialized application settings."""
    return Settings()
