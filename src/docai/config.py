"""Runtime configuration, read from environment variables prefixed ``DOCAI_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables for the pipeline, the model and the service.

    Every field can be overridden with an environment variable, e.g.
    ``DOCAI_MODEL_PATH=/models/invoice-layoutlmv3``.
    """

    model_config = SettingsConfigDict(env_prefix="DOCAI_", env_file=".env", extra="ignore")

    # --- Stage 1: ingestion & preprocessing -------------------------------
    pdf_dpi: int = Field(200, ge=72, le=600, description="Rasterisation DPI for PDFs.")
    max_pages: int = Field(20, ge=1, description="Pages beyond this are ignored.")
    deskew_max_angle: float = Field(15.0, gt=0, le=45, description="Larger estimated skews are ignored.")
    deskew_min_angle: float = Field(0.1, ge=0, description="Smaller estimated skews are not corrected.")
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    poppler_path: str | None = Field(None, description="Directory holding pdftoppm/pdfinfo, if not on PATH.")

    # --- Stage 2: OCR -----------------------------------------------------
    tesseract_cmd: str | None = Field(None, description="Path to the tesseract binary, if not on PATH.")
    tesseract_lang: str = "eng"
    tesseract_config: str = "--oem 3 --psm 3"
    ocr_min_confidence: float = Field(0.0, ge=0, le=100, description="Tesseract word confidence floor (0-100).")

    # --- Stage 3: model ---------------------------------------------------
    model_path: str = Field(
        "microsoft/layoutlmv3-base",
        description="Fine-tuned token-classification checkpoint (local dir or HF hub id).",
    )
    processor_path: str = "microsoft/layoutlmv3-base"
    device: str = Field("auto", description="'auto', 'cpu', 'cuda', 'cuda:0', 'mps' ...")
    max_seq_length: int = Field(512, ge=64, le=512)
    window_stride: int = Field(128, ge=0, description="Token overlap between windows on long pages.")

    # --- Stage 4: post-processing -----------------------------------------
    confidence_threshold: float = Field(0.70, ge=0, le=1)
    default_currency: str = Field("USD", min_length=3, max_length=3)
    day_first_dates: bool = Field(False, description="Read ambiguous 03/04/2024 as 3 April instead of March 4.")

    # --- Stage 5: service -------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: Path = Path("/tmp/docai/uploads")
    max_upload_mb: int = Field(25, ge=1)
    task_time_limit_s: int = Field(300, ge=10)
    result_ttl_s: int = Field(86_400, ge=60)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
