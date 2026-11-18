"""SQLAlchemy 2.0 ORM models for SARVision analysis history and human verification."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRun(Base):
    """Stores one aerial SAR imagery inference execution and telemetry snapshot."""

    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)

    tile_size: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    tile_overlap: Mapped[float] = mapped_column(Float, nullable=False, default=0.20)
    tile_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    tiles_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    conf_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.45)
    nms_iou_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.40)
    detector_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="fusion")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="sarvision_micro_detector.onnx")
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    raw_tile_detections_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    has_gps_exif: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    uav_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    uav_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    uav_alt_m: Mapped[float] = mapped_column(Float, nullable=False, default=120.0)
    gimbal_pitch_deg: Mapped[float] = mapped_column(Float, nullable=False, default=-90.0)
    heading_deg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hfov_deg: Mapped[float] = mapped_column(Float, nullable=False, default=64.0)
    gsd_cm_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    swath_width_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    swath_height_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    synthetic_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    detections: Mapped[list[DetectionRecord]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="DetectionRecord.confidence.desc()",
    )


class DetectionRecord(Base):
    """Stores an individual SAR target detection, geolocation, and operator verification status."""

    __tablename__ = "detection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    onnx_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    opencv_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    detector_source: Mapped[str] = mapped_column(String(64), nullable=False, default="fusion")
    color_signature: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    tile_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    wgs84_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    wgs84_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    north_offset_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    east_offset_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    slant_range_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    verification_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    verified_class_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operator_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    analysis: Mapped[AnalysisRun] = relationship(back_populates="detections")
