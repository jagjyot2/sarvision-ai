"""Pydantic V2 request and response schemas for SARVision API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TileRect(BaseModel):
    tile_id: int
    row: int
    col: int
    x_offset: int
    y_offset: int
    width: int
    height: int


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    analysis_id: str
    class_id: int
    class_name: str
    confidence: float
    onnx_score: float
    opencv_score: float
    detector_source: str
    color_signature: str
    x1: float
    y1: float
    x2: float
    y2: float
    tile_id: int | None = None
    wgs84_lat: float | None = None
    wgs84_lon: float | None = None
    north_offset_m: float | None = None
    east_offset_m: float | None = None
    slant_range_m: float | None = None
    verification_status: str
    verified_class_name: str | None = None
    operator_notes: str | None = None
    updated_at: datetime


class AnalysisDetailOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    image_width: int
    image_height: int
    tile_size: int
    tile_overlap: float
    tile_count: int
    tiles: list[TileRect] = Field(default_factory=list)
    conf_threshold: float
    nms_iou_threshold: float
    detector_mode: str
    model_name: str
    processing_time_ms: float
    raw_tile_detections_count: int
    has_gps_exif: bool
    uav_lat: float | None = None
    uav_lon: float | None = None
    uav_alt_m: float
    gimbal_pitch_deg: float
    heading_deg: float
    hfov_deg: float
    gsd_cm_px: float | None = None
    swath_width_m: float | None = None
    swath_height_m: float | None = None
    synthetic_flag: bool
    batch_id: str | None = None
    created_at: datetime
    image_url: str
    detections: list[DetectionOut] = Field(default_factory=list)


class AnalysisSummaryOut(BaseModel):
    id: str
    filename: str
    image_width: int
    image_height: int
    detector_mode: str
    model_name: str
    processing_time_ms: float
    has_gps_exif: bool
    uav_lat: float | None = None
    uav_lon: float | None = None
    detection_count: int
    confirmed_count: int
    escalated_count: int
    false_positive_count: int
    pending_count: int
    batch_id: str | None = None
    created_at: datetime


class ReanalyzeRequest(BaseModel):
    tile_size: int = Field(default=640, ge=128, le=2048)
    tile_overlap: float = Field(default=0.20, ge=0.0, lt=0.8)
    conf_threshold: float = Field(default=0.45, ge=0.05, le=0.99)
    nms_iou_threshold: float = Field(default=0.40, ge=0.05, le=0.95)
    detector_mode: Literal["fusion", "onnx", "opencv"] = "fusion"
    model_name: str | None = None
    agl_m: float | None = Field(default=None, gt=1.0, lt=15000.0)
    gimbal_pitch_deg: float | None = Field(default=None, ge=-90.0, le=-5.0)
    hfov_deg: float | None = Field(default=None, ge=10.0, le=150.0)
    heading_deg: float | None = Field(default=None, ge=0.0, le=360.0)


class VerifyDetectionRequest(BaseModel):
    verification_status: Literal["pending", "confirmed", "false_positive", "escalated"]
    verified_class_name: Literal["person", "life_raft", "vehicle", "distress_marker"] | None = None
    operator_notes: str | None = Field(default=None, max_length=1000)


class BatchAnalyzeResponse(BaseModel):
    batch_id: str
    total_files: int
    total_detections: int
    analyses: list[AnalysisDetailOut]


class EvaluateRequest(BaseModel):
    conf_threshold: float = Field(default=0.45, ge=0.05, le=0.99)
    nms_iou_threshold: float = Field(default=0.40, ge=0.05, le=0.95)
    iou_match_threshold: float = Field(default=0.50, ge=0.10, le=0.95)
    tile_size: int = Field(default=640, ge=128, le=2048)
    tile_overlap: float = Field(default=0.20, ge=0.0, lt=0.8)
    detector_mode: Literal["fusion", "onnx", "opencv"] = "fusion"
    model_name: str | None = None
    analysis_ids: list[str] | None = None
