"""Tests for ONNX/OpenCV detector inference, YOLO ONNX adapter, EXIF GPS, and pinhole raycasting."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

from backend.cv_engine.detector import SARDetector
from backend.cv_engine.exif_geo import ExifTelemetry, project_pixel_to_wgs84, read_exif_telemetry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"


def test_exif_extraction_from_synthetic_scene() -> None:
    img_path = SAMPLE_DIR / "sar_wilderness_clearing.jpg"
    assert img_path.exists()
    tel = read_exif_telemetry(img_path)

    assert tel.has_gps_exif is True
    assert tel.synthetic_flag is True
    assert tel.latitude_deg == pytest.approx(47.45128, abs=1e-4)
    assert tel.longitude_deg == pytest.approx(-121.41264, abs=1e-4)
    assert tel.altitude_m == pytest.approx(115.0, abs=0.5)
    assert tel.heading_deg == pytest.approx(35.0, abs=0.5)
    assert tel.gimbal_pitch_deg == pytest.approx(-90.0, abs=0.1)
    assert tel.gsd_cm_px is not None and 4.0 < tel.gsd_cm_px < 15.0


def test_pinhole_raycast_nadir_center_and_cardinals() -> None:
    tel = ExifTelemetry(
        has_gps_exif=True,
        latitude_deg=45.0,
        longitude_deg=-120.0,
        altitude_m=100.0,
        altitude_source="test",
        heading_deg=0.0,
        gimbal_pitch_deg=-90.0,
        hfov_deg=90.0,  # tan(45) = 1 -> half-width ground offset == altitude_m
    )
    W, H = 1000, 1000

    # Image center -> zero North/East offset
    c = project_pixel_to_wgs84(500.0, 500.0, W, H, tel)
    assert c is not None
    assert c.north_offset_m == pytest.approx(0.0, abs=1e-3)
    assert c.east_offset_m == pytest.approx(0.0, abs=1e-3)
    assert c.wgs84_lat == pytest.approx(45.0, abs=1e-6)
    assert c.wgs84_lon == pytest.approx(-120.0, abs=1e-6)

    # Right edge (px=1000, py=500) at HFOV=90 deg, heading=0 -> +100m East
    r = project_pixel_to_wgs84(1000.0, 500.0, W, H, tel)
    assert r is not None
    assert r.east_offset_m == pytest.approx(100.0, abs=0.5)
    assert r.north_offset_m == pytest.approx(0.0, abs=0.5)
    assert r.wgs84_lon > -120.0

    # Top edge (px=500, py=0) at heading=0 -> +100m North
    top = project_pixel_to_wgs84(500.0, 0.0, W, H, tel)
    assert top is not None
    assert top.north_offset_m == pytest.approx(100.0, abs=0.5)
    assert top.wgs84_lat > 45.0

    # Heading = 90 deg East -> image top now points East (+100m East, ~0m North)
    tel_east = ExifTelemetry(
        has_gps_exif=True,
        latitude_deg=45.0,
        longitude_deg=-120.0,
        altitude_m=100.0,
        altitude_source="test",
        heading_deg=90.0,
        gimbal_pitch_deg=-90.0,
        hfov_deg=90.0,
    )
    top_east = project_pixel_to_wgs84(500.0, 0.0, W, H, tel_east)
    assert top_east is not None
    assert top_east.east_offset_m == pytest.approx(100.0, abs=0.5)
    assert top_east.north_offset_m == pytest.approx(0.0, abs=0.5)


def test_pinhole_raycast_oblique_pitch() -> None:
    tel = ExifTelemetry(
        has_gps_exif=True,
        latitude_deg=45.0,
        longitude_deg=-120.0,
        altitude_m=100.0,
        altitude_source="test",
        heading_deg=0.0,
        gimbal_pitch_deg=-45.0,
        hfov_deg=60.0,
    )
    # Center pixel at -45 deg pitch looks forward 100m North at 100m AGL
    proj = project_pixel_to_wgs84(500.0, 500.0, 1000, 1000, tel)
    assert proj is not None
    assert proj.north_offset_m == pytest.approx(100.0, abs=0.5)
    assert proj.slant_range_m == pytest.approx(100.0 * math.sqrt(2.0), abs=0.5)


def test_detector_real_inference_and_blank_frame_rejection() -> None:
    det = SARDetector(MODELS_DIR, "sarvision_micro_detector.onnx")
    assert det.model is not None
    assert det.model.kind == "patch_cnn"

    # Uniform blank green frame must produce ZERO detections (proving no hardcoded outputs)
    blank = np.full((640, 640, 3), (35, 80, 40), dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", blank)
    assert ok
    blank_res = det.analyze_image(enc.tobytes(), conf_threshold=0.45, detector_mode="fusion")
    assert len(blank_res.detections) == 0

    # Real inference on wilderness sample scene produces geotagged detections
    res = det.analyze_image(SAMPLE_DIR / "sar_wilderness_clearing.jpg", conf_threshold=0.45, detector_mode="fusion")
    assert res.image_width == 1920 and res.image_height == 1280
    assert len(res.tiles) == 12
    assert len(res.detections) >= 3
    assert all(d.geo is not None for d in res.detections)


def test_yolo_onnx_adapter_auto_loading(tmp_path: Path) -> None:
    """Verify automatic YOLOv8-shaped ONNX model loading & inference pipeline."""

    class DummyYoloLike(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Output shape (1, 8, 16) -> (4 box coords + 4 SAR classes) x 16 anchors
            b = x.shape[0]
            anchor_zero = x[:, 0, 0, :16].sum() * 0.0
            out = torch.zeros((b, 8, 16), dtype=torch.float32) + anchor_zero
            out[:, 0, 0] = 320.0
            out[:, 1, 0] = 320.0
            out[:, 2, 0] = 50.0
            out[:, 3, 0] = 60.0
            out[:, 4, 0] = 0.91  # class 0
            return out

    yolo_onnx = tmp_path / "yolov8_sar_custom.onnx"
    torch.onnx.export(
        DummyYoloLike(),
        torch.randn(1, 3, 640, 640),
        str(yolo_onnx),
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(yolo_onnx)))

    det = SARDetector(tmp_path, "yolov8_sar_custom.onnx")
    assert det.model is not None
    assert det.model.kind == "yolo"

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", frame)
    assert ok
    res = det.analyze_image(enc.tobytes(), conf_threshold=0.50, detector_mode="onnx")
    assert len(res.detections) == 1
    assert res.detections[0].box.source == "onnx-yolo"
