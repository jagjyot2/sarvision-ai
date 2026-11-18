"""Unit tests for ImageTiler, IoU calculation, seam suppression, and global NMS / WBF."""

from __future__ import annotations

import numpy as np
import pytest

from backend.cv_engine.tiler import (
    DetectionBox,
    ImageTiler,
    compute_iou,
    compute_iou_matrix,
    global_nms,
)


def test_tile_grid_covers_full_1920x1280_with_20pct_overlap() -> None:
    tiler = ImageTiler(tile_size=640, overlap=0.20)
    tiles = tiler.generate_tiles(1920, 1280)

    # Stride = 512 -> xs=[0, 512, 1024, 1280], ys=[0, 512, 640] => 4*3 = 12 tiles
    assert len(tiles) == 12
    assert tiles[0].x_offset == 0 and tiles[0].y_offset == 0
    assert tiles[-1].x_offset + tiles[-1].width == 1920
    assert tiles[-1].y_offset + tiles[-1].height == 1280

    # Verify every pixel in 1920x1280 is covered by at least one tile
    cov = np.zeros((1280, 1920), dtype=np.int32)
    for t in tiles:
        cov[t.y_offset : t.y_offset + t.height, t.x_offset : t.x_offset + t.width] += 1
    assert np.all(cov >= 1)
    assert np.any(cov >= 2)


def test_tile_grid_handles_small_image() -> None:
    tiler = ImageTiler(tile_size=640, overlap=0.20)
    tiles = tiler.generate_tiles(320, 240)
    assert len(tiles) == 1
    assert tiles[0].width == 320 and tiles[0].height == 240


def test_invalid_tiler_params_raise() -> None:
    with pytest.raises(ValueError):
        ImageTiler(tile_size=32, overlap=0.2)
    with pytest.raises(ValueError):
        ImageTiler(tile_size=640, overlap=0.95)


def test_iou_and_matrix_math() -> None:
    a = [0.0, 0.0, 10.0, 10.0]
    b = [5.0, 5.0, 15.0, 15.0]
    # Intersection = 25, union = 100 + 100 - 25 = 175
    assert compute_iou(a, b) == pytest.approx(25.0 / 175.0, rel=1e-5)
    assert compute_iou(a, [20.0, 20.0, 30.0, 30.0]) == 0.0

    mat = compute_iou_matrix(np.array([a, b]), np.array([b, a]))
    assert mat.shape == (2, 2)
    assert mat[0, 1] == pytest.approx(1.0)
    assert mat[1, 0] == pytest.approx(1.0)
    assert mat[0, 0] == pytest.approx(25.0 / 175.0, rel=1e-5)


def test_seam_fragment_suppression_and_wbf_merge() -> None:
    tiler = ImageTiler(tile_size=640, overlap=0.20)
    tiles = tiler.generate_tiles(1920, 1280)
    tile_left = tiles[0]   # x: 0..640
    tile_right = tiles[1]  # x: 512..1152

    # Clipped left-tile detection right against seam x=640: global [618, 300, 640, 355]
    det_clipped = DetectionBox(
        bbox=[618.0, 300.0, 640.0, 355.0],
        confidence=0.72,
        class_id=1,
        class_name="person",
    )
    # Full object seen inside overlapping right tile: local [103, 300, 143, 355] -> global [615, 300, 655, 355]
    det_full = DetectionBox(
        bbox=[103.0, 300.0, 143.0, 355.0],
        confidence=0.91,
        class_id=1,
        class_name="person",
    )

    projected = tiler.project_to_global([det_clipped], tile_left) + tiler.project_to_global([det_full], tile_right)
    assert projected[0].local_touching_seam is True
    assert projected[1].local_touching_seam is False

    merged = global_nms(projected, iou_threshold=0.40, use_wbf=True)
    assert len(merged) == 1
    assert merged[0].bbox[0] == pytest.approx(615.0, abs=1.0)
    assert merged[0].bbox[2] == pytest.approx(655.0, abs=1.0)


def test_global_nms_greedy_vs_wbf_distinct_classes() -> None:
    d1 = DetectionBox(bbox=[100, 100, 150, 150], confidence=0.88, class_id=1, class_name="person", tile_id=0)
    d2 = DetectionBox(bbox=[104, 102, 154, 152], confidence=0.80, class_id=1, class_name="person", tile_id=1)
    d3 = DetectionBox(bbox=[100, 100, 150, 150], confidence=0.75, class_id=3, class_name="vehicle", tile_id=0)

    greedy = global_nms([d1, d2, d3], iou_threshold=0.40, class_agnostic=False, use_wbf=False)
    assert len(greedy) == 2

    wbf = global_nms([d1, d2, d3], iou_threshold=0.40, class_agnostic=False, use_wbf=True)
    assert len(wbf) == 2
    person_det = next(d for d in wbf if d.class_name == "person")
    # Corroborated across two tiles boosts slightly above 0.88
    assert person_det.confidence > 0.88
