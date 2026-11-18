"""Tests for COCO Precision / Recall / F1 / AP evaluator and real sample benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.cv_engine.detector import SARDetector
from backend.cv_engine.evaluator import evaluate_predictions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DIR = PROJECT_ROOT / "sample_data"


def test_evaluator_exact_perfect_and_partial_metrics() -> None:
    gt = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "life_raft"},
        ],
        "images": [{"id": 1, "file_name": "scene_a.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 40, 50]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [300, 200, 80, 60]},
        ],
    }

    # Perfect predictions
    preds_perfect = {
        "scene_a.jpg": [
            {"class_name": "person", "confidence": 0.95, "bbox_xyxy": [100, 100, 140, 150]},
            {"class_name": "life_raft", "confidence": 0.90, "bbox_xyxy": [302, 202, 380, 260]},
        ]
    }
    res_perf = evaluate_predictions(gt, preds_perfect, iou_threshold=0.50)
    assert res_perf["summary"]["precision"] == pytest.approx(1.0)
    assert res_perf["summary"]["recall"] == pytest.approx(1.0)
    assert res_perf["summary"]["f1_score"] == pytest.approx(1.0)
    assert res_perf["summary"]["mAP"] == pytest.approx(1.0)

    # 1 TP, 1 FP, 1 FN -> Precision=0.5, Recall=0.5, F1=0.5
    preds_mixed = {
        "scene_a.jpg": [
            {"class_name": "person", "confidence": 0.92, "bbox_xyxy": [100, 100, 140, 150]},
            {"class_name": "person", "confidence": 0.65, "bbox_xyxy": [500, 500, 540, 540]},
        ]
    }
    res_mix = evaluate_predictions(gt, preds_mixed, iou_threshold=0.50)
    assert res_mix["summary"]["tp"] == 1
    assert res_mix["summary"]["fp"] == 1
    assert res_mix["summary"]["fn"] == 1
    assert res_mix["summary"]["precision"] == pytest.approx(0.5)
    assert res_mix["summary"]["recall"] == pytest.approx(0.5)
    assert res_mix["summary"]["f1_score"] == pytest.approx(0.5)


def test_real_end_to_end_evaluation_on_synthetic_sample_data() -> None:
    gt_path = SAMPLE_DIR / "ground_truth.json"
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    det = SARDetector(MODELS_DIR, "sarvision_micro_detector.onnx")

    preds = {}
    for im in gt_data["images"]:
        res = det.analyze_image(
            SAMPLE_DIR / im["file_name"],
            conf_threshold=0.45,
            nms_iou_threshold=0.40,
            detector_mode="fusion",
        )
        preds[im["file_name"]] = [
            {
                "class_name": g.box.class_name,
                "confidence": g.box.confidence,
                "bbox_xyxy": g.box.bbox,
            }
            for g in res.detections
        ]

    metrics = evaluate_predictions(gt_data, preds, iou_threshold=0.50)
    summary = metrics["summary"]
    assert summary["images_evaluated"] == 3
    assert summary["total_ground_truth"] == 12
    assert summary["recall"] >= 0.75
    assert summary["precision"] >= 0.75
    assert summary["mean_iou"] >= 0.65
