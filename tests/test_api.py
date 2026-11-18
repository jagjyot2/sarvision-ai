"""Integration tests for FastAPI REST endpoints, SQLite persistence, verification, and exports."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.database import create_engine_and_session, get_db, init_db
from backend.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "sample_data"


@pytest.fixture()
def api_client(tmp_path: Path):
    db_file = tmp_path / "test_sarvision.db"
    uploads_dir = tmp_path / "uploads"
    test_cfg = Settings(
        db_url=f"sqlite:///{db_file.as_posix()}",
        uploads_dir=uploads_dir,
        model_dir=PROJECT_ROOT / "models",
        sample_data_dir=SAMPLE_DIR,
        frontend_dist_dir=PROJECT_ROOT / "frontend" / "dist",
    )
    engine, session_local = create_engine_and_session(test_cfg.db_url)
    init_db(engine)

    app = create_app(test_cfg)

    def _override_db():
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_db
    with TestClient(app) as client:
        yield client
    engine.dispose()


def test_health_models_samples_and_root(api_client: TestClient) -> None:
    r = api_client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["active_model"] == "sarvision_micro_detector.onnx"

    m = api_client.get("/api/models")
    assert m.status_code == 200
    assert any(item["file_name"] == "sarvision_micro_detector.onnx" for item in m.json()["models"])

    s = api_client.get("/api/samples")
    assert s.status_code == 200
    assert s.json()["has_ground_truth"] is True
    assert len(s.json()["samples"]) == 3

    root = api_client.get("/")
    assert root.status_code == 200
    assert "SARVision" in root.text


def test_upload_verify_reanalyze_export_and_delete_workflow(api_client: TestClient) -> None:
    sample_img = SAMPLE_DIR / "sar_maritime_openwater.jpg"
    with sample_img.open("rb") as fh:
        up = api_client.post(
            "/api/analyze",
            files={"file": ("sar_maritime_openwater.jpg", fh, "image/jpeg")},
            data={"conf_threshold": "0.45", "nms_iou_threshold": "0.40", "detector_mode": "fusion"},
        )
    assert up.status_code == 200
    detail = up.json()
    analysis_id = detail["id"]
    assert detail["has_gps_exif"] is True
    assert len(detail["detections"]) >= 3

    # Verify first detection -> confirmed + note + reclassify
    det_id = detail["detections"][0]["id"]
    ver = api_client.patch(
        f"/api/detections/{det_id}/verify",
        json={
            "verification_status": "confirmed",
            "verified_class_name": "life_raft",
            "operator_notes": "SOLAS orange raft spotted in swell.",
        },
    )
    assert ver.status_code == 200
    assert ver.json()["verification_status"] == "confirmed"
    assert ver.json()["verified_class_name"] == "life_raft"

    # Check history reflects confirmed count
    hist = api_client.get("/api/analyses")
    assert hist.status_code == 200
    assert hist.json()[0]["confirmed_count"] == 1

    # Exports (COCO, CSV, Markdown, HTML)
    coco = api_client.get(f"/api/analyses/{analysis_id}/export?format=coco")
    assert coco.status_code == 200
    assert len(coco.json()["annotations"]) >= 3

    csv_resp = api_client.get(f"/api/analyses/{analysis_id}/export?format=csv")
    assert csv_resp.status_code == 200
    assert "SOLAS orange raft spotted in swell." in csv_resp.text

    md_resp = api_client.get(f"/api/analyses/{analysis_id}/export?format=markdown")
    assert md_resp.status_code == 200
    assert "SARVision AI" in md_resp.text

    html_resp = api_client.get(f"/api/analyses/{analysis_id}/export?format=html")
    assert html_resp.status_code == 200
    assert "<html" in html_resp.text.lower()
    assert "mission intelligence report" in html_resp.text.lower()

    # Re-analyze existing image
    re = api_client.post(
        f"/api/analyses/{analysis_id}/reanalyze",
        json={"conf_threshold": 0.50, "nms_iou_threshold": 0.40, "detector_mode": "fusion"},
    )
    assert re.status_code == 200
    assert re.json()["conf_threshold"] == 0.50

    # Delete run
    d = api_client.delete(f"/api/analyses/{analysis_id}")
    assert d.status_code == 200
    assert api_client.get(f"/api/analyses/{analysis_id}").status_code == 404


def test_batch_analyze_and_evaluate_endpoint(api_client: TestClient) -> None:
    files = []
    fhs = []
    for name in ("sar_wilderness_clearing.jpg", "sar_alpine_snowfield.jpg"):
        fh = (SAMPLE_DIR / name).open("rb")
        fhs.append(fh)
        files.append(("files", (name, fh, "image/jpeg")))

    try:
        batch = api_client.post("/api/analyze/batch", files=files)
    finally:
        for fh in fhs:
            fh.close()

    assert batch.status_code == 200
    b_data = batch.json()
    assert b_data["total_files"] == 2
    assert b_data["total_detections"] >= 6

    ev = api_client.post(
        "/api/evaluate",
        json={"conf_threshold": 0.45, "nms_iou_threshold": 0.40, "iou_match_threshold": 0.50, "detector_mode": "fusion"},
    )
    assert ev.status_code == 200
    summary = ev.json()["summary"]
    assert summary["images_evaluated"] == 3
    assert summary["recall"] >= 0.75
    assert summary["precision"] >= 0.75
