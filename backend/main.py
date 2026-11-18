"""SARVision AI FastAPI REST server and static dashboard host."""

from __future__ import annotations

import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import cv2
import onnxruntime as ort
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .config import Settings, get_settings
from .cv_engine.detector import InferenceResult, SARDetector
from .cv_engine.evaluator import evaluate_predictions
from .cv_engine.reporter import (
    export_coco_json,
    export_csv,
    generate_sar_mission_report_html,
    generate_sar_mission_report_md,
)
from .database import get_db, init_db
from .models import AnalysisRun, DetectionRecord
from .schemas import (
    AnalysisDetailOut,
    AnalysisSummaryOut,
    BatchAnalyzeResponse,
    DetectionOut,
    EvaluateRequest,
    ReanalyzeRequest,
    TileRect,
    VerifyDetectionRequest,
)


def _analysis_to_detail_dict(run: AnalysisRun) -> dict[str, Any]:
    try:
        tiles = json.loads(run.tiles_json) if run.tiles_json else []
    except json.JSONDecodeError:
        tiles = []
    dets = [DetectionOut.model_validate(d).model_dump() for d in run.detections]
    return {
        "id": run.id,
        "filename": run.filename,
        "image_width": run.image_width,
        "image_height": run.image_height,
        "tile_size": run.tile_size,
        "tile_overlap": run.tile_overlap,
        "tile_count": run.tile_count,
        "tiles": tiles,
        "conf_threshold": run.conf_threshold,
        "nms_iou_threshold": run.nms_iou_threshold,
        "detector_mode": run.detector_mode,
        "model_name": run.model_name,
        "processing_time_ms": run.processing_time_ms,
        "raw_tile_detections_count": run.raw_tile_detections_count,
        "has_gps_exif": run.has_gps_exif,
        "uav_lat": run.uav_lat,
        "uav_lon": run.uav_lon,
        "uav_alt_m": run.uav_alt_m,
        "gimbal_pitch_deg": run.gimbal_pitch_deg,
        "heading_deg": run.heading_deg,
        "hfov_deg": run.hfov_deg,
        "gsd_cm_px": run.gsd_cm_px,
        "swath_width_m": run.swath_width_m,
        "swath_height_m": run.swath_height_m,
        "synthetic_flag": run.synthetic_flag,
        "batch_id": run.batch_id,
        "created_at": run.created_at,
        "image_url": f"/api/analyses/{run.id}/image",
        "detections": dets,
    }


def _persist_inference(
    db: Session,
    filename: str,
    stored_path: Path,
    result: InferenceResult,
    tile_size: int,
    tile_overlap: float,
    conf_threshold: float,
    nms_iou_threshold: float,
    batch_id: str | None = None,
) -> AnalysisRun:
    tiles_payload = [
        TileRect(
            tile_id=t.tile_id,
            row=t.row,
            col=t.col,
            x_offset=t.x_offset,
            y_offset=t.y_offset,
            width=t.width,
            height=t.height,
        ).model_dump()
        for t in result.tiles
    ]

    run = AnalysisRun(
        filename=filename,
        stored_image_path=str(stored_path),
        image_width=result.image_width,
        image_height=result.image_height,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_count=len(result.tiles),
        tiles_json=json.dumps(tiles_payload),
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        detector_mode=result.detector_mode,
        model_name=result.model_name,
        processing_time_ms=result.processing_time_ms,
        raw_tile_detections_count=result.raw_tile_detections_count,
        has_gps_exif=result.telemetry.has_gps_exif,
        uav_lat=result.telemetry.latitude_deg,
        uav_lon=result.telemetry.longitude_deg,
        uav_alt_m=result.telemetry.altitude_m,
        gimbal_pitch_deg=result.telemetry.gimbal_pitch_deg,
        heading_deg=result.telemetry.heading_deg,
        hfov_deg=result.telemetry.hfov_deg,
        gsd_cm_px=result.telemetry.gsd_cm_px,
        swath_width_m=result.telemetry.swath_width_m,
        swath_height_m=result.telemetry.swath_height_m,
        synthetic_flag=result.telemetry.synthetic_flag,
        batch_id=batch_id,
    )
    for gd in result.detections:
        b = gd.box
        g = gd.geo
        run.detections.append(
            DetectionRecord(
                class_id=b.class_id,
                class_name=b.class_name,
                confidence=b.confidence,
                onnx_score=b.onnx_score,
                opencv_score=b.opencv_score,
                detector_source=b.source,
                color_signature=b.color_signature,
                x1=b.bbox[0],
                y1=b.bbox[1],
                x2=b.bbox[2],
                y2=b.bbox[3],
                tile_id=b.tile_id,
                wgs84_lat=g.wgs84_lat if g else None,
                wgs84_lon=g.wgs84_lon if g else None,
                north_offset_m=g.north_offset_m if g else None,
                east_offset_m=g.east_offset_m if g else None,
                slant_range_m=g.slant_range_m if g else None,
                verification_status="pending",
            )
        )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    """FastAPI application factory."""
    cfg = custom_settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
        init_db()
        application.state.detector = SARDetector(
            model_dir=cfg.model_dir,
            model_name=cfg.onnx_model,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
        )
        yield

    application = FastAPI(
        title=cfg.app_name,
        description="Offline-capable Aerial Search & Rescue Computer Vision & Geolocation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_detector() -> SARDetector:
        det = getattr(application.state, "detector", None)
        if det is None:
            det = SARDetector(
                model_dir=cfg.model_dir,
                model_name=cfg.onnx_model,
                tile_size=cfg.tile_size,
                tile_overlap=cfg.tile_overlap,
            )
            application.state.detector = det
        return det

    @application.get("/api/health")
    def health_check() -> dict[str, Any]:
        det = _get_detector()
        return {
            "status": "ok",
            "app_name": cfg.app_name,
            "opencv_version": cv2.__version__,
            "onnxruntime_version": ort.__version__,
            "model_loaded": det.model is not None,
            "active_model": det.model.model_name if det.model else None,
            "model_architecture": det.model.kind if det.model else "opencv-only",
            "classes": det.model.classes if det.model else ["background", "person", "life_raft", "vehicle", "distress_marker"],
        }

    @application.get("/api/models")
    def list_models() -> dict[str, Any]:
        det = _get_detector()
        return {
            "active_model": det.model.model_name if det.model else None,
            "models": det.list_available_models(),
        }

    @application.get("/api/samples")
    def list_samples() -> dict[str, Any]:
        sdir = cfg.sample_data_dir
        gt_path = sdir / "ground_truth.json"
        images: list[dict[str, Any]] = []
        gt_meta: dict[str, Any] = {}
        if gt_path.exists():
            try:
                gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
                gt_meta = gt_data.get("info", {})
                ann_counts: dict[int, int] = {}
                for ann in gt_data.get("annotations", []):
                    ann_counts[ann["image_id"]] = ann_counts.get(ann["image_id"], 0) + 1
                for im in gt_data.get("images", []):
                    fname = im["file_name"]
                    if (sdir / fname).exists():
                        images.append(
                            {
                                "file_name": fname,
                                "width": im["width"],
                                "height": im["height"],
                                "synthetic": im.get("synthetic", True),
                                "gt_targets": ann_counts.get(im["id"], 0),
                                "exif_gps": im.get("exif_gps", {}),
                            }
                        )
            except json.JSONDecodeError:
                pass
        if not images and sdir.exists():
            for p in sorted(sdir.glob("*.jpg")) + sorted(sdir.glob("*.png")):
                images.append({"file_name": p.name, "synthetic": True})
        return {"has_ground_truth": gt_path.exists(), "dataset_info": gt_meta, "samples": images}

    @application.post("/api/analyze", response_model=AnalysisDetailOut)
    async def analyze_upload(
        file: Annotated[UploadFile, File(...)],
        tile_size: Annotated[int, Form()] = 640,
        tile_overlap: Annotated[float, Form()] = 0.20,
        conf_threshold: Annotated[float, Form()] = 0.45,
        nms_iou_threshold: Annotated[float, Form()] = 0.40,
        detector_mode: Annotated[str, Form()] = "fusion",
        model_name: Annotated[str | None, Form()] = None,
        agl_m: Annotated[float | None, Form()] = None,
        gimbal_pitch_deg: Annotated[float | None, Form()] = None,
        hfov_deg: Annotated[float | None, Form()] = None,
        heading_deg: Annotated[float | None, Form()] = None,
        batch_id: Annotated[str | None, Form()] = None,
        db: Session = Depends(get_db),
    ):
        cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

        safe_name = Path(file.filename or "upload.jpg").name
        ext = Path(safe_name).suffix or ".jpg"
        stored_path = cfg.uploads_dir / f"{uuid.uuid4().hex[:12]}{ext}"
        stored_path.write_bytes(raw_bytes)

        det = _get_detector()
        try:
            result = det.analyze_image(
                stored_path,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                detector_mode=detector_mode,
                model_name=model_name or None,
                agl_override_m=agl_m,
                pitch_override_deg=gimbal_pitch_deg,
                hfov_override_deg=hfov_deg,
                heading_override_deg=heading_deg,
            )
        except ValueError as exc:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run = _persist_inference(
            db,
            filename=safe_name,
            stored_path=stored_path,
            result=result,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            batch_id=batch_id,
        )
        return _analysis_to_detail_dict(run)

    @application.post("/api/analyze/batch", response_model=BatchAnalyzeResponse)
    async def analyze_batch(
        files: list[UploadFile] = File(...),
        tile_size: Annotated[int, Form()] = 640,
        tile_overlap: Annotated[float, Form()] = 0.20,
        conf_threshold: Annotated[float, Form()] = 0.45,
        nms_iou_threshold: Annotated[float, Form()] = 0.40,
        detector_mode: Annotated[str, Form()] = "fusion",
        model_name: Annotated[str | None, Form()] = None,
        db: Session = Depends(get_db),
    ):
        if not files:
            raise HTTPException(status_code=400, detail="No files provided for batch analysis.")

        cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
        batch_id = uuid.uuid4().hex[:10]
        det = _get_detector()
        analyses_out: list[dict[str, Any]] = []
        total_dets = 0

        for upload in files:
            raw_bytes = await upload.read()
            if not raw_bytes:
                continue
            safe_name = Path(upload.filename or "batch_frame.jpg").name
            ext = Path(safe_name).suffix or ".jpg"
            stored_path = cfg.uploads_dir / f"{uuid.uuid4().hex[:12]}{ext}"
            stored_path.write_bytes(raw_bytes)
            result = det.analyze_image(
                stored_path,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                detector_mode=detector_mode,
                model_name=model_name or None,
            )
            run = _persist_inference(
                db,
                filename=safe_name,
                stored_path=stored_path,
                result=result,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                batch_id=batch_id,
            )
            total_dets += len(run.detections)
            analyses_out.append(_analysis_to_detail_dict(run))

        return {
            "batch_id": batch_id,
            "total_files": len(analyses_out),
            "total_detections": total_dets,
            "analyses": analyses_out,
        }

    @application.post("/api/analyze/sample/{filename}", response_model=AnalysisDetailOut)
    def analyze_sample_scene(
        filename: str,
        tile_size: int = Query(640, ge=128, le=2048),
        tile_overlap: float = Query(0.20, ge=0.0, lt=0.8),
        conf_threshold: float = Query(0.45, ge=0.05, le=0.99),
        nms_iou_threshold: float = Query(0.40, ge=0.05, le=0.95),
        detector_mode: str = Query("fusion"),
        model_name: str | None = Query(None),
        db: Session = Depends(get_db),
    ):
        safe_name = Path(filename).name
        sample_path = cfg.sample_data_dir / safe_name
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample image '{safe_name}' not found.")

        cfg.uploads_dir.mkdir(parents=True, exist_ok=True)
        stored_path = cfg.uploads_dir / f"{uuid.uuid4().hex[:12]}_{safe_name}"
        shutil.copy2(sample_path, stored_path)

        det = _get_detector()
        result = det.analyze_image(
            stored_path,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            detector_mode=detector_mode,
            model_name=model_name,
        )
        run = _persist_inference(
            db,
            filename=safe_name,
            stored_path=stored_path,
            result=result,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        return _analysis_to_detail_dict(run)

    @application.get("/api/analyses", response_model=list[AnalysisSummaryOut])
    def list_analyses(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
        stmt = (
            select(AnalysisRun)
            .options(selectinload(AnalysisRun.detections))
            .order_by(AnalysisRun.created_at.desc())
            .limit(limit)
        )
        runs = db.scalars(stmt).all()
        summaries: list[dict[str, Any]] = []
        for run in runs:
            dets = run.detections
            confirmed = sum(1 for d in dets if d.verification_status == "confirmed")
            escalated = sum(1 for d in dets if d.verification_status == "escalated")
            fp = sum(1 for d in dets if d.verification_status == "false_positive")
            pending = len(dets) - confirmed - escalated - fp
            summaries.append(
                {
                    "id": run.id,
                    "filename": run.filename,
                    "image_width": run.image_width,
                    "image_height": run.image_height,
                    "detector_mode": run.detector_mode,
                    "model_name": run.model_name,
                    "processing_time_ms": run.processing_time_ms,
                    "has_gps_exif": run.has_gps_exif,
                    "uav_lat": run.uav_lat,
                    "uav_lon": run.uav_lon,
                    "detection_count": len(dets),
                    "confirmed_count": confirmed,
                    "escalated_count": escalated,
                    "false_positive_count": fp,
                    "pending_count": pending,
                    "batch_id": run.batch_id,
                    "created_at": run.created_at,
                }
            )
        return summaries

    @application.get("/api/analyses/{analysis_id}", response_model=AnalysisDetailOut)
    def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
        run = db.scalar(
            select(AnalysisRun)
            .options(selectinload(AnalysisRun.detections))
            .where(AnalysisRun.id == analysis_id)
        )
        if not run:
            raise HTTPException(status_code=404, detail="Analysis run not found.")
        return _analysis_to_detail_dict(run)

    @application.delete("/api/analyses/{analysis_id}")
    def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
        run = db.get(AnalysisRun, analysis_id)
        if not run:
            raise HTTPException(status_code=404, detail="Analysis run not found.")
        img_path = Path(run.stored_image_path)
        db.delete(run)
        db.commit()
        if img_path.exists() and cfg.uploads_dir in img_path.parents:
            img_path.unlink(missing_ok=True)
        return {"deleted": True, "analysis_id": analysis_id}

    @application.get("/api/analyses/{analysis_id}/image")
    def get_analysis_image(analysis_id: str, db: Session = Depends(get_db)):
        run = db.get(AnalysisRun, analysis_id)
        if not run:
            raise HTTPException(status_code=404, detail="Analysis run not found.")
        img_path = Path(run.stored_image_path)
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Source aerial image file missing on disk.")
        return FileResponse(img_path)

    @application.post("/api/analyses/{analysis_id}/reanalyze", response_model=AnalysisDetailOut)
    def reanalyze_existing(
        analysis_id: str,
        payload: ReanalyzeRequest,
        db: Session = Depends(get_db),
    ):
        run = db.scalar(
            select(AnalysisRun)
            .options(selectinload(AnalysisRun.detections))
            .where(AnalysisRun.id == analysis_id)
        )
        if not run:
            raise HTTPException(status_code=404, detail="Analysis run not found.")
        img_path = Path(run.stored_image_path)
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Source aerial image missing on disk.")

        det = _get_detector()
        result = det.analyze_image(
            img_path,
            tile_size=payload.tile_size,
            tile_overlap=payload.tile_overlap,
            conf_threshold=payload.conf_threshold,
            nms_iou_threshold=payload.nms_iou_threshold,
            detector_mode=payload.detector_mode,
            model_name=payload.model_name,
            agl_override_m=payload.agl_m,
            pitch_override_deg=payload.gimbal_pitch_deg,
            hfov_override_deg=payload.hfov_deg,
            heading_override_deg=payload.heading_deg,
        )

        # Replace detections and update run metadata in place
        run.detections.clear()
        db.flush()
        run.tile_size = payload.tile_size
        run.tile_overlap = payload.tile_overlap
        run.tile_count = len(result.tiles)
        run.tiles_json = json.dumps(
            [
                TileRect(
                    tile_id=t.tile_id,
                    row=t.row,
                    col=t.col,
                    x_offset=t.x_offset,
                    y_offset=t.y_offset,
                    width=t.width,
                    height=t.height,
                ).model_dump()
                for t in result.tiles
            ]
        )
        run.conf_threshold = payload.conf_threshold
        run.nms_iou_threshold = payload.nms_iou_threshold
        run.detector_mode = result.detector_mode
        run.model_name = result.model_name
        run.processing_time_ms = result.processing_time_ms
        run.raw_tile_detections_count = result.raw_tile_detections_count
        run.uav_alt_m = result.telemetry.altitude_m
        run.gimbal_pitch_deg = result.telemetry.gimbal_pitch_deg
        run.heading_deg = result.telemetry.heading_deg
        run.hfov_deg = result.telemetry.hfov_deg
        run.gsd_cm_px = result.telemetry.gsd_cm_px
        run.swath_width_m = result.telemetry.swath_width_m
        run.swath_height_m = result.telemetry.swath_height_m

        for gd in result.detections:
            b = gd.box
            g = gd.geo
            run.detections.append(
                DetectionRecord(
                    class_id=b.class_id,
                    class_name=b.class_name,
                    confidence=b.confidence,
                    onnx_score=b.onnx_score,
                    opencv_score=b.opencv_score,
                    detector_source=b.source,
                    color_signature=b.color_signature,
                    x1=b.bbox[0],
                    y1=b.bbox[1],
                    x2=b.bbox[2],
                    y2=b.bbox[3],
                    tile_id=b.tile_id,
                    wgs84_lat=g.wgs84_lat if g else None,
                    wgs84_lon=g.wgs84_lon if g else None,
                    north_offset_m=g.north_offset_m if g else None,
                    east_offset_m=g.east_offset_m if g else None,
                    slant_range_m=g.slant_range_m if g else None,
                    verification_status="pending",
                )
            )
        db.commit()
        db.refresh(run)
        return _analysis_to_detail_dict(run)

    @application.patch("/api/detections/{detection_id}/verify", response_model=DetectionOut)
    def verify_detection(
        detection_id: int,
        payload: VerifyDetectionRequest,
        db: Session = Depends(get_db),
    ):
        det_rec = db.get(DetectionRecord, detection_id)
        if not det_rec:
            raise HTTPException(status_code=404, detail="Detection record not found.")

        det_rec.verification_status = payload.verification_status
        if payload.verified_class_name is not None:
            det_rec.verified_class_name = payload.verified_class_name
        if payload.operator_notes is not None:
            det_rec.operator_notes = payload.operator_notes
        db.commit()
        db.refresh(det_rec)
        return det_rec

    @application.post("/api/evaluate")
    def evaluate_on_dataset(payload: EvaluateRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
        gt_path = cfg.sample_data_dir / "ground_truth.json"
        if not gt_path.exists():
            raise HTTPException(status_code=404, detail="sample_data/ground_truth.json not found.")
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))

        preds_by_image: dict[str, list[dict[str, Any]]] = {}
        if payload.analysis_ids:
            runs = db.scalars(
                select(AnalysisRun)
                .options(selectinload(AnalysisRun.detections))
                .where(AnalysisRun.id.in_(payload.analysis_ids))
            ).all()
            for run in runs:
                preds_by_image[run.filename] = [
                    {
                        "class_name": d.verified_class_name or d.class_name,
                        "confidence": d.confidence,
                        "bbox_xyxy": [d.x1, d.y1, d.x2, d.y2],
                    }
                    for d in run.detections
                    if d.verification_status != "false_positive"
                ]
        else:
            det = _get_detector()
            for im in gt_data.get("images", []):
                fname = im["file_name"]
                img_path = cfg.sample_data_dir / fname
                if not img_path.exists():
                    continue
                res = det.analyze_image(
                    img_path,
                    tile_size=payload.tile_size,
                    tile_overlap=payload.tile_overlap,
                    conf_threshold=payload.conf_threshold,
                    nms_iou_threshold=payload.nms_iou_threshold,
                    detector_mode=payload.detector_mode,
                    model_name=payload.model_name,
                )
                preds_by_image[fname] = [
                    {
                        "class_name": gd.box.class_name,
                        "confidence": gd.box.confidence,
                        "bbox_xyxy": gd.box.bbox,
                    }
                    for gd in res.detections
                ]

        metrics = evaluate_predictions(gt_data, preds_by_image, iou_threshold=payload.iou_match_threshold)
        metrics["evaluation_config"] = {
            "detector_mode": payload.detector_mode,
            "conf_threshold": payload.conf_threshold,
            "nms_iou_threshold": payload.nms_iou_threshold,
            "tile_size": payload.tile_size,
            "tile_overlap": payload.tile_overlap,
        }
        return metrics

    @application.get("/api/analyses/{analysis_id}/export")
    def export_analysis(
        analysis_id: str,
        format: str = Query("coco", pattern="^(coco|csv|markdown|html)$"),
        include_rejected: bool = Query(False),
        db: Session = Depends(get_db),
    ):
        run = db.scalar(
            select(AnalysisRun)
            .options(selectinload(AnalysisRun.detections))
            .where(AnalysisRun.id == analysis_id)
        )
        if not run:
            raise HTTPException(status_code=404, detail="Analysis run not found.")

        payload = [_analysis_to_detail_dict(run)]
        return _format_export_response(payload, format_name=format, prefix=f"sarvision_{run.id}", include_rejected=include_rejected)

    @application.get("/api/batch/{batch_id}/export")
    def export_batch(
        batch_id: str,
        format: str = Query("coco", pattern="^(coco|csv|markdown|html)$"),
        include_rejected: bool = Query(False),
        db: Session = Depends(get_db),
    ):
        runs = db.scalars(
            select(AnalysisRun)
            .options(selectinload(AnalysisRun.detections))
            .where(AnalysisRun.batch_id == batch_id)
            .order_by(AnalysisRun.created_at.asc())
        ).all()
        if not runs:
            raise HTTPException(status_code=404, detail="Batch not found.")

        payload = [_analysis_to_detail_dict(r) for r in runs]
        return _format_export_response(payload, format_name=format, prefix=f"sarvision_batch_{batch_id}", include_rejected=include_rejected)

    # Serve built React+TS+Vite production UI if present
    dist_dir = cfg.frontend_dist_dir
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        application.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @application.get("/", response_class=HTMLResponse)
    def serve_dashboard():
        index_file = dist_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;background:#0b111b;color:#e2e8f0;padding:2rem'>"
            "<h2>SARVision AI Backend Online</h2>"
            "<p>API docs available at <a href='/docs' style='color:#38bdf8'>/docs</a>. "
            "Build frontend with <code>cd frontend && npm install && npm run build</code>.</p>"
            "</body></html>"
        )

    return application


def _format_export_response(
    analyses: list[dict[str, Any]],
    format_name: str,
    prefix: str,
    include_rejected: bool,
) -> Response:
    if format_name == "coco":
        data = export_coco_json(analyses, include_rejected=include_rejected)
        return JSONResponse(
            content=data,
            headers={"Content-Disposition": f'attachment; filename="{prefix}_coco.json"'},
        )
    if format_name == "csv":
        csv_text = export_csv(analyses)
        return PlainTextResponse(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{prefix}.csv"'},
        )
    if format_name == "html":
        html_text = generate_sar_mission_report_html(analyses)
        return HTMLResponse(
            content=html_text,
            headers={"Content-Disposition": f'attachment; filename="{prefix}_report.html"'},
        )
    md_text = generate_sar_mission_report_md(analyses)
    return PlainTextResponse(
        content=md_text,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{prefix}_report.md"'},
    )


app = create_app()
