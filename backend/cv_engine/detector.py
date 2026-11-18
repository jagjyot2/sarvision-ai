"""Dual-stage aerial SAR target detector (ONNX Runtime neural model + OpenCV HSV/CLAHE saliency)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .exif_geo import ExifTelemetry, GeoProjection, project_pixel_to_wgs84, read_exif_telemetry
from .tiler import DetectionBox, ImageTiler, TileSpec, compute_iou, global_nms

DEFAULT_CLASSES = ["background", "person", "life_raft", "vehicle", "distress_marker"]
DEFAULT_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Standard COCO-80 subset mapping if a generic YOLOv8/v11 ONNX model is supplied
YOLO_TO_SAR_MAP: dict[int, tuple[int, str]] = {
    0: (1, "person"),
    2: (3, "vehicle"),
    3: (3, "vehicle"),
    5: (3, "vehicle"),
    7: (3, "vehicle"),
    8: (2, "life_raft"),
}


@dataclass(slots=True)
class GeoDetection:
    """Detection enriched with optional WGS84 geospatial projection."""

    box: DetectionBox
    geo: GeoProjection | None = None


@dataclass(slots=True)
class InferenceResult:
    """Complete result returned by SARDetector.analyze_image."""

    image_width: int
    image_height: int
    tiles: list[TileSpec]
    detections: list[GeoDetection]
    telemetry: ExifTelemetry
    processing_time_ms: float
    detector_mode: str
    model_name: str
    raw_tile_detections_count: int


def _nms_local(
    detections: list[DetectionBox],
    iou_threshold: float = 0.40,
    class_agnostic: bool = False,
    min_center_dist: float = 0.0,
) -> list[DetectionBox]:
    """Tile-local greedy NMS with optional center-distance deduplication."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[DetectionBox] = []
    for cand in ordered:
        ccx, ccy = cand.center
        suppressed = False
        for prev in kept:
            if not class_agnostic and cand.class_name != prev.class_name:
                continue
            if compute_iou(cand.bbox, prev.bbox) >= iou_threshold:
                suppressed = True
                break
            if min_center_dist > 0.0:
                pcx, pcy = prev.center
                if math.hypot(ccx - pcx, ccy - pcy) < min_center_dist:
                    suppressed = True
                    break
        if not suppressed:
            kept.append(cand)
    return kept


def _merge_close_proposals(
    proposals: list[tuple[list[float], float, str, str]],
    iou_thresh: float = 0.35,
) -> list[tuple[list[float], float, str, str]]:
    """Merge heavily overlapping or nested OpenCV candidate proposals."""
    if not proposals:
        return []
    ordered = sorted(proposals, key=lambda p: p[1], reverse=True)
    merged: list[tuple[list[float], float, str, str]] = []
    for box, score, hint, color_sig in ordered:
        duplicate = False
        ax1, ay1, ax2, ay2 = box
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        for m_box, _, _, _ in merged:
            if compute_iou(box, m_box) >= iou_thresh:
                duplicate = True
                break
            ix1, iy1 = max(ax1, m_box[0]), max(ay1, m_box[1])
            ix2, iy2 = min(ax2, m_box[2]), min(ay2, m_box[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / area_a >= 0.65:
                duplicate = True
                break
        if not duplicate:
            merged.append((box, score, hint, color_sig))
    return merged


def _center_inside_expanded(inner_box: list[float], outer_box: list[float], expand: float = 0.35) -> bool:
    """Return True if inner_box center lies inside expanded outer_box."""
    cx = 0.5 * (inner_box[0] + inner_box[2])
    cy = 0.5 * (inner_box[1] + inner_box[3])
    ow = outer_box[2] - outer_box[0]
    oh = outer_box[3] - outer_box[1]
    pad_x = max(16.0, ow * expand)
    pad_y = max(16.0, oh * expand)
    return (outer_box[0] - pad_x <= cx <= outer_box[2] + pad_x) and (
        outer_box[1] - pad_y <= cy <= outer_box[3] + pad_y
    )


class ONNXModelAdapter:
    """Loads either SARVisionMicroNet patch classifier ONNX or YOLOv8/YOLOv11 ONNX weights."""

    def __init__(self, model_path: Path, metadata_path: Path | None = None) -> None:
        self.model_path = model_path
        self.model_name = model_path.name
        self.metadata: dict[str, Any] = {}
        if metadata_path and metadata_path.exists():
            try:
                self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.metadata = {}

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name
        self.input_shape = list(inp.shape)
        self.output_shape = list(out.shape)

        arch = self.metadata.get("architecture", "")
        in_h = self.input_shape[2] if len(self.input_shape) >= 4 and isinstance(self.input_shape[2], int) else 64
        in_w = self.input_shape[3] if len(self.input_shape) >= 4 and isinstance(self.input_shape[3], int) else 64

        if arch == "sarvision_micro_cnn" or (in_h <= 128 and len(self.output_shape) == 2):
            self.kind = "patch_cnn"
            self.patch_size = int(in_h)
            self.classes: list[str] = list(self.metadata.get("classes", DEFAULT_CLASSES))
            norm = self.metadata.get("normalization", {})
            self.mean = np.array(norm.get("mean", DEFAULT_NORM_MEAN), dtype=np.float32).reshape(1, 1, 1, 3)
            self.std = np.array(norm.get("std", DEFAULT_NORM_STD), dtype=np.float32).reshape(1, 1, 1, 3)
        else:
            self.kind = "yolo"
            self.yolo_size = (
                int(in_h) if isinstance(in_h, int) else 640,
                int(in_w) if isinstance(in_w, int) else 640,
            )
            self.classes = list(self.metadata.get("classes", ["person", "life_raft", "vehicle", "distress_marker"]))

    def classify_patches(self, rgb_patches: list[np.ndarray]) -> np.ndarray:
        """Run batched ONNX inference on RGB patches -> softmax probabilities (N, C)."""
        if not rgb_patches:
            return np.zeros((0, len(self.classes)), dtype=np.float32)

        size = self.patch_size
        resized = [
            cv2.resize(p, (size, size), interpolation=cv2.INTER_LINEAR) if p.shape[:2] != (size, size) else p
            for p in rgb_patches
        ]
        batch = np.stack(resized, axis=0).astype(np.float32) / 255.0
        batch = (batch - self.mean) / self.std
        nchw = np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)

        logits = self.session.run([self.output_name], {self.input_name: nchw})[0]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs.astype(np.float32)

    def detect_yolo_tile(self, bgr_tile: np.ndarray, conf_threshold: float, iou_threshold: float) -> list[DetectionBox]:
        """Run YOLOv8/YOLOv11 ONNX inference on a single tile with letterbox pre/post-processing."""
        th, tw = bgr_tile.shape[:2]
        in_h, in_w = self.yolo_size
        scale = min(in_w / max(tw, 1), in_h / max(th, 1))
        nw, nh = int(round(tw * scale)), int(round(th * scale))
        resized = cv2.resize(bgr_tile, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (in_w - nw) // 2, (in_h - nh) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        raw = self.session.run([self.output_name], {self.input_name: blob})[0]

        preds = np.squeeze(raw, axis=0)
        detections: list[DetectionBox] = []
        if preds.ndim == 2 and preds.shape[1] == 6:
            for row in preds:
                conf = float(row[4])
                if conf < conf_threshold:
                    continue
                cls_idx = int(row[5])
                cid, cname = self._map_yolo_class(cls_idx)
                x1 = max(0.0, (float(row[0]) - pad_x) / scale)
                y1 = max(0.0, (float(row[1]) - pad_y) / scale)
                x2 = min(float(tw), (float(row[2]) - pad_x) / scale)
                y2 = min(float(th), (float(row[3]) - pad_y) / scale)
                if x2 > x1 and y2 > y1:
                    detections.append(
                        DetectionBox(
                            bbox=[x1, y1, x2, y2],
                            confidence=round(conf, 4),
                            class_id=cid,
                            class_name=cname,
                            source="onnx-yolo",
                            onnx_score=round(conf, 4),
                        )
                    )
            return _nms_local(detections, iou_threshold=iou_threshold)

        if preds.ndim == 2 and preds.shape[0] < preds.shape[1] and preds.shape[0] >= 5:
            preds = preds.T

        if preds.ndim != 2 or preds.shape[1] < 5:
            return []

        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4:]
        best_cls = np.argmax(cls_scores, axis=1)
        best_conf = np.max(cls_scores, axis=1)
        keep = best_conf >= conf_threshold
        for xywh, conf, cls_idx in zip(boxes_xywh[keep], best_conf[keep], best_cls[keep]):
            cx, cy, bw, bh = [float(v) for v in xywh]
            x1 = max(0.0, ((cx - bw / 2.0) - pad_x) / scale)
            y1 = max(0.0, ((cy - bh / 2.0) - pad_y) / scale)
            x2 = min(float(tw), ((cx + bw / 2.0) - pad_x) / scale)
            y2 = min(float(th), ((cy + bh / 2.0) - pad_y) / scale)
            if x2 <= x1 or y2 <= y1:
                continue
            cid, cname = self._map_yolo_class(int(cls_idx))
            detections.append(
                DetectionBox(
                    bbox=[x1, y1, x2, y2],
                    confidence=round(float(conf), 4),
                    class_id=cid,
                    class_name=cname,
                    source="onnx-yolo",
                    onnx_score=round(float(conf), 4),
                )
            )
        return _nms_local(detections, iou_threshold=iou_threshold)

    def _map_yolo_class(self, cls_idx: int) -> tuple[int, str]:
        if len(self.classes) <= 5 and 0 <= cls_idx < len(self.classes):
            name = self.classes[cls_idx]
            if name == "background":
                return 4, "distress_marker"
            return cls_idx + (0 if cls_idx > 0 else 1), name
        return YOLO_TO_SAR_MAP.get(cls_idx, (1, "person"))


class SARDetector:
    """Dual-stage SAR object detector combining OpenCV saliency/HSV + ONNX Runtime."""

    def __init__(
        self,
        model_dir: Path | str,
        model_name: str | None = None,
        tile_size: int = 640,
        tile_overlap: float = 0.20,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.model: ONNXModelAdapter | None = None
        self.load_model(model_name)

    def list_available_models(self) -> list[dict[str, Any]]:
        """List .onnx models present inside model_dir."""
        if not self.model_dir.exists():
            return []
        models = []
        meta_path = self.model_dir / "model_metadata.json"
        default_meta = {}
        if meta_path.exists():
            try:
                default_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                default_meta = {}
        for p in sorted(self.model_dir.glob("*.onnx")):
            side_meta = p.with_suffix(".json")
            meta = default_meta if p.name == default_meta.get("onnx_file") else {}
            if side_meta.exists():
                try:
                    meta = json.loads(side_meta.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            models.append(
                {
                    "file_name": p.name,
                    "size_kb": round(p.stat().st_size / 1024.0, 2),
                    "architecture": meta.get("architecture", "yolo_or_custom_onnx"),
                    "classes": meta.get("classes", DEFAULT_CLASSES),
                }
            )
        return models

    def load_model(self, model_name: str | None = None) -> None:
        """Load specified ONNX model or auto-discover model in model_dir."""
        if not self.model_dir.exists():
            self.model = None
            return

        candidate: Path | None = None
        if model_name:
            p = self.model_dir / model_name
            if p.exists():
                candidate = p
        if candidate is None:
            onnx_files = sorted(self.model_dir.glob("*.onnx"))
            micro = self.model_dir / "sarvision_micro_detector.onnx"
            if micro.exists():
                candidate = micro
            elif onnx_files:
                candidate = onnx_files[0]

        if candidate is None:
            self.model = None
            return

        meta_path = candidate.with_suffix(".json")
        if not meta_path.exists():
            meta_path = self.model_dir / "model_metadata.json"
        self.model = ONNXModelAdapter(candidate, meta_path if meta_path.exists() else None)

    @staticmethod
    def _extract_opencv_proposals(bgr_tile: np.ndarray) -> list[tuple[list[float], float, str, str]]:
        """Extract candidate target bounding boxes using HSV rescue-color + CLAHE local saliency."""
        th, tw = bgr_tile.shape[:2]
        hsv = cv2.cvtColor(bgr_tile, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr_tile, cv2.COLOR_BGR2LAB)

        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # Channel 1: High-visibility rescue colors (orange/red, safety yellow, alpine cobalt/cyan)
        mask_orange = ((h_ch <= 17) | (h_ch >= 164)) & (s_ch >= 135) & (v_ch >= 90)
        mask_yellow = (h_ch > 18) & (h_ch <= 38) & (s_ch >= 120) & (v_ch >= 115)
        mask_cyan = (h_ch >= 94) & (h_ch <= 128) & (s_ch >= 130) & (v_ch >= 90)
        hsv_mask = ((mask_orange | mask_yellow | mask_cyan).astype(np.uint8)) * 255

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        hsv_clean = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, k_open)
        hsv_clean = cv2.morphologyEx(hsv_clean, cv2.MORPH_CLOSE, k_close)

        # Channel 2: CLAHE L-channel + multi-scale LAB local anomaly saliency (for muted/dark/metallic targets)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        l_clahe = clahe.apply(lab[:, :, 0]).astype(np.float32)
        a_f = lab[:, :, 1].astype(np.float32)
        b_f = lab[:, :, 2].astype(np.float32)

        l_bg = cv2.GaussianBlur(l_clahe, (51, 51), 0)
        a_bg = cv2.GaussianBlur(a_f, (51, 51), 0)
        b_bg = cv2.GaussianBlur(b_f, (51, 51), 0)

        l_diff = np.abs(l_clahe - l_bg)
        c_diff = np.hypot(a_f - a_bg, b_f - b_bg)
        sal = 0.45 * (l_diff / max(float(l_diff.max()), 1.0)) + 0.55 * (c_diff / max(float(c_diff.max()), 1.0))
        sal_mean, sal_std = float(sal.mean()), float(sal.std())
        sal_thresh = max(0.38, sal_mean + 3.4 * sal_std)
        sal_mask = (sal >= sal_thresh).astype(np.uint8) * 255
        sal_clean = cv2.morphologyEx(sal_mask, cv2.MORPH_OPEN, k_open)
        sal_clean = cv2.morphologyEx(sal_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        gray = cv2.cvtColor(bgr_tile, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)

        proposals: list[tuple[list[float], float, str, str]] = []

        for source_name, mask in (("hsv", hsv_clean), ("saliency", sal_clean)):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                c_area = float(cv2.contourArea(cnt))
                if c_area < 150.0 or c_area > 35000.0:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 14 or bh < 14 or bw > int(tw * 0.65) or bh > int(th * 0.65):
                    continue
                aspect = max(bw / max(bh, 1), bh / max(bw, 1))
                if aspect > 3.8:
                    continue

                bbox_area = float(bw * bh)
                extent = c_area / max(bbox_area, 1.0)
                if extent < 0.30:
                    continue

                roi_s = s_ch[y : y + bh, x : x + bw]
                roi_h = h_ch[y : y + bh, x : x + bw]
                roi_sal = sal[y : y + bh, x : x + bw]
                roi_grad = grad_mag[y : y + bh, x : x + bw]

                mean_sat = float(np.percentile(roi_s, 75)) / 255.0
                peak_sal = float(np.percentile(roi_sal, 85))
                edge_score = min(1.0, float(np.percentile(roi_grad, 88)) / 145.0)
                med_all_hue = float(np.median(roi_h))

                if source_name == "saliency":
                    # Reject soft ocean foam / smooth snow drift rocks & natural foliage crowns
                    if edge_score < 0.42:
                        continue
                    if 38.0 <= med_all_hue <= 88.0 and mean_sat < 0.72:
                        continue
                    if mean_sat < 0.25 and edge_score < 0.55:
                        continue

                hi_sat = roi_s > 110
                if np.any(hi_sat):
                    med_hue = float(np.median(roi_h[hi_sat]))
                    if med_hue <= 18 or med_hue >= 162:
                        color_sig = "rescue-orange-red"
                    elif 18 < med_hue <= 40:
                        color_sig = "safety-yellow"
                    elif 92 <= med_hue <= 132:
                        color_sig = "alpine-cobalt-cyan"
                    else:
                        color_sig = "high-contrast-chroma"
                else:
                    color_sig = "thermal-luminance-contrast"

                if bbox_area < 1600 and aspect <= 2.2:
                    cls_hint = "person"
                elif aspect >= 1.48 and bbox_area >= 2200:
                    cls_hint = "vehicle"
                elif 1800 <= bbox_area <= 8500 and extent >= 0.55 and color_sig in ("rescue-orange-red", "safety-yellow"):
                    perim = max(cv2.arcLength(cnt, True), 1.0)
                    circularity = 4.0 * math.pi * c_area / (perim * perim)
                    cls_hint = "life_raft" if circularity >= 0.70 else "distress_marker"
                else:
                    cls_hint = "distress_marker" if color_sig == "rescue-orange-red" else "vehicle"

                cv_score = min(
                    0.95,
                    0.35 * min(1.0, mean_sat * 1.15)
                    + 0.35 * min(1.0, peak_sal * 1.35)
                    + 0.20 * edge_score
                    + 0.10 * min(1.0, extent * 1.2),
                )
                if source_name == "hsv":
                    cv_score = min(0.96, cv_score + 0.08)

                if cv_score >= 0.38:
                    proposals.append(
                        ([float(x), float(y), float(x + bw), float(y + bh)], round(cv_score, 4), cls_hint, color_sig)
                    )

        return _merge_close_proposals(proposals, iou_thresh=0.35)

    @staticmethod
    def _crop_context_patch(
        rgb_tile: np.ndarray,
        bbox: list[float],
        pad_ratio: float = 0.48,
        min_ctx: int = 56,
    ) -> np.ndarray:
        """Extract square context-padded RGB patch centered on candidate box for CNN classification."""
        th, tw = rgb_tile.shape[:2]
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        side = max(min_ctx, int(round(max(x2 - x1, y2 - y1) * (1.0 + pad_ratio))))

        sx1 = int(round(cx - side / 2.0))
        sy1 = int(round(cy - side / 2.0))
        sx2 = sx1 + side
        sy2 = sy1 + side

        if sx1 < 0:
            sx2 = min(tw, sx2 - sx1)
            sx1 = 0
        if sy1 < 0:
            sy2 = min(th, sy2 - sy1)
            sy1 = 0
        if sx2 > tw:
            sx1 = max(0, sx1 - (sx2 - tw))
            sx2 = tw
        if sy2 > th:
            sy1 = max(0, sy1 - (sy2 - th))
            sy2 = th

        return rgb_tile[sy1:sy2, sx1:sx2]

    def _sliding_window_onnx(
        self,
        rgb_tile: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[DetectionBox]:
        """Multi-scale sliding-window ONNX neural scan across tile with contrast gate."""
        if self.model is None or self.model.kind != "patch_cnn":
            return []

        th, tw = rgb_tile.shape[:2]
        windows: list[tuple[int, int, int, int]] = []
        patches: list[np.ndarray] = []

        for win_size, stride in ((54, 28), (94, 48)):
            if tw < win_size or th < win_size:
                continue
            xs = list(range(0, tw - win_size + 1, stride))
            if xs[-1] != tw - win_size:
                xs.append(tw - win_size)
            ys = list(range(0, th - win_size + 1, stride))
            if ys[-1] != th - win_size:
                ys.append(th - win_size)
            for y0 in ys:
                for x0 in xs:
                    crop = rgb_tile[y0 : y0 + win_size, x0 : x0 + win_size]
                    # Skip low-variance flat terrain patches
                    if float(np.std(crop)) < 16.0:
                        continue
                    windows.append((x0, y0, x0 + win_size, y0 + win_size))
                    patches.append(crop)

        if not patches:
            return []

        probs = self.model.classify_patches(patches)
        fg_probs = probs[:, 1:]
        best_fg_idx = np.argmax(fg_probs, axis=1) + 1
        best_fg_prob = np.max(fg_probs, axis=1)
        bg_prob = probs[:, 0]

        dets: list[DetectionBox] = []
        for win, cls_id, p_fg, p_bg in zip(windows, best_fg_idx, best_fg_prob, bg_prob):
            if p_fg < conf_threshold or p_bg >= 0.25 or p_fg <= p_bg:
                continue
            x1, y1, x2, y2 = win
            inset = 0.16 * (x2 - x1)
            dets.append(
                DetectionBox(
                    bbox=[round(x1 + inset, 2), round(y1 + inset, 2), round(x2 - inset, 2), round(y2 - inset, 2)],
                    confidence=round(float(p_fg), 4),
                    class_id=int(cls_id),
                    class_name=self.model.classes[int(cls_id)],
                    source="onnx-sliding",
                    onnx_score=round(float(p_fg), 4),
                    opencv_score=0.0,
                    color_signature="neural-pattern",
                )
            )
        return _nms_local(
            dets,
            iou_threshold=min(iou_threshold, 0.25),
            class_agnostic=True,
            min_center_dist=36.0,
        )

    def _detect_tile(
        self,
        bgr_tile: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        mode: str,
    ) -> list[DetectionBox]:
        """Run inference on a single tile using requested detector mode."""
        if mode == "onnx" and self.model is not None and self.model.kind == "yolo":
            return self.model.detect_yolo_tile(bgr_tile, conf_threshold, iou_threshold)

        rgb_tile = cv2.cvtColor(bgr_tile, cv2.COLOR_BGR2RGB)

        if mode == "onnx":
            if self.model is None:
                raise RuntimeError("No ONNX model loaded in models/ directory.")
            return self._sliding_window_onnx(rgb_tile, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

        proposals = self._extract_opencv_proposals(bgr_tile)

        if mode == "opencv" or self.model is None:
            cls_to_id = {name: idx for idx, name in enumerate(DEFAULT_CLASSES)}
            dets = [
                DetectionBox(
                    bbox=box,
                    confidence=score,
                    class_id=cls_to_id.get(hint, 1),
                    class_name=hint,
                    source="opencv-hsv-clahe",
                    onnx_score=0.0,
                    opencv_score=score,
                    color_signature=csig,
                )
                for box, score, hint, csig in proposals
                if score >= conf_threshold
            ]
            return _nms_local(dets, iou_threshold=iou_threshold)

        if self.model.kind == "yolo":
            yolo_dets = self.model.detect_yolo_tile(bgr_tile, conf_threshold, iou_threshold)
            cls_to_id = {name: idx for idx, name in enumerate(DEFAULT_CLASSES)}
            cv_dets = [
                DetectionBox(
                    bbox=box,
                    confidence=round(score * 0.85, 4),
                    class_id=cls_to_id.get(hint, 1),
                    class_name=hint,
                    source="opencv-hsv-clahe",
                    opencv_score=score,
                    color_signature=csig,
                )
                for box, score, hint, csig in proposals
                if score >= conf_threshold
            ]
            return _nms_local(yolo_dets + cv_dets, iou_threshold=iou_threshold, class_agnostic=True)

        # Dual-stage fusion: OpenCV ROI extraction + ONNX Runtime CNN classification
        dets: list[DetectionBox] = []
        if proposals:
            roi_patches = [self._crop_context_patch(rgb_tile, box) for box, _, _, _ in proposals]
            probs = self.model.classify_patches(roi_patches)

            for (box, cv_score, _, csig), prob_vec in zip(proposals, probs):
                p_bg = float(prob_vec[0])
                fg_vec = prob_vec[1:]
                cls_id = int(np.argmax(fg_vec)) + 1
                p_fg = float(fg_vec[cls_id - 1])
                cls_name = self.model.classes[cls_id]

                # Reject clutter regions classified as background by the CNN
                if p_bg > 0.50 or p_fg < 0.38:
                    continue

                fused_conf = round(min(0.99, 0.65 * p_fg + 0.35 * cv_score), 4)
                if fused_conf >= conf_threshold:
                    dets.append(
                        DetectionBox(
                            bbox=box,
                            confidence=fused_conf,
                            class_id=cls_id,
                            class_name=cls_name,
                            source="fusion",
                            onnx_score=round(p_fg, 4),
                            opencv_score=cv_score,
                            color_signature=csig,
                        )
                    )

        # Supplement only genuine ultra-high-confidence peaks missed by Stage-1 proposals
        sw_threshold = max(0.955, conf_threshold + 0.45)
        sw_dets = self._sliding_window_onnx(rgb_tile, conf_threshold=sw_threshold, iou_threshold=iou_threshold)
        for sw in sw_dets:
            already_covered = any(
                compute_iou(sw.bbox, existing.bbox) >= 0.12
                or _center_inside_expanded(sw.bbox, existing.bbox, expand=0.55)
                or _center_inside_expanded(existing.bbox, sw.bbox, expand=0.45)
                for existing in dets
            )
            if not already_covered:
                dets.append(sw)

        return _nms_local(dets, iou_threshold=iou_threshold, class_agnostic=True)

    def analyze_image(
        self,
        image_source: Path | str | bytes,
        tile_size: int | None = None,
        tile_overlap: float | None = None,
        conf_threshold: float = 0.45,
        nms_iou_threshold: float = 0.40,
        detector_mode: str = "fusion",
        model_name: str | None = None,
        agl_override_m: float | None = None,
        pitch_override_deg: float | None = None,
        hfov_override_deg: float | None = None,
        heading_override_deg: float | None = None,
    ) -> InferenceResult:
        """Tile high-res image, run dual-stage detection, apply global NMS, and raycast WGS84 coords."""
        t0 = time.perf_counter()
        if model_name and (self.model is None or self.model.model_name != model_name):
            self.load_model(model_name)

        if isinstance(image_source, (bytes, bytearray)):
            arr = np.frombuffer(image_source, dtype=np.uint8)
            bgr_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            bgr_img = cv2.imread(str(image_source), cv2.IMREAD_COLOR)

        if bgr_img is None:
            raise ValueError("Unable to decode aerial image.")

        img_h, img_w = bgr_img.shape[:2]
        telemetry = read_exif_telemetry(
            image_source,
            img_width=img_w,
            img_height=img_h,
            default_agl_m=agl_override_m if agl_override_m is not None else 120.0,
            default_pitch_deg=pitch_override_deg if pitch_override_deg is not None else -90.0,
            default_hfov_deg=hfov_override_deg if hfov_override_deg is not None else 64.0,
            default_heading_deg=heading_override_deg if heading_override_deg is not None else 0.0,
        )
        if agl_override_m is not None:
            telemetry.altitude_m = float(agl_override_m)
            telemetry.altitude_source = "operator_override"
        if pitch_override_deg is not None:
            telemetry.gimbal_pitch_deg = float(pitch_override_deg)
        if hfov_override_deg is not None:
            telemetry.hfov_deg = float(hfov_override_deg)
        if heading_override_deg is not None:
            telemetry.heading_deg = float(heading_override_deg) % 360.0

        tiler = ImageTiler(
            tile_size=tile_size or self.tile_size,
            overlap=tile_overlap if tile_overlap is not None else self.tile_overlap,
        )
        tiles = tiler.generate_tiles(img_w, img_h)

        mode = detector_mode.lower().strip()
        if mode not in ("fusion", "onnx", "opencv"):
            mode = "fusion"

        all_global: list[DetectionBox] = []
        raw_count = 0
        for tile in tiles:
            crop = tiler.crop_tile(bgr_img, tile)
            local_dets = self._detect_tile(
                crop,
                conf_threshold=conf_threshold,
                iou_threshold=nms_iou_threshold,
                mode=mode,
            )
            raw_count += len(local_dets)
            all_global.extend(tiler.project_to_global(local_dets, tile))

        merged = global_nms(all_global, iou_threshold=nms_iou_threshold, class_agnostic=True, use_wbf=True)
        # Drop any remaining sliding-window halo around higher-precision Stage-1 fusion boxes
        primary_boxes = [d for d in merged if d.source != "onnx-sliding"]
        if primary_boxes:
            merged = [
                d
                for d in merged
                if d.source != "onnx-sliding"
                or not any(
                    _center_inside_expanded(d.bbox, p.bbox, expand=0.55)
                    or _center_inside_expanded(p.bbox, d.bbox, expand=0.45)
                    for p in primary_boxes
                )
            ]

        geo_detections: list[GeoDetection] = []
        for det in merged:
            cx, cy = det.center
            proj = project_pixel_to_wgs84(cx, cy, img_w, img_h, telemetry)
            geo_detections.append(GeoDetection(box=det, geo=proj))

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        active_model_name = self.model.model_name if (self.model and mode != "opencv") else "opencv-hsv-clahe"

        return InferenceResult(
            image_width=img_w,
            image_height=img_h,
            tiles=tiles,
            detections=geo_detections,
            telemetry=telemetry,
            processing_time_ms=elapsed_ms,
            detector_mode=mode,
            model_name=active_model_name,
            raw_tile_detections_count=raw_count,
        )
