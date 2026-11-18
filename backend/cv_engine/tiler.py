"""High-resolution aerial image tiler, coordinate mapper, and cross-tile global NMS / WBF."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TileSpec:
    """Metadata describing one tile crop inside a full-resolution image."""

    tile_id: int
    row: int
    col: int
    x_offset: int
    y_offset: int
    width: int
    height: int
    image_width: int
    image_height: int


@dataclass(slots=True)
class DetectionBox:
    """Unified detection bounding box in [x1, y1, x2, y2] pixel coordinates."""

    bbox: list[float]
    confidence: float
    class_id: int
    class_name: str
    source: str = "fusion"
    onnx_score: float = 0.0
    opencv_score: float = 0.0
    tile_id: int | None = None
    color_signature: str = ""
    local_touching_seam: bool = False

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.bbox[0] + self.bbox[2]), 0.5 * (self.bbox[1] + self.bbox[3]))


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix of shape (N, M) for [x1, y1, x2, y2] arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / union, 0.0).astype(np.float32)


def _compute_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Compute sliding tile offsets covering [0, length] cleanly."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if not starts or starts[-1] < last:
        starts.append(last)
    return starts


class ImageTiler:
    """Splits large aerial images into overlapping tiles and maps detections back globally."""

    def __init__(self, tile_size: int = 640, overlap: float = 0.20) -> None:
        if tile_size < 64:
            raise ValueError(f"tile_size must be >= 64, got {tile_size}")
        if not (0.0 <= overlap < 0.9):
            raise ValueError(f"overlap must be in [0.0, 0.9), got {overlap}")
        self.tile_size = int(tile_size)
        self.overlap = float(overlap)
        self.stride = max(16, int(round(self.tile_size * (1.0 - self.overlap))))

    def generate_tiles(self, image_width: int, image_height: int) -> list[TileSpec]:
        """Generate tile grid covering the full image dimensions."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_width}x{image_height}")

        xs = _compute_starts(image_width, self.tile_size, self.stride)
        ys = _compute_starts(image_height, self.tile_size, self.stride)

        tiles: list[TileSpec] = []
        tid = 0
        for r, y0 in enumerate(ys):
            th = min(self.tile_size, image_height - y0)
            for c, x0 in enumerate(xs):
                tw = min(self.tile_size, image_width - x0)
                tiles.append(
                    TileSpec(
                        tile_id=tid,
                        row=r,
                        col=c,
                        x_offset=x0,
                        y_offset=y0,
                        width=tw,
                        height=th,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )
                tid += 1
        return tiles

    @staticmethod
    def crop_tile(image: np.ndarray, tile: TileSpec) -> np.ndarray:
        """Return image slice corresponding to TileSpec."""
        return image[tile.y_offset : tile.y_offset + tile.height, tile.x_offset : tile.x_offset + tile.width]

    @staticmethod
    def project_to_global(detections: list[DetectionBox], tile: TileSpec) -> list[DetectionBox]:
        """Project tile-local [x1, y1, x2, y2] detections into full-image coordinates."""
        projected: list[DetectionBox] = []
        edge_tol = 3.0
        for det in detections:
            lx1, ly1, lx2, ly2 = [float(v) for v in det.bbox]
            touches_internal_seam = (
                (lx1 <= edge_tol and tile.x_offset > 0)
                or (lx2 >= tile.width - edge_tol and (tile.x_offset + tile.width) < tile.image_width)
                or (ly1 <= edge_tol and tile.y_offset > 0)
                or (ly2 >= tile.height - edge_tol and (tile.y_offset + tile.height) < tile.image_height)
            )
            gx1 = max(0.0, min(float(tile.image_width), lx1 + tile.x_offset))
            gy1 = max(0.0, min(float(tile.image_height), ly1 + tile.y_offset))
            gx2 = max(0.0, min(float(tile.image_width), lx2 + tile.x_offset))
            gy2 = max(0.0, min(float(tile.image_height), ly2 + tile.y_offset))
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            projected.append(
                replace(
                    det,
                    bbox=[round(gx1, 2), round(gy1, 2), round(gx2, 2), round(gy2, 2)],
                    tile_id=tile.tile_id,
                    local_touching_seam=touches_internal_seam,
                )
            )
        return projected


def suppress_seam_fragments(
    detections: list[DetectionBox],
    containment_threshold: float = 0.68,
    class_agnostic: bool = False,
) -> list[DetectionBox]:
    """Suppress clipped detections touching internal tile seams when largely contained in a full box."""
    if len(detections) <= 1:
        return list(detections)

    keep = [True] * len(detections)
    for i, det_i in enumerate(detections):
        if not det_i.local_touching_seam or det_i.area <= 0:
            continue
        bx1, by1, bx2, by2 = det_i.bbox
        for j, det_j in enumerate(detections):
            if i == j or not keep[j]:
                continue
            if not class_agnostic and det_i.class_name != det_j.class_name:
                continue
            # Prefer full non-seam boxes or notably larger boxes containing the clipped fragment
            if det_j.local_touching_seam and det_j.area <= det_i.area:
                continue
            if det_j.area < det_i.area * 1.10 and det_j.local_touching_seam:
                continue
            ix1 = max(bx1, det_j.bbox[0])
            iy1 = max(by1, det_j.bbox[1])
            ix2 = min(bx2, det_j.bbox[2])
            iy2 = min(by2, det_j.bbox[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            containment = inter / max(det_i.area, 1e-6)
            if containment >= containment_threshold:
                keep[i] = False
                break
    return [d for d, k in zip(detections, keep) if k]


def _greedy_nms(detections: list[DetectionBox], iou_threshold: float, class_agnostic: bool) -> list[DetectionBox]:
    """Standard greedy Non-Maximum Suppression sorted by descending confidence."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[DetectionBox] = []
    for cand in ordered:
        suppressed = False
        for prev in kept:
            if not class_agnostic and cand.class_name != prev.class_name:
                continue
            if compute_iou(cand.bbox, prev.bbox) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    return kept


def _weighted_box_fusion(
    detections: list[DetectionBox],
    iou_threshold: float,
    class_agnostic: bool,
) -> list[DetectionBox]:
    """Cluster overlapping cross-tile detections and fuse coordinates by confidence weights."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    clusters: list[list[DetectionBox]] = []

    for cand in ordered:
        matched_cluster: list[DetectionBox] | None = None
        best_iou = 0.0
        for cluster in clusters:
            rep = cluster[0]
            if not class_agnostic and cand.class_name != rep.class_name:
                continue
            # Compare against confidence-weighted cluster centroid box
            w_sum = sum(d.confidence for d in cluster)
            avg_box = [
                sum(d.bbox[k] * d.confidence for d in cluster) / max(w_sum, 1e-6)
                for k in range(4)
            ]
            iou = compute_iou(cand.bbox, avg_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                matched_cluster = cluster
        if matched_cluster is not None:
            matched_cluster.append(cand)
        else:
            clusters.append([cand])

    fused: list[DetectionBox] = []
    for cluster in clusters:
        best = max(cluster, key=lambda d: d.confidence)
        w_sum = sum(max(d.confidence, 1e-4) for d in cluster)
        fused_box = [
            round(sum(d.bbox[k] * max(d.confidence, 1e-4) for d in cluster) / w_sum, 2)
            for k in range(4)
        ]
        # Slight consensus boost when independent tiles corroborate the same object (clamped <= 0.99)
        distinct_tiles = len({d.tile_id for d in cluster if d.tile_id is not None})
        conf = best.confidence
        if distinct_tiles > 1:
            conf = min(0.99, conf + 0.03 * min(distinct_tiles - 1, 2))
        fused.append(
            replace(
                best,
                bbox=fused_box,
                confidence=round(conf, 4),
                onnx_score=round(max(d.onnx_score for d in cluster), 4),
                opencv_score=round(max(d.opencv_score for d in cluster), 4),
                local_touching_seam=all(d.local_touching_seam for d in cluster),
            )
        )
    return sorted(fused, key=lambda d: d.confidence, reverse=True)


def global_nms(
    detections: list[DetectionBox],
    iou_threshold: float = 0.40,
    class_agnostic: bool = False,
    use_wbf: bool = True,
) -> list[DetectionBox]:
    """Suppress cross-tile seam fragments and merge overlapping global detections."""
    if not detections:
        return []
    cleaned = suppress_seam_fragments(detections, containment_threshold=0.68, class_agnostic=class_agnostic)
    if use_wbf:
        return _weighted_box_fusion(cleaned, iou_threshold=iou_threshold, class_agnostic=class_agnostic)
    return _greedy_nms(cleaned, iou_threshold=iou_threshold, class_agnostic=class_agnostic)
