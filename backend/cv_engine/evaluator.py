"""COCO-format ground-truth evaluator computing IoU matrices, Precision, Recall, F1, and AP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .tiler import compute_iou_matrix


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute all-point interpolated Average Precision (COCO/VOC continuous PR AUC)."""
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _coco_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def evaluate_predictions(
    ground_truth: dict[str, Any] | Path | str,
    predictions_by_image: dict[str, list[dict[str, Any]]],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate predicted detections against COCO ground-truth annotations.

    predictions_by_image maps image file_name -> list of dicts with:
      {"class_name": str, "confidence": float, "bbox_xyxy": [x1, y1, x2, y2]}
    """
    if isinstance(ground_truth, (Path, str)):
        gt_data = json.loads(Path(ground_truth).read_text(encoding="utf-8"))
    else:
        gt_data = ground_truth

    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in gt_data.get("categories", [])}
    img_id_to_name = {int(im["id"]): str(im["file_name"]) for im in gt_data.get("images", [])}

    gt_by_image: dict[str, list[dict[str, Any]]] = {name: [] for name in img_id_to_name.values()}
    for ann in gt_data.get("annotations", []):
        img_name = img_id_to_name.get(int(ann["image_id"]))
        if img_name is None:
            continue
        cname = str(ann.get("category_name") or cat_id_to_name.get(int(ann["category_id"]), "unknown"))
        gt_by_image.setdefault(img_name, []).append(
            {
                "id": int(ann["id"]),
                "class_name": cname,
                "bbox_xyxy": _coco_xywh_to_xyxy(ann["bbox"]),
            }
        )

    classes = sorted({c["name"] for c in gt_data.get("categories", [])} | {g["class_name"] for lst in gt_by_image.values() for g in lst})

    # Per-class matching records for AP computation
    class_records: dict[str, list[tuple[float, bool]]] = {c: [] for c in classes}
    class_gt_counts: dict[str, int] = {c: 0 for c in classes}
    class_ious: dict[str, list[float]] = {c: [] for c in classes}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_ious: list[float] = []
    per_image_results: list[dict[str, Any]] = []

    for img_name, gts in gt_by_image.items():
        preds = sorted(predictions_by_image.get(img_name, []), key=lambda p: float(p["confidence"]), reverse=True)
        for g in gts:
            class_gt_counts[g["class_name"]] = class_gt_counts.get(g["class_name"], 0) + 1

        gt_boxes = np.array([g["bbox_xyxy"] for g in gts], dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)
        pred_boxes = np.array([p["bbox_xyxy"] for p in preds], dtype=np.float32) if preds else np.zeros((0, 4), dtype=np.float32)
        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

        matched_gt: set[int] = set()
        img_tp = 0
        img_fp = 0
        img_ious: list[float] = []
        match_details: list[dict[str, Any]] = []

        for p_idx, pred in enumerate(preds):
            p_cls = str(pred["class_name"])
            p_conf = float(pred["confidence"])

            best_gt_idx = -1
            best_iou = 0.0
            for g_idx, gt in enumerate(gts):
                if g_idx in matched_gt or gt["class_name"] != p_cls:
                    continue
                iou = float(iou_mat[g_idx, p_idx])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                img_tp += 1
                img_ious.append(best_iou)
                all_matched_ious.append(best_iou)
                class_records.setdefault(p_cls, []).append((p_conf, True))
                class_ious.setdefault(p_cls, []).append(best_iou)
                match_details.append(
                    {
                        "status": "TP",
                        "class_name": p_cls,
                        "confidence": round(p_conf, 4),
                        "iou": round(best_iou, 4),
                        "pred_bbox": pred["bbox_xyxy"],
                        "gt_bbox": gts[best_gt_idx]["bbox_xyxy"],
                    }
                )
            else:
                img_fp += 1
                class_records.setdefault(p_cls, []).append((p_conf, False))
                match_details.append(
                    {
                        "status": "FP",
                        "class_name": p_cls,
                        "confidence": round(p_conf, 4),
                        "iou": round(float(np.max(iou_mat[:, p_idx])) if len(gts) else 0.0, 4),
                        "pred_bbox": pred["bbox_xyxy"],
                        "gt_bbox": None,
                    }
                )

        img_fn = len(gts) - len(matched_gt)
        for g_idx, gt in enumerate(gts):
            if g_idx not in matched_gt:
                match_details.append(
                    {
                        "status": "FN",
                        "class_name": gt["class_name"],
                        "confidence": 0.0,
                        "iou": 0.0,
                        "pred_bbox": None,
                        "gt_bbox": gt["bbox_xyxy"],
                    }
                )

        total_tp += img_tp
        total_fp += img_fp
        total_fn += img_fn

        img_prec = img_tp / max(img_tp + img_fp, 1) if (img_tp + img_fp) > 0 else 0.0
        img_rec = img_tp / max(len(gts), 1) if len(gts) > 0 else 0.0
        img_f1 = 2 * img_prec * img_rec / (img_prec + img_rec) if (img_prec + img_rec) > 0 else 0.0
        per_image_results.append(
            {
                "file_name": img_name,
                "gt_count": len(gts),
                "pred_count": len(preds),
                "tp": img_tp,
                "fp": img_fp,
                "fn": img_fn,
                "precision": round(img_prec, 4),
                "recall": round(img_rec, 4),
                "f1": round(img_f1, 4),
                "mean_iou": round(float(np.mean(img_ious)), 4) if img_ious else 0.0,
                "matches": match_details,
            }
        )

    per_class: dict[str, dict[str, Any]] = {}
    ap_values: list[float] = []
    for cls_name in classes:
        recs = sorted(class_records.get(cls_name, []), key=lambda x: x[0], reverse=True)
        n_gt = class_gt_counts.get(cls_name, 0)
        tp_list = np.array([1.0 if is_tp else 0.0 for _, is_tp in recs], dtype=np.float32)
        fp_list = 1.0 - tp_list
        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        tp_c = int(tp_cum[-1]) if len(tp_cum) else 0
        fp_c = int(fp_cum[-1]) if len(fp_cum) else 0
        fn_c = max(0, n_gt - tp_c)

        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec_c = tp_c / n_gt if n_gt > 0 else 0.0
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0

        if n_gt > 0 and len(recs) > 0:
            rec_curve = tp_cum / float(n_gt)
            prec_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = _compute_ap(rec_curve, prec_curve)
            ap_values.append(ap)
        elif n_gt > 0:
            ap = 0.0
            ap_values.append(0.0)
        else:
            ap = 0.0

        ious = class_ious.get(cls_name, [])
        per_class[cls_name] = {
            "gt_count": n_gt,
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "precision": round(prec_c, 4),
            "recall": round(rec_c, 4),
            "f1": round(f1_c, 4),
            "ap": round(ap, 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        }

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2.0 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0.0
    )
    map_50 = float(np.mean(ap_values)) if ap_values else 0.0

    # Class-agnostic localization check
    loc_tp = 0
    loc_gt_total = 0
    loc_pred_total = 0
    for img_name, gts in gt_by_image.items():
        preds = sorted(predictions_by_image.get(img_name, []), key=lambda p: float(p["confidence"]), reverse=True)
        loc_gt_total += len(gts)
        loc_pred_total += len(preds)
        if not gts or not preds:
            continue
        iou_m = compute_iou_matrix(
            np.array([g["bbox_xyxy"] for g in gts], dtype=np.float32),
            np.array([p["bbox_xyxy"] for p in preds], dtype=np.float32),
        )
        used_g: set[int] = set()
        for p_idx in range(len(preds)):
            best_g = -1
            best_v = 0.0
            for g_idx in range(len(gts)):
                if g_idx not in used_g and float(iou_m[g_idx, p_idx]) >= iou_threshold and float(iou_m[g_idx, p_idx]) > best_v:
                    best_v = float(iou_m[g_idx, p_idx])
                    best_g = g_idx
            if best_g >= 0:
                used_g.add(best_g)
                loc_tp += 1

    loc_prec = loc_tp / loc_pred_total if loc_pred_total > 0 else 0.0
    loc_rec = loc_tp / loc_gt_total if loc_gt_total > 0 else 0.0

    return {
        "iou_threshold": iou_threshold,
        "summary": {
            "images_evaluated": len(gt_by_image),
            "total_ground_truth": total_tp + total_fn,
            "total_predictions": total_tp + total_fp,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(overall_prec, 4),
            "recall": round(overall_rec, 4),
            "f1_score": round(overall_f1, 4),
            "mAP": round(map_50, 4),
            "mean_iou": round(float(np.mean(all_matched_ious)), 4) if all_matched_ious else 0.0,
            "class_agnostic_precision": round(loc_prec, 4),
            "class_agnostic_recall": round(loc_rec, 4),
        },
        "per_class": per_class,
        "per_image": per_image_results,
    }
