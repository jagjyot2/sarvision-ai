"""Export COCO JSON, CSV logs, and SAR Mission Intelligence Reports (Markdown & HTML)."""

from __future__ import annotations

import csv
import html
import io
from datetime import datetime, timezone
from typing import Any

CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "sar_target"},
    {"id": 2, "name": "life_raft", "supercategory": "sar_target"},
    {"id": 3, "name": "vehicle", "supercategory": "sar_target"},
    {"id": 4, "name": "distress_marker", "supercategory": "sar_target"},
]
CAT_NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}


def _format_dms(dd: float | None, is_lat: bool) -> str:
    if dd is None:
        return "N/A"
    hemi = ("N" if dd >= 0 else "S") if is_lat else ("E" if dd >= 0 else "W")
    val = abs(dd)
    deg = int(val)
    rem = (val - deg) * 60.0
    mins = int(rem)
    secs = (rem - mins) * 60.0
    return f"{deg:02d}°{mins:02d}'{secs:05.2f}\"{hemi}"


def export_coco_json(analyses: list[dict[str, Any]], include_rejected: bool = False) -> dict[str, Any]:
    """Serialize one or more AnalysisRun payloads into COCO annotation JSON."""
    images = []
    annotations = []
    ann_id = 1

    for idx, run in enumerate(analyses, start=1):
        images.append(
            {
                "id": idx,
                "analysis_id": run["id"],
                "file_name": run["filename"],
                "width": run["image_width"],
                "height": run["image_height"],
                "uav_telemetry": {
                    "latitude": run.get("uav_lat"),
                    "longitude": run.get("uav_lon"),
                    "altitude_m": run.get("uav_alt_m"),
                    "heading_deg": run.get("heading_deg"),
                    "gimbal_pitch_deg": run.get("gimbal_pitch_deg"),
                    "gsd_cm_px": run.get("gsd_cm_px"),
                },
            }
        )
        for det in run.get("detections", []):
            status = det.get("verification_status", "pending")
            if not include_rejected and status == "false_positive":
                continue
            cls_name = det.get("verified_class_name") or det["class_name"]
            x1, y1, x2, y2 = float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": CAT_NAME_TO_ID.get(cls_name, 1),
                    "category_name": cls_name,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "score": det["confidence"],
                    "iscrowd": 0,
                    "sarvision_attributes": {
                        "detection_id": det["id"],
                        "verification_status": status,
                        "detector_source": det.get("detector_source"),
                        "wgs84_lat": det.get("wgs84_lat"),
                        "wgs84_lon": det.get("wgs84_lon"),
                        "operator_notes": det.get("operator_notes"),
                    },
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": "SARVision AI Operator-Verified Annotation Export",
            "version": "1.0.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def export_csv(analyses: list[dict[str, Any]]) -> str:
    """Serialize analysis detections into a flat CSV dispatch sheet."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "analysis_id",
            "filename",
            "detection_id",
            "class_name",
            "verified_class_name",
            "confidence",
            "onnx_score",
            "opencv_score",
            "detector_source",
            "x1",
            "y1",
            "x2",
            "y2",
            "wgs84_lat",
            "wgs84_lon",
            "north_offset_m",
            "east_offset_m",
            "slant_range_m",
            "verification_status",
            "operator_notes",
        ]
    )
    for run in analyses:
        for det in run.get("detections", []):
            writer.writerow(
                [
                    run["id"],
                    run["filename"],
                    det["id"],
                    det["class_name"],
                    det.get("verified_class_name") or "",
                    det["confidence"],
                    det.get("onnx_score", 0.0),
                    det.get("opencv_score", 0.0),
                    det.get("detector_source", ""),
                    det["x1"],
                    det["y1"],
                    det["x2"],
                    det["y2"],
                    det.get("wgs84_lat") if det.get("wgs84_lat") is not None else "",
                    det.get("wgs84_lon") if det.get("wgs84_lon") is not None else "",
                    det.get("north_offset_m") if det.get("north_offset_m") is not None else "",
                    det.get("east_offset_m") if det.get("east_offset_m") is not None else "",
                    det.get("slant_range_m") if det.get("slant_range_m") is not None else "",
                    det.get("verification_status", "pending"),
                    det.get("operator_notes") or "",
                ]
            )
    return buf.getvalue()


def generate_sar_mission_report_md(analyses: list[dict[str, Any]]) -> str:
    """Build a structured SAR Mission Intelligence Report in Markdown."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# SARVision AI — Aerial Search & Rescue Mission Intelligence Report",
        "",
        f"**Generated:** {now_str}  ",
        f"**Imagery Frames Analyzed:** {len(analyses)}",
        "",
        "---",
        "",
    ]

    all_priority: list[tuple[str, dict[str, Any]]] = []
    for run in analyses:
        dets = run.get("detections", [])
        confirmed = sum(1 for d in dets if d.get("verification_status") == "confirmed")
        escalated = sum(1 for d in dets if d.get("verification_status") == "escalated")
        fp = sum(1 for d in dets if d.get("verification_status") == "false_positive")
        pending = len(dets) - confirmed - escalated - fp

        lines.extend(
            [
                f"## Frame: `{run['filename']}` (Analysis `{run['id']}`)",
                "",
                f"- **Resolution / Tiling:** {run['image_width']}x{run['image_height']} px | "
                f"{run['tile_count']} tiles ({run['tile_size']}px, {int(round(run['tile_overlap'] * 100))}% overlap)",
                f"- **Detector Pipeline:** `{run['detector_mode']}` (`{run['model_name']}`) | "
                f"Conf >= {run['conf_threshold']:.2f}, NMS IoU <= {run['nms_iou_threshold']:.2f} | "
                f"Runtime: {run['processing_time_ms']:.1f} ms",
                f"- **UAV EXIF Telemetry:** Lat `{run.get('uav_lat')}` ({_format_dms(run.get('uav_lat'), True)}), "
                f"Lon `{run.get('uav_lon')}` ({_format_dms(run.get('uav_lon'), False)}), "
                f"AGL `{run.get('uav_alt_m')} m`, Pitch `{run.get('gimbal_pitch_deg')}°`, "
                f"Heading `{run.get('heading_deg')}°`, GSD `{run.get('gsd_cm_px')} cm/px`",
                f"- **Verification Tally:** Confirmed `{confirmed}` | Escalated `{escalated}` | Pending `{pending}` | False Positive `{fp}`",
                "",
                "| ID | Target Class | Conf | Pixel Box [x1,y1,x2,y2] | WGS84 Lat / Lon | Offset (N, E m) | Status | Notes |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )

        for det in dets:
            cls_label = det.get("verified_class_name") or det["class_name"]
            lat_s = f"{det['wgs84_lat']:.6f}" if det.get("wgs84_lat") is not None else "N/A"
            lon_s = f"{det['wgs84_lon']:.6f}" if det.get("wgs84_lon") is not None else "N/A"
            off_s = (
                f"{det['north_offset_m']:+.1f}m, {det['east_offset_m']:+.1f}m"
                if det.get("north_offset_m") is not None
                else "N/A"
            )
            box_s = f"[{int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])}]"
            notes_s = (det.get("operator_notes") or "").replace("|", "/")
            status = det.get("verification_status", "pending")
            lines.append(
                f"| #{det['id']} | **{cls_label}** | {det['confidence']:.2f} | `{box_s}` | "
                f"`{lat_s}, {lon_s}` | {off_s} | `{status}` | {notes_s} |"
            )
            if status in ("confirmed", "escalated", "pending"):
                all_priority.append((run["filename"], det))

        lines.append("")

    lines.extend(["## Priority Dispatch Coordinates", ""])
    if not all_priority:
        lines.append("No active non-rejected targets logged.")
    else:
        for fname, det in all_priority:
            cls_label = det.get("verified_class_name") or det["class_name"]
            if det.get("wgs84_lat") is not None:
                lines.append(
                    f"- **[{det.get('verification_status', 'pending').upper()}] {cls_label}** in `{fname}`: "
                    f"`{det['wgs84_lat']:.6f}, {det['wgs84_lon']:.6f}` "
                    f"({_format_dms(det['wgs84_lat'], True)} {_format_dms(det['wgs84_lon'], False)}) | "
                    f"Slant range {det.get('slant_range_m', 0):.1f} m | Conf {det['confidence']:.2f}"
                )
            else:
                lines.append(
                    f"- **[{det.get('verification_status', 'pending').upper()}] {cls_label}** in `{fname}`: "
                    f"Pixel center ({(det['x1']+det['x2'])/2:.0f}, {(det['y1']+det['y2'])/2:.0f}) — No GPS EXIF"
                )
    return "\n".join(lines) + "\n"


def generate_sar_mission_report_html(analyses: list[dict[str, Any]]) -> str:
    """Render a clean printable HTML SAR Mission Intelligence Report."""
    md = generate_sar_mission_report_md(analyses)
    escaped = html.escape(md)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SARVision AI Mission Intelligence Report</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0b111b; color: #e2e8f0; margin: 2rem auto; max-width: 980px; padding: 0 1rem; }}
  h1, h2 {{ color: #38bdf8; }}
  pre {{ background: #111a2b; border: 1px solid #1e293b; border-radius: 8px; padding: 1.25rem; white-space: pre-wrap; font-family: ui-monospace, Consolas, monospace; font-size: 0.9rem; line-height: 1.5; }}
</style>
</head>
<body>
<h1>SARVision AI — Mission Intelligence Report</h1>
<pre>{escaped}</pre>
</body>
</html>"""
