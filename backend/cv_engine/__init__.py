"""Computer Vision engine package for aerial SAR tiling, detection, geolocation, evaluation, and reporting."""

from .tiler import DetectionBox, ImageTiler, TileSpec, compute_iou, compute_iou_matrix, global_nms
from .exif_geo import ExifTelemetry, GeoProjection, project_pixel_to_wgs84, read_exif_telemetry

__all__ = [
    "DetectionBox",
    "ExifTelemetry",
    "GeoProjection",
    "ImageTiler",
    "TileSpec",
    "compute_iou",
    "compute_iou_matrix",
    "global_nms",
    "project_pixel_to_wgs84",
    "read_exif_telemetry",
]
