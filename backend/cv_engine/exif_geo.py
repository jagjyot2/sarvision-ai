"""EXIF GPS extraction and pinhole camera ground-plane WGS84 geolocation raycaster."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

# WGS84 ellipsoid constants
WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014


@dataclass(slots=True)
class ExifTelemetry:
    """Parsed aerial camera & GPS metadata needed for ground-plane projection."""

    has_gps_exif: bool
    latitude_deg: float | None
    longitude_deg: float | None
    altitude_m: float
    altitude_source: str
    heading_deg: float
    gimbal_pitch_deg: float
    hfov_deg: float
    focal_length_mm: float | None = None
    focal_35mm: int | None = None
    synthetic_flag: bool = False
    gsd_cm_px: float | None = None
    swath_width_m: float | None = None
    swath_height_m: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GeoProjection:
    """Projected ground target coordinates in WGS84 and local North/East meter offsets."""

    wgs84_lat: float
    wgs84_lon: float
    north_offset_m: float
    east_offset_m: float
    slant_range_m: float


def _to_float(val: Any) -> float:
    """Convert IFDRational, (num, den) tuple, or numeric scalar to float."""
    if isinstance(val, tuple) and len(val) == 2:
        num, den = val
        return float(num) / float(den) if float(den) != 0.0 else 0.0
    return float(val)


def _dms_to_decimal(dms: Any, ref: Any) -> float | None:
    """Convert EXIF DMS rational triplet + hemisphere ref to signed decimal degrees."""
    if dms is None or len(dms) != 3:
        return None
    deg = _to_float(dms[0])
    mins = _to_float(dms[1])
    secs = _to_float(dms[2])
    dd = deg + mins / 60.0 + secs / 3600.0
    ref_str = ref.decode("ascii", errors="ignore").strip().upper() if isinstance(ref, bytes) else str(ref).strip().upper()
    if ref_str in ("S", "W"):
        dd = -abs(dd)
    return round(dd, 8)


def _parse_user_comment(raw: Any) -> dict[str, Any]:
    """Extract optional JSON telemetry embedded in EXIF UserComment (0x9286) or ImageDescription."""
    if not raw:
        return {}
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)
    if text.startswith("ASCII"):
        text = text[5:].lstrip("\x00 ")
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def compute_gsd_and_swath(
    img_width: int,
    img_height: int,
    altitude_m: float,
    hfov_deg: float,
    gimbal_pitch_deg: float = -90.0,
) -> tuple[float, float, float]:
    """Compute Ground Sample Distance (cm/px) and ground swath (width_m, height_m) at image center."""
    hfov_rad = math.radians(max(5.0, min(160.0, hfov_deg)))
    aspect = img_height / max(img_width, 1)
    tilt_from_nadir = math.radians(max(-80.0, min(80.0, 90.0 + gimbal_pitch_deg)))
    slant_center = altitude_m / max(math.cos(tilt_from_nadir), 0.1)

    swath_w = 2.0 * slant_center * math.tan(hfov_rad / 2.0)
    vfov_rad = 2.0 * math.atan(aspect * math.tan(hfov_rad / 2.0))
    swath_h = 2.0 * slant_center * math.tan(vfov_rad / 2.0) / max(math.cos(tilt_from_nadir), 0.1)
    gsd_cm_px = (swath_w * 100.0) / max(img_width, 1)
    return round(gsd_cm_px, 2), round(swath_w, 2), round(swath_h, 2)


def read_exif_telemetry(
    image_source: Path | str | bytes,
    img_width: int | None = None,
    img_height: int | None = None,
    default_agl_m: float = 120.0,
    default_pitch_deg: float = -90.0,
    default_hfov_deg: float = 64.0,
    default_heading_deg: float = 0.0,
) -> ExifTelemetry:
    """Read EXIF GPS/camera tags from JPEG/TIFF and apply explicit telemetry overrides/fallbacks."""
    if isinstance(image_source, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_source))
    else:
        pil_img = Image.open(image_source)

    w, h = pil_img.size
    width = img_width or w
    height = img_height or h
    exif = pil_img.getexif()

    warnings: list[str] = []
    gps_ifd = exif.get_ifd(0x8825) if exif else {}
    exif_ifd = exif.get_ifd(0x8769) if exif else {}

    comment_meta = _parse_user_comment(exif_ifd.get(0x9286))
    desc = str(exif.get(0x010E, "")) if exif else ""
    synthetic_flag = bool(comment_meta.get("synthetic")) or ("SYNTHETIC" in desc.upper())

    lat = _dms_to_decimal(gps_ifd.get(2), gps_ifd.get(1, "N")) if gps_ifd else None
    lon = _dms_to_decimal(gps_ifd.get(4), gps_ifd.get(3, "E")) if gps_ifd else None
    has_gps = lat is not None and lon is not None

    if not has_gps:
        warnings.append("No GPS EXIF tags found; WGS84 projection disabled unless UAV coordinates provided.")

    alt_m = float(default_agl_m)
    alt_source = "default_agl"
    if "agl_m" in comment_meta:
        alt_m = float(comment_meta["agl_m"])
        alt_source = "exif_telemetry_agl"
    elif gps_ifd and 6 in gps_ifd:
        raw_alt = _to_float(gps_ifd[6])
        ref_alt = gps_ifd.get(5, 0)
        is_below = ref_alt in (1, b"\x01")
        alt_m = max(5.0, abs(raw_alt) if not is_below else default_agl_m)
        alt_source = "exif_gps_altitude"

    heading_deg = float(default_heading_deg)
    if "heading_deg" in comment_meta:
        heading_deg = float(comment_meta["heading_deg"])
    elif gps_ifd and 17 in gps_ifd:
        heading_deg = _to_float(gps_ifd[17])

    gimbal_pitch_deg = float(comment_meta.get("gimbal_pitch_deg", default_pitch_deg))

    focal_mm = _to_float(exif_ifd[0x920A]) if exif_ifd and 0x920A in exif_ifd else None
    focal_35 = int(exif_ifd[0xA405]) if exif_ifd and 0xA405 in exif_ifd else None
    if "focal_35mm" in comment_meta and focal_35 is None:
        focal_35 = int(comment_meta["focal_35mm"])

    if "hfov_deg" in comment_meta:
        hfov_deg = float(comment_meta["hfov_deg"])
    elif focal_35 and focal_35 > 0:
        # 36mm full-frame horizontal sensor width
        hfov_deg = math.degrees(2.0 * math.atan(36.0 / (2.0 * float(focal_35))))
    else:
        hfov_deg = float(default_hfov_deg)

    gsd_cm, swath_w, swath_h = compute_gsd_and_swath(width, height, alt_m, hfov_deg, gimbal_pitch_deg)

    return ExifTelemetry(
        has_gps_exif=has_gps,
        latitude_deg=lat,
        longitude_deg=lon,
        altitude_m=round(alt_m, 2),
        altitude_source=alt_source,
        heading_deg=round(heading_deg % 360.0, 2),
        gimbal_pitch_deg=round(gimbal_pitch_deg, 2),
        hfov_deg=round(hfov_deg, 2),
        focal_length_mm=round(focal_mm, 2) if focal_mm is not None else None,
        focal_35mm=focal_35,
        synthetic_flag=synthetic_flag,
        gsd_cm_px=gsd_cm,
        swath_width_m=swath_w,
        swath_height_m=swath_h,
        warnings=warnings,
    )


def project_pixel_to_wgs84(
    px: float,
    py: float,
    img_width: int,
    img_height: int,
    telemetry: ExifTelemetry,
) -> GeoProjection | None:
    """Project pixel coordinate (px, py) onto ground plane z=0 and convert to WGS84 lat/lon.

    Uses a pinhole camera model rotated from camera frame (Xc right, Yc down, Zc optical)
    into local NED frame via gimbal pitch (where -90 deg is nadir straight down) and
    clockwise UAV heading, intersected with horizontal ground plane at depth = altitude_m.
    """
    if telemetry.latitude_deg is None or telemetry.longitude_deg is None:
        return None
    if telemetry.altitude_m <= 0.0 or img_width <= 0 or img_height <= 0:
        return None

    cx = img_width / 2.0
    cy = img_height / 2.0
    hfov_rad = math.radians(max(5.0, min(160.0, telemetry.hfov_deg)))
    f_px = (img_width / 2.0) / math.tan(hfov_rad / 2.0)

    # Normalized ray in camera coordinates
    xc = (px - cx) / f_px
    yc = (py - cy) / f_px
    zc = 1.0
    norm = math.sqrt(xc * xc + yc * yc + zc * zc)
    xc, yc, zc = xc / norm, yc / norm, zc / norm

    # Tilt from nadir: 0 rad at -90 deg pitch (nadir), +pi/4 at -45 deg forward look
    tilt = math.radians(90.0 + telemetry.gimbal_pitch_deg)
    cos_t = math.cos(tilt)
    sin_t = math.sin(tilt)

    n_unrot = zc * sin_t - yc * cos_t
    e_unrot = xc
    d_ned = zc * cos_t + yc * sin_t

    # Ray points at or above horizon -> no ground intersection
    if d_ned <= 1e-3:
        return None

    # Rotate North/East vector clockwise by UAV heading
    psi = math.radians(telemetry.heading_deg)
    cos_p = math.cos(psi)
    sin_p = math.sin(psi)

    n_dir = n_unrot * cos_p - e_unrot * sin_p
    e_dir = n_unrot * sin_p + e_unrot * cos_p

    scale = telemetry.altitude_m / d_ned
    north_m = n_dir * scale
    east_m = e_dir * scale
    slant_range_m = math.sqrt(north_m * north_m + east_m * east_m + telemetry.altitude_m ** 2)

    lat0_rad = math.radians(telemetry.latitude_deg)
    sin_lat = math.sin(lat0_rad)
    denom = 1.0 - WGS84_E2 * (sin_lat * sin_lat)
    meridional_r = WGS84_A * (1.0 - WGS84_E2) / (denom ** 1.5)
    prime_vert_r = WGS84_A / math.sqrt(denom)

    delta_lat = math.degrees(north_m / meridional_r)
    delta_lon = math.degrees(east_m / (prime_vert_r * max(math.cos(lat0_rad), 1e-6)))

    return GeoProjection(
        wgs84_lat=round(telemetry.latitude_deg + delta_lat, 7),
        wgs84_lon=round(telemetry.longitude_deg + delta_lon, 7),
        north_offset_m=round(north_m, 2),
        east_offset_m=round(east_m, 2),
        slant_range_m=round(slant_range_m, 2),
    )
