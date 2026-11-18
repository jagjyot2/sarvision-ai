"""Generate high-resolution synthetic aerial SAR evaluation imagery with EXIF GPS metadata.

Creates 3 realistic 1920x1280 synthetic aerial SAR scenes inside sample_data/:
  1. sar_wilderness_clearing.jpg  (Cascade Range forest & trail clearing)
  2. sar_maritime_openwater.jpg   (Monterey Bay offshore swell & whitecaps)
  3. sar_alpine_snowfield.jpg     (Mt. Rainier glacial snowfield & rock ridge)

Also writes COCO-format ground_truth.json clearly labeled as a synthetic
evaluation dataset. Run directly:
    python scripts/generate_sar_test_imagery.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "sample_data"

CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "sar_target"},
    {"id": 2, "name": "life_raft", "supercategory": "sar_target"},
    {"id": 3, "name": "vehicle", "supercategory": "sar_target"},
    {"id": 4, "name": "distress_marker", "supercategory": "sar_target"},
]
NAME_TO_CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}


def _fbm_noise(height: int, width: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    """Generate smooth multi-octave 2D fractional Brownian motion noise in [0, 1]."""
    acc = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    total_amp = 0.0
    for i in range(octaves):
        grid_h = max(4, height // (2 ** (octaves - i + 1)))
        grid_w = max(4, width // (2 ** (octaves - i + 1)))
        low = rng.random((grid_h, grid_w), dtype=np.float32)
        up = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
        acc += up * amp
        total_amp += amp
        amp *= 0.52
    acc /= max(total_amp, 1e-6)
    acc = (acc - acc.min()) / max(float(acc.max() - acc.min()), 1e-6)
    return acc


def _dd_to_dms_rationals(decimal_deg: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    """Convert decimal degrees to EXIF DMS rational tuple."""
    val = abs(decimal_deg)
    deg = int(val)
    rem = (val - deg) * 60.0
    mins = int(rem)
    secs = round((rem - mins) * 60.0, 4)
    if secs >= 60.0:
        secs -= 60.0
        mins += 1
    if mins >= 60:
        mins -= 60
        deg += 1
    return (
        IFDRational(deg, 1),
        IFDRational(mins, 1),
        IFDRational(int(round(secs * 10000)), 10000),
    )


def _embed_exif_and_save(
    bgr_img: np.ndarray,
    out_path: Path,
    lat: float,
    lon: float,
    agl_m: float,
    heading_deg: float,
    gimbal_pitch_deg: float,
    focal_35mm: int,
    scene_label: str,
) -> None:
    """Encode real JPEG EXIF GPS + camera telemetry tags and save image."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    exif = pil_img.getexif()

    telemetry = {
        "synthetic": True,
        "dataset": "SARVision Synthetic Evaluation Benchmark",
        "scene": scene_label,
        "agl_m": agl_m,
        "gimbal_pitch_deg": gimbal_pitch_deg,
        "heading_deg": heading_deg,
        "focal_35mm": focal_35mm,
    }
    exif[0x010E] = f"SYNTHETIC SARVISION SCENE: {scene_label}"
    exif[0x010F] = "SARVision-Sim"
    exif[0x0110] = "UAV-Mapper-24mm"
    exif[0x0131] = "generate_sar_test_imagery.py"

    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x920A] = IFDRational(24, 1)
    exif_ifd[0xA405] = int(focal_35mm)
    exif_ifd[0x9286] = b"ASCII\x00\x00\x00" + json.dumps(telemetry).encode("ascii")

    gps_ifd = exif.get_ifd(0x8825)
    gps_ifd[1] = "N" if lat >= 0 else "S"
    gps_ifd[2] = _dd_to_dms_rationals(lat)
    gps_ifd[3] = "E" if lon >= 0 else "W"
    gps_ifd[4] = _dd_to_dms_rationals(lon)
    gps_ifd[5] = 0
    gps_ifd[6] = IFDRational(int(round(agl_m * 100)), 100)
    gps_ifd[16] = "T"
    gps_ifd[17] = IFDRational(int(round((heading_deg % 360.0) * 100)), 100)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path, format="JPEG", quality=93, exif=exif.tobytes())


def _draw_rotated_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    alpha: np.ndarray,
    cx: int,
    cy: int,
    angle_deg: float,
) -> tuple[int, int, int, int]:
    """Composite an RGBA-style patch onto canvas with rotation, returning tight bbox [x1,y1,x2,y2]."""
    ph, pw = patch.shape[:2]
    pad = int(math.ceil(math.hypot(pw, ph))) + 8
    padded_p = np.zeros((pad, pad, 3), dtype=np.uint8)
    padded_a = np.zeros((pad, pad), dtype=np.float32)

    y0 = (pad - ph) // 2
    x0 = (pad - pw) // 2
    padded_p[y0 : y0 + ph, x0 : x0 + pw] = patch
    padded_a[y0 : y0 + ph, x0 : x0 + pw] = alpha

    M = cv2.getRotationMatrix2D((pad / 2.0, pad / 2.0), angle_deg, 1.0)
    rot_p = cv2.warpAffine(padded_p, M, (pad, pad), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rot_a = cv2.warpAffine(padded_a, M, (pad, pad), flags=cv2.INTER_LINEAR, borderValue=0.0)

    x_start = cx - pad // 2
    y_start = cy - pad // 2
    x_end = x_start + pad
    y_end = y_start + pad

    H, W = canvas.shape[:2]
    cx1, cy1 = max(0, x_start), max(0, y_start)
    cx2, cy2 = min(W, x_end), min(H, y_end)

    px1, py1 = cx1 - x_start, cy1 - y_start
    px2, py2 = px1 + (cx2 - cx1), py1 + (cy2 - cy1)

    a_crop = rot_a[py1:py2, px1:px2, None]
    roi = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
    blended = roi * (1.0 - a_crop) + rot_p[py1:py2, px1:px2].astype(np.float32) * a_crop
    canvas[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)

    ys, xs = np.where(rot_a[py1:py2, px1:px2] > 0.28)
    if len(xs) == 0:
        return [cx - 10, cy - 10, cx + 10, cy + 10]
    return [
        int(cx1 + xs.min()),
        int(cy1 + ys.min()),
        int(cx1 + xs.max() + 1),
        int(cy1 + ys.max() + 1),
    ]


def _add_cast_shadow(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    dx: int = 10,
    dy: int = 12,
    strength: float = 0.42,
) -> None:
    """Render soft directional ground shadow beneath an aerial object."""
    mask = np.zeros(canvas.shape[:2], dtype=np.float32)
    cv2.ellipse(mask, (cx + dx, cy + dy), (max(4, rx), max(4, ry)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 4.0)
    canvas[:] = np.clip(canvas.astype(np.float32) * (1.0 - strength * mask[:, :, None]), 0, 255).astype(np.uint8)


def _render_person(
    jacket_bgr: tuple[int, int, int],
    pack_bgr: tuple[int, int, int] = (45, 48, 52),
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-down aerial person patch (head/hood, high-vis shoulders/torso, backpack, legs)."""
    s = scale
    w, h = int(round(36 * s)), int(round(44 * s))
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)

    cx, cy = w // 2, h // 2
    # Extended arm / legs silhouette
    cv2.ellipse(patch, (cx, cy + int(5 * s)), (int(8 * s), int(13 * s)), 0, 0, 360, (55, 60, 70), -1)
    cv2.ellipse(alpha, (cx, cy + int(5 * s)), (int(8 * s), int(13 * s)), 0, 0, 360, 0.95, -1)

    # High-vis jacket shoulders & arms
    cv2.ellipse(patch, (cx, cy - int(2 * s)), (int(13 * s), int(8 * s)), 0, 0, 360, jacket_bgr, -1)
    cv2.ellipse(alpha, (cx, cy - int(2 * s)), (int(13 * s), int(8 * s)), 0, 0, 360, 1.0, -1)

    # Backpack
    cv2.rectangle(
        patch,
        (cx - int(6 * s), cy + int(1 * s)),
        (cx + int(6 * s), cy + int(10 * s)),
        pack_bgr,
        -1,
    )
    cv2.rectangle(
        alpha,
        (cx - int(6 * s), cy + int(1 * s)),
        (cx + int(6 * s), cy + int(10 * s)),
        1.0,
        -1,
    )

    # Head / helmet / hood
    head_color = (min(255, jacket_bgr[0] + 25), min(255, jacket_bgr[1] + 25), min(255, jacket_bgr[2] + 25))
    cv2.circle(patch, (cx, cy - int(5 * s)), max(3, int(5 * s)), head_color, -1)
    cv2.circle(alpha, (cx, cy - int(5 * s)), max(3, int(5 * s)), 1.0, -1)
    return patch, alpha


def _render_life_raft(
    tube_bgr: tuple[int, int, int] = (22, 112, 245),
    canopy_bgr: tuple[int, int, int] = (30, 138, 250),
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-down SOLAS inflatable life raft with octagonal/oval buoyancy tube and canopy."""
    s = scale
    w, h = int(round(92 * s)), int(round(72 * s))
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)

    cx, cy = w // 2, h // 2
    # Outer buoyancy tube
    cv2.ellipse(patch, (cx, cy), (int(38 * s), int(28 * s)), 0, 0, 360, tube_bgr, -1)
    cv2.ellipse(alpha, (cx, cy), (int(38 * s), int(28 * s)), 0, 0, 360, 1.0, -1)
    # Grab-line dark rim
    cv2.ellipse(patch, (cx, cy), (int(35 * s), int(25 * s)), 0, 0, 360, (25, 35, 50), max(1, int(2 * s)))
    # Canopy top & retro-reflective tape strips
    cv2.ellipse(patch, (cx, cy), (int(28 * s), int(20 * s)), 0, 0, 360, canopy_bgr, -1)
    cv2.ellipse(patch, (cx + int(10 * s), cy), (int(9 * s), int(12 * s)), 0, 0, 360, (28, 34, 42), -1)
    cv2.line(patch, (cx - int(22 * s), cy), (cx - int(8 * s), cy), (235, 245, 250), max(2, int(3 * s)))
    cv2.line(patch, (cx, cy - int(17 * s)), (cx, cy - int(7 * s)), (235, 245, 250), max(2, int(3 * s)))
    return patch, alpha


def _render_vehicle(
    body_bgr: tuple[int, int, int] = (46, 54, 148),
    roof_bgr: tuple[int, int, int] = (175, 182, 188),
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-down aerial 4x4 overland vehicle / SAR snowcat with windshield, roof rack, tires."""
    s = scale
    w, h = int(round(108 * s)), int(round(56 * s))
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)

    # Four protruding tires
    for wx in (int(22 * s), int(82 * s)):
        for wy in (int(6 * s), int(46 * s)):
            cv2.rectangle(patch, (wx - int(8 * s), wy - int(3 * s)), (wx + int(8 * s), wy + int(3 * s)), (24, 24, 26), -1)
            cv2.rectangle(alpha, (wx - int(8 * s), wy - int(3 * s)), (wx + int(8 * s), wy + int(3 * s)), 1.0, -1)

    # Chassis body
    cv2.rectangle(patch, (int(8 * s), int(9 * s)), (int(98 * s), int(47 * s)), body_bgr, -1)
    cv2.rectangle(alpha, (int(8 * s), int(9 * s)), (int(98 * s), int(47 * s)), 1.0, -1)
    # Hood panel lines
    cv2.rectangle(patch, (int(68 * s), int(12 * s)), (int(95 * s), int(44 * s)), (max(0, body_bgr[0] - 12), max(0, body_bgr[1] - 12), max(0, body_bgr[2] - 12)), -1)
    # Dark reflective windshield & rear glass
    cv2.rectangle(patch, (int(56 * s), int(12 * s)), (int(68 * s), int(44 * s)), (36, 48, 62), -1)
    cv2.rectangle(patch, (int(12 * s), int(13 * s)), (int(20 * s), int(43 * s)), (32, 42, 54), -1)
    # Roof & roof rack rails
    cv2.rectangle(patch, (int(21 * s), int(12 * s)), (int(55 * s), int(44 * s)), roof_bgr, -1)
    cv2.rectangle(patch, (int(24 * s), int(15 * s)), (int(52 * s), int(41 * s)), (40, 44, 48), max(1, int(2 * s)))
    return patch, alpha


def _render_distress_marker(
    panel_bgr: tuple[int, int, int] = (20, 108, 250),
    cross_bgr: tuple[int, int, int] = (240, 248, 255),
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """High-visibility aerial ground SOS tarp / VS-17 distress panel with X cross."""
    s = scale
    w, h = int(round(68 * s)), int(round(60 * s))
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)

    cv2.rectangle(patch, (int(4 * s), int(4 * s)), (w - int(4 * s), h - int(4 * s)), panel_bgr, -1)
    cv2.rectangle(alpha, (int(4 * s), int(4 * s)), (w - int(4 * s), h - int(4 * s)), 1.0, -1)
    # High-contrast X cross + border tie-downs
    t = max(2, int(5 * s))
    cv2.line(patch, (int(11 * s), int(11 * s)), (w - int(11 * s), h - int(11 * s)), cross_bgr, t)
    cv2.line(patch, (w - int(11 * s), int(11 * s)), (int(11 * s), h - int(11 * s)), cross_bgr, t)
    cv2.rectangle(patch, (int(4 * s), int(4 * s)), (w - int(4 * s), h - int(4 * s)), (35, 35, 35), max(1, int(2 * s)))
    return patch, alpha


def _build_wilderness_scene(rng: np.random.Generator, W: int = 1920, H: int = 1280) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate 1920x1280 wilderness forest + clearing scene with 4 SAR targets."""
    base = _fbm_noise(H, W, rng, octaves=6)
    detail = _fbm_noise(H, W, rng, octaves=4)

    img = np.zeros((H, W, 3), dtype=np.float32)
    # Deep pine green terrain base
    img[:, :, 0] = 28 + 30 * base + 14 * detail  # B
    img[:, :, 1] = 56 + 48 * base + 22 * detail  # G
    img[:, :, 2] = 32 + 28 * base + 15 * detail  # R

    # Dirt clearing & winding trail near center-left and center-right
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    clearing1 = np.exp(-(((xx - 640) / 290) ** 2 + ((yy - 630) / 230) ** 2))
    clearing2 = np.exp(-(((xx - 1350) / 230) ** 2 + ((yy - 410) / 180) ** 2))
    trail_y = 660 + 130 * np.sin(xx / 310.0)
    trail = np.exp(-(((yy - trail_y) / 42.0) ** 2))
    dirt_mask = np.clip(clearing1 * 0.85 + clearing2 * 0.75 + trail * 0.75, 0.0, 1.0)
    dirt_tex = 0.75 + 0.45 * detail
    img[:, :, 0] = img[:, :, 0] * (1 - dirt_mask) + (68 * dirt_tex) * dirt_mask
    img[:, :, 1] = img[:, :, 1] * (1 - dirt_mask) + (96 * dirt_tex) * dirt_mask
    img[:, :, 2] = img[:, :, 2] * (1 - dirt_mask) + (116 * dirt_tex) * dirt_mask

    canvas = np.clip(img, 0, 255).astype(np.uint8)

    # Add textured pine tree canopies outside clearings + subtle autumn shrub clutter
    for _ in range(320):
        tx = int(rng.integers(20, W - 20))
        ty = int(rng.integers(20, H - 20))
        if dirt_mask[ty, tx] > 0.42:
            continue
        r = int(rng.integers(18, 44))
        cv2.circle(canvas, (tx + 8, ty + 10), r, (18, 30, 18), -1)
        tree_g = int(rng.integers(58, 98))
        cv2.circle(canvas, (tx, ty), r, (24, tree_g, 28), -1)
        cv2.circle(canvas, (tx - r // 4, ty - r // 4), max(4, r // 2), (32, min(125, tree_g + 18), 36), -1)

    # Small muted autumn foliage speckles (realistic hard negatives, kept small < 9px)
    for _ in range(65):
        fx = int(rng.integers(30, W - 30))
        fy = int(rng.integers(30, H - 30))
        cv2.circle(canvas, (fx, fy), int(rng.integers(3, 7)), (34, 82, 132), -1)

    targets = []

    # Target 1: Person in high-vis orange right near horizontal tile seam (x=632 -> straddles 640!)
    _add_cast_shadow(canvas, 632, 575, 14, 18)
    p1, a1 = _render_person((22, 118, 250), scale=1.05)
    bbox1 = _draw_rotated_patch(canvas, p1, a1, 632, 575, 18.0)
    targets.append({"class_name": "person", "bbox_xyxy": bbox1})

    # Target 2: Stranded 4x4 overland vehicle on clearing trail
    _add_cast_shadow(canvas, 820, 715, 52, 28)
    v1, va1 = _render_vehicle((44, 52, 162), (185, 192, 198), scale=1.0)
    bbox2 = _draw_rotated_patch(canvas, v1, va1, 820, 715, -24.0)
    targets.append({"class_name": "vehicle", "bbox_xyxy": bbox2})

    # Target 3: SOS ground distress marker panel near vertical tile seam (y=642!)
    _add_cast_shadow(canvas, 1340, 642, 34, 30, dx=5, dy=6, strength=0.25)
    d1, da1 = _render_distress_marker((20, 110, 248), (242, 248, 255), scale=1.0)
    bbox3 = _draw_rotated_patch(canvas, d1, da1, 1340, 642, 12.0)
    targets.append({"class_name": "distress_marker", "bbox_xyxy": bbox3})

    # Target 4: Second hiker in high-vis yellow jacket on upper clearing
    _add_cast_shadow(canvas, 1365, 375, 14, 18)
    p2, a2 = _render_person((28, 215, 240), scale=1.0)
    bbox4 = _draw_rotated_patch(canvas, p2, a2, 1365, 375, -35.0)
    targets.append({"class_name": "person", "bbox_xyxy": bbox4})

    # Gentle atmospheric grain
    noise = rng.normal(0, 3.0, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return canvas, targets


def _build_maritime_scene(rng: np.random.Generator, W: int = 1920, H: int = 1280) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate 1920x1280 offshore maritime swell & whitecap scene with 4 SAR targets."""
    swell = _fbm_noise(H, W, rng, octaves=5)
    ripple = _fbm_noise(H, W, rng, octaves=3)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wave_bands = 0.5 + 0.5 * np.sin(0.032 * xx + 0.055 * yy + 3.2 * swell)

    img = np.zeros((H, W, 3), dtype=np.float32)
    img[:, :, 0] = 92 + 34 * swell + 16 * wave_bands + 10 * ripple   # B
    img[:, :, 1] = 66 + 26 * swell + 14 * wave_bands + 8 * ripple    # G
    img[:, :, 2] = 24 + 14 * swell + 10 * wave_bands                 # R

    # Realistic sea-foam whitecaps & sun glints (clutter)
    foam_mask = np.clip((wave_bands * swell - 0.58) * 3.5, 0.0, 1.0)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1.0 - 0.55 * foam_mask) + 218.0 * (0.55 * foam_mask)

    canvas = np.clip(img, 0, 255).astype(np.uint8)
    targets = []

    # Target 1: 6-person SOLAS orange life raft near center-left
    r1, ra1 = _render_life_raft((22, 108, 248), (32, 134, 252), scale=1.05)
    bbox1 = _draw_rotated_patch(canvas, r1, ra1, 495, 480, -22.0)
    targets.append({"class_name": "life_raft", "bbox_xyxy": bbox1})

    # Target 2: Smaller yellow/orange coastal raft straddling y=640 seam
    r2, ra2 = _render_life_raft((25, 185, 245), (30, 205, 248), scale=0.88)
    bbox2 = _draw_rotated_patch(canvas, r2, ra2, 1090, 640, 31.0)
    targets.append({"class_name": "life_raft", "bbox_xyxy": bbox2})

    # Target 3: Person in high-vis orange PFD floating
    p1, pa1 = _render_person((18, 118, 252), (40, 60, 80), scale=1.0)
    bbox3 = _draw_rotated_patch(canvas, p1, pa1, 790, 860, 48.0)
    targets.append({"class_name": "person", "bbox_xyxy": bbox3})

    # Target 4: Floating distress marker beacon tarp/buoy panel
    d1, da1 = _render_distress_marker((20, 114, 250), (240, 248, 255), scale=0.92)
    bbox4 = _draw_rotated_patch(canvas, d1, da1, 1520, 340, -14.0)
    targets.append({"class_name": "distress_marker", "bbox_xyxy": bbox4})

    noise = rng.normal(0, 2.8, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return canvas, targets


def _build_alpine_scene(rng: np.random.Generator, W: int = 1920, H: int = 1280) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate 1920x1280 alpine snowfield & slate rock outcrop scene with 4 SAR targets."""
    drift = _fbm_noise(H, W, rng, octaves=5)
    ridge = _fbm_noise(H, W, rng, octaves=4)

    img = np.zeros((H, W, 3), dtype=np.float32)
    # Cool bright snowfield with subtle blue crevasse shading
    img[:, :, 0] = 212 + 32 * drift
    img[:, :, 1] = 205 + 34 * drift
    img[:, :, 2] = 198 + 36 * drift

    # Exposed dark slate/granite rock ridges (clutter)
    rock_mask = np.clip((ridge - 0.61) * 5.2, 0.0, 1.0)
    rock_tex = 58 + 42 * drift
    img[:, :, 0] = img[:, :, 0] * (1 - rock_mask) + (rock_tex + 6) * rock_mask
    img[:, :, 1] = img[:, :, 1] * (1 - rock_mask) + rock_tex * rock_mask
    img[:, :, 2] = img[:, :, 2] * (1 - rock_mask) + (rock_tex - 4) * rock_mask

    canvas = np.clip(img, 0, 255).astype(np.uint8)
    targets = []

    # Target 1: Mountaineer in crimson/red jacket with dark pack on bright snow
    _add_cast_shadow(canvas, 560, 410, 16, 22, dx=14, dy=15, strength=0.38)
    p1, pa1 = _render_person((34, 42, 220), (35, 38, 42), scale=1.08)
    bbox1 = _draw_rotated_patch(canvas, p1, pa1, 560, 410, -15.0)
    targets.append({"class_name": "person", "bbox_xyxy": bbox1})

    # Target 2: Second mountaineer in high-vis cobalt/cyan shell near x=1280 seam
    _add_cast_shadow(canvas, 1282, 745, 15, 20, dx=13, dy=14, strength=0.36)
    p2, pa2 = _render_person((215, 145, 24), (30, 35, 45), scale=1.04)
    bbox2 = _draw_rotated_patch(canvas, p2, pa2, 1282, 745, 22.0)
    targets.append({"class_name": "person", "bbox_xyxy": bbox2})

    # Target 3: Rescue-orange emergency bivouac distress panel on snowfield
    _add_cast_shadow(canvas, 930, 895, 34, 30, dx=8, dy=9, strength=0.28)
    d1, da1 = _render_distress_marker((20, 112, 248), (248, 250, 255), scale=1.05)
    bbox3 = _draw_rotated_patch(canvas, d1, da1, 930, 895, 8.0)
    targets.append({"class_name": "distress_marker", "bbox_xyxy": bbox3})

    # Target 4: SAR tracked snowcat vehicle (high-vis red/orange body, dark tracks)
    _add_cast_shadow(canvas, 1545, 515, 54, 30, dx=12, dy=14, strength=0.35)
    v1, va1 = _render_vehicle((28, 86, 228), (228, 232, 238), scale=1.02)
    bbox4 = _draw_rotated_patch(canvas, v1, va1, 1545, 515, 19.0)
    targets.append({"class_name": "vehicle", "bbox_xyxy": bbox4})

    noise = rng.normal(0, 2.5, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return canvas, targets


def generate_dataset(output_dir: Path = SAMPLE_DIR, seed: int = 20261118) -> dict[str, Any]:
    """Generate all 3 synthetic aerial SAR evaluation scenes and ground_truth.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    scenes = [
        {
            "filename": "sar_wilderness_clearing.jpg",
            "builder": _build_wilderness_scene,
            "lat": 47.451280,
            "lon": -121.412640,
            "agl_m": 115.0,
            "heading_deg": 35.0,
            "gimbal_pitch_deg": -90.0,
            "focal_35mm": 28,
            "scene_label": "Cascade Wilderness Clearing (Synthetic)",
        },
        {
            "filename": "sar_maritime_openwater.jpg",
            "builder": _build_maritime_scene,
            "lat": 36.798420,
            "lon": -122.024180,
            "agl_m": 130.0,
            "heading_deg": 115.0,
            "gimbal_pitch_deg": -90.0,
            "focal_35mm": 28,
            "scene_label": "Monterey Bay Offshore Swell (Synthetic)",
        },
        {
            "filename": "sar_alpine_snowfield.jpg",
            "builder": _build_alpine_scene,
            "lat": 46.814950,
            "lon": -121.732110,
            "agl_m": 105.0,
            "heading_deg": 310.0,
            "gimbal_pitch_deg": -90.0,
            "focal_35mm": 28,
            "scene_label": "Mt. Rainier Alpine Snowfield (Synthetic)",
        },
    ]

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_id, spec in enumerate(scenes, start=1):
        canvas, targets = spec["builder"](rng, 1920, 1280)
        out_path = output_dir / spec["filename"]
        _embed_exif_and_save(
            canvas,
            out_path,
            lat=spec["lat"],
            lon=spec["lon"],
            agl_m=spec["agl_m"],
            heading_deg=spec["heading_deg"],
            gimbal_pitch_deg=spec["gimbal_pitch_deg"],
            focal_35mm=spec["focal_35mm"],
            scene_label=spec["scene_label"],
        )

        coco_images.append(
            {
                "id": img_id,
                "file_name": spec["filename"],
                "width": 1920,
                "height": 1280,
                "synthetic": True,
                "exif_gps": {
                    "latitude": spec["lat"],
                    "longitude": spec["lon"],
                    "altitude_agl_m": spec["agl_m"],
                    "heading_deg": spec["heading_deg"],
                    "gimbal_pitch_deg": spec["gimbal_pitch_deg"],
                    "focal_35mm": spec["focal_35mm"],
                },
            }
        )

        for t in targets:
            x1, y1, x2, y2 = t["bbox_xyxy"]
            bw, bh = float(x2 - x1), float(y2 - y1)
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": NAME_TO_CAT_ID[t["class_name"]],
                    "category_name": t["class_name"],
                    "bbox": [float(x1), float(y1), bw, bh],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    gt_payload = {
        "info": {
            "description": "SYNTHETIC SARVision Aerial Search & Rescue Evaluation Dataset (Clearly labeled synthetic test imagery — non-operational).",
            "version": "1.0.0",
            "synthetic": True,
            "date_created": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
        },
        "categories": CATEGORIES,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    gt_path = output_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")
    return gt_payload


if __name__ == "__main__":
    gt = generate_dataset()
    print(
        f"Generated {len(gt['images'])} synthetic SAR evaluation scenes and "
        f"{len(gt['annotations'])} ground-truth annotations -> {SAMPLE_DIR}"
    )
