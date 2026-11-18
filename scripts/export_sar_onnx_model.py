"""Train and export a lightweight PyTorch aerial SAR target classifier to ONNX.

Trains SARVisionMicroNet on 10,000 procedurally synthesized 64x64 RGB patches
covering 5 classes (background + hard terrain clutter + off-center fragments,
person, life_raft, vehicle, distress_marker), validates generalization accuracy,
exports a self-contained ONNX model via torch.onnx.export, checks ONNX graph
validity, and verifies numerical parity with ONNX Runtime.

Outputs:
  - models/sarvision_micro_detector.onnx
  - models/model_metadata.json

Run directly:
    python scripts/export_sar_onnx_model.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ONNX_PATH = MODELS_DIR / "sarvision_micro_detector.onnx"
META_PATH = MODELS_DIR / "model_metadata.json"

CLASSES = ["background", "person", "life_raft", "vehicle", "distress_marker"]
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class SARVisionMicroNet(nn.Module):
    """Lightweight 4-stage CNN for 64x64 aerial SAR patch classification."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _random_terrain_patch(rng: np.random.Generator, size: int = 64) -> tuple[np.ndarray, str]:
    """Synthesize a diverse 64x64 RGB terrain patch (forest, dirt, ocean, snow, asphalt)."""
    biome = str(rng.choice(["forest", "dirt", "water", "snow", "asphalt"], p=[0.28, 0.20, 0.24, 0.22, 0.06]))
    low = rng.random((8, 8), dtype=np.float32)
    tex = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((20, 20), dtype=np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    n = 0.65 * tex + 0.35 * fine

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    if biome == "forest":
        base_g = float(rng.uniform(52, 102))
        rgb[:, :, 0] = base_g * 0.45 + 28 * n
        rgb[:, :, 1] = base_g + 42 * n
        rgb[:, :, 2] = base_g * 0.40 + 24 * n
        for _ in range(int(rng.integers(2, 6))):
            cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
            r = int(rng.integers(10, 22))
            cv2.circle(rgb, (cx + 5, cy + 6), r, (18, 28, 18), -1)
            tg = int(rng.integers(60, 108))
            cv2.circle(rgb, (cx, cy), r, (24, tg, 24), -1)
            cv2.circle(rgb, (cx - r // 4, cy - r // 4), max(3, r // 2), (32, min(130, tg + 18), 32), -1)
    elif biome == "dirt":
        rgb[:, :, 0] = 105 + 45 * n + float(rng.uniform(-12, 18))
        rgb[:, :, 1] = 85 + 38 * n + float(rng.uniform(-10, 14))
        rgb[:, :, 2] = 60 + 30 * n
    elif biome == "water":
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        freq = float(rng.uniform(0.10, 0.26))
        ang = float(rng.uniform(0, math.pi))
        waves = 0.5 + 0.5 * np.sin(freq * (xx * math.cos(ang) + yy * math.sin(ang)) + 3.0 * n)
        rgb[:, :, 0] = 18 + 22 * n + 14 * waves
        rgb[:, :, 1] = 58 + 30 * n + 18 * waves
        rgb[:, :, 2] = 88 + 42 * n + 22 * waves
    elif biome == "snow":
        base_v = float(rng.uniform(198, 232))
        rgb[:, :, 0] = base_v - 16 + 26 * n
        rgb[:, :, 1] = base_v - 8 + 24 * n
        rgb[:, :, 2] = base_v + 18 * n
    else:
        g = float(rng.uniform(75, 130))
        rgb[:, :, 0] = g + 24 * n
        rgb[:, :, 1] = g + 24 * n
        rgb[:, :, 2] = g + 26 * n

    return np.clip(rgb, 0, 255).astype(np.uint8), biome


def _add_hard_background_clutter(patch: np.ndarray, biome: str, rng: np.random.Generator) -> np.ndarray:
    """Inject realistic terrain distractors (autumn speckles, tree crowns, whitecaps, granite rocks)."""
    out = patch.copy()
    if biome in ("forest", "dirt") and rng.random() < 0.70:
        for _ in range(int(rng.integers(1, 5))):
            cx, cy = int(rng.integers(5, 59)), int(rng.integers(5, 59))
            r = int(rng.integers(2, 6))
            col = (int(rng.integers(110, 165)), int(rng.integers(68, 108)), int(rng.integers(22, 46)))
            cv2.circle(out, (cx, cy), r, col, -1)
    elif biome == "water" and rng.random() < 0.75:
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = int(rng.integers(8, 56)), int(rng.integers(8, 56))
            axes = (int(rng.integers(5, 15)), int(rng.integers(2, 6)))
            ang = float(rng.uniform(0, 180))
            cv2.ellipse(out, (cx, cy), axes, ang, 0, 360, (218, 226, 232), -1)
    elif biome == "snow" and rng.random() < 0.75:
        for _ in range(int(rng.integers(2, 7))):
            cx, cy = int(rng.integers(5, 59)), int(rng.integers(5, 59))
            axes = (int(rng.integers(4, 15)), int(rng.integers(3, 11)))
            ang = float(rng.uniform(0, 180))
            shade = int(rng.integers(45, 100))
            cv2.ellipse(out, (cx, cy), axes, ang, 0, 360, (shade, shade + 3, shade + 8), -1)
    return out


def _draw_rotated(
    canvas: np.ndarray,
    fg: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    angle_deg: float,
) -> np.ndarray:
    """Alpha-composite rotated RGB foreground onto 64x64 patch."""
    h, w = fg.shape[:2]
    pad = max(h, w) + 12
    p_fg = np.zeros((pad, pad, 3), dtype=np.uint8)
    p_m = np.zeros((pad, pad), dtype=np.float32)
    y0, x0 = (pad - h) // 2, (pad - w) // 2
    p_fg[y0 : y0 + h, x0 : x0 + w] = fg
    p_m[y0 : y0 + h, x0 : x0 + w] = mask

    M = cv2.getRotationMatrix2D((pad / 2.0, pad / 2.0), angle_deg, 1.0)
    r_fg = cv2.warpAffine(p_fg, M, (pad, pad), flags=cv2.INTER_LINEAR)
    r_m = cv2.warpAffine(p_m, M, (pad, pad), flags=cv2.INTER_LINEAR)

    H, W = canvas.shape[:2]
    xs, ys = cx - pad // 2, cy - pad // 2
    cx1, cy1 = max(0, xs), max(0, ys)
    cx2, cy2 = min(W, xs + pad), min(H, ys + pad)
    px1, py1 = cx1 - xs, cy1 - ys
    px2, py2 = px1 + (cx2 - cx1), py1 + (cy2 - cy1)
    if cx2 <= cx1 or cy2 <= cy1:
        return canvas

    alpha = r_m[py1:py2, px1:px2, None]
    out = canvas.copy().astype(np.float32)
    out[cy1:cy2, cx1:cx2] = out[cy1:cy2, cx1:cx2] * (1.0 - alpha) + r_fg[py1:py2, px1:px2].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_fg_object(cls_id: int, scale: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Render an isolated RGB target + alpha mask for classes 1..4."""
    if cls_id == 1:  # person
        palette = [
            (245, 105, 20),  # orange
            (240, 210, 25),  # yellow
            (218, 42, 36),   # crimson
            (28, 138, 222),  # cobalt/cyan
            (70, 215, 70),   # lime
            (45, 52, 65),    # dark
        ]
        j_col = palette[int(rng.integers(0, len(palette)))]
        w, h = max(16, int(round(26 * scale))), max(18, int(round(32 * scale)))
        fg = np.zeros((h, w, 3), dtype=np.uint8)
        m = np.zeros((h, w), dtype=np.float32)
        px, py = w // 2, h // 2
        cv2.ellipse(fg, (px, py + int(4 * scale)), (int(6 * scale), int(9 * scale)), 0, 0, 360, (55, 58, 66), -1)
        cv2.ellipse(m, (px, py + int(4 * scale)), (int(6 * scale), int(9 * scale)), 0, 0, 360, 0.95, -1)
        cv2.ellipse(fg, (px, py - int(1 * scale)), (int(9 * scale), int(6 * scale)), 0, 0, 360, j_col, -1)
        cv2.ellipse(m, (px, py - int(1 * scale)), (int(9 * scale), int(6 * scale)), 0, 0, 360, 1.0, -1)
        cv2.rectangle(fg, (px - int(4 * scale), py + 1), (px + int(4 * scale), py + int(8 * scale)), (42, 44, 48), -1)
        cv2.rectangle(m, (px - int(4 * scale), py + 1), (px + int(4 * scale), py + int(8 * scale)), 1.0, -1)
        cv2.circle(fg, (px, py - int(4 * scale)), max(2, int(4 * scale)), j_col, -1)
        cv2.circle(m, (px, py - int(4 * scale)), max(2, int(4 * scale)), 1.0, -1)
        return fg, m

    if cls_id == 2:  # life_raft
        tube_pal = [(248, 108, 22), (242, 195, 28), (230, 62, 32)]
        t_col = tube_pal[int(rng.integers(0, len(tube_pal)))]
        w, h = max(26, int(round(44 * scale))), max(22, int(round(34 * scale)))
        fg = np.zeros((h, w, 3), dtype=np.uint8)
        m = np.zeros((h, w), dtype=np.float32)
        px, py = w // 2, h // 2
        cv2.ellipse(fg, (px, py), (max(8, int(18 * scale)), max(6, int(14 * scale))), 0, 0, 360, t_col, -1)
        cv2.ellipse(m, (px, py), (max(8, int(18 * scale)), max(6, int(14 * scale))), 0, 0, 360, 1.0, -1)
        cv2.ellipse(fg, (px, py), (max(6, int(15 * scale)), max(5, int(12 * scale))), 0, 0, 360, (35, 35, 35), 1)
        cv2.ellipse(fg, (px + int(5 * scale), py), (max(3, int(5 * scale)), max(3, int(6 * scale))), 0, 0, 360, (28, 32, 38), -1)
        cv2.line(fg, (px - int(11 * scale), py), (px - int(3 * scale), py), (245, 245, 245), 2)
        return fg, m

    if cls_id == 3:  # vehicle
        body_pal = [
            (165, 48, 42),
            (228, 86, 28),   # SAR orange/red snowcat
            (45, 82, 138),
            (205, 210, 215),
            (78, 92, 65),
            (65, 68, 72),
        ]
        b_col = body_pal[int(rng.integers(0, len(body_pal)))]
        roof_col = (225, 230, 236) if rng.random() < 0.5 else (180, 186, 192)
        w, h = max(28, int(round(46 * scale))), max(16, int(round(26 * scale)))
        fg = np.zeros((h, w, 3), dtype=np.uint8)
        m = np.zeros((h, w), dtype=np.float32)
        # Wheels/tracks
        for wx in (int(w * 0.22), int(w * 0.78)):
            for wy in (int(h * 0.14), int(h * 0.86)):
                cv2.rectangle(fg, (wx - 3, wy - 2), (wx + 3, wy + 2), (24, 24, 26), -1)
                cv2.rectangle(m, (wx - 3, wy - 2), (wx + 3, wy + 2), 1.0, -1)
        cv2.rectangle(fg, (int(3 * scale), int(4 * scale)), (w - int(3 * scale), h - int(4 * scale)), b_col, -1)
        cv2.rectangle(m, (int(3 * scale), int(4 * scale)), (w - int(3 * scale), h - int(4 * scale)), 1.0, -1)
        wx1, wx2 = int(w * 0.54), int(w * 0.69)
        cv2.rectangle(fg, (wx1, int(5 * scale)), (wx2, h - int(5 * scale)), (35, 44, 58), -1)
        cv2.rectangle(fg, (int(w * 0.18), int(5 * scale)), (wx1, h - int(5 * scale)), roof_col, -1)
        cv2.rectangle(fg, (int(w * 0.21), int(7 * scale)), (wx1 - 2, h - int(7 * scale)), (45, 48, 52), 1)
        return fg, m

    # distress_marker
    p_pal = [(250, 108, 18), (248, 205, 22), (226, 38, 46), (235, 65, 155)]
    p_col = p_pal[int(rng.integers(0, len(p_pal)))]
    w, h = max(22, int(round(36 * scale))), max(20, int(round(32 * scale)))
    fg = np.zeros((h, w, 3), dtype=np.uint8)
    m = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(fg, (2, 2), (w - 2, h - 2), p_col, -1)
    cv2.rectangle(m, (2, 2), (w - 2, h - 2), 1.0, -1)
    t = max(2, int(round(3 * scale)))
    cv2.line(fg, (6, 6), (w - 6, h - 6), (248, 250, 252), t)
    cv2.line(fg, (w - 6, 6), (6, h - 6), (248, 250, 252), t)
    cv2.rectangle(fg, (2, 2), (w - 2, h - 2), (35, 35, 35), 1)
    return fg, m


def _synthesize_target_patch(cls_id: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize a single 64x64 RGB patch for class cls_id (0..4)."""
    canvas, biome = _random_terrain_patch(rng, 64)
    if rng.random() < 0.55:
        canvas = _add_hard_background_clutter(canvas, biome, rng)

    if cls_id == 0:
        # Half of background samples include terrain clutter; 25% include off-center partial edges
        canvas = _add_hard_background_clutter(canvas, biome, rng)
        if rng.random() < 0.25:
            fg_cls = int(rng.integers(1, 5))
            fg, m = _render_fg_object(fg_cls, float(rng.uniform(0.75, 1.1)), rng)
            side = int(rng.integers(0, 4))
            ox, oy = [(2, 32), (62, 32), (32, 2), (32, 62)][side]
            canvas = _draw_rotated(canvas, fg, m, ox, oy, float(rng.uniform(0, 360)))
    else:
        cx = 32 + int(rng.integers(-5, 6))
        cy = 32 + int(rng.integers(-5, 6))
        angle = float(rng.uniform(0, 360))
        scale = float(rng.uniform(0.72, 1.18))

        if biome != "water" and rng.random() < 0.78:
            sh = np.zeros((64, 64), dtype=np.float32)
            rad = int(round((8 if cls_id == 1 else 16) * scale))
            cv2.circle(sh, (min(60, cx + 6), min(60, cy + 7)), max(4, rad), 1.0, -1)
            sh = cv2.GaussianBlur(sh, (9, 9), 2.5)
            canvas = np.clip(canvas.astype(np.float32) * (1.0 - 0.35 * sh[:, :, None]), 0, 255).astype(np.uint8)

        fg, m = _render_fg_object(cls_id, scale, rng)
        canvas = _draw_rotated(canvas, fg, m, cx, cy, angle)

    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        canvas = cv2.GaussianBlur(canvas, (k, k), float(rng.uniform(0.4, 1.0)))
    alpha_gain = float(rng.uniform(0.84, 1.16))
    beta_bias = float(rng.uniform(-16, 16))
    noise = rng.normal(0, float(rng.uniform(1.5, 5.0)), canvas.shape)
    return np.clip(canvas.astype(np.float32) * alpha_gain + beta_bias + noise, 0, 255).astype(np.uint8)


def _normalize_patches(images_uint8: np.ndarray) -> np.ndarray:
    x = images_uint8.astype(np.float32) / 255.0
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(NORM_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    return np.transpose(x, (0, 3, 1, 2)).astype(np.float32)


def train_and_export(
    samples_per_class: int = 1800,
    epochs: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    """Train SARVisionMicroNet on synthetic aerial patches and export ONNX + metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    imgs = []
    labels = []
    # Give background extra samples to cover diverse terrain + edge distractors
    counts = {0: int(samples_per_class * 1.4), 1: samples_per_class, 2: samples_per_class, 3: samples_per_class, 4: samples_per_class}
    for cls_id, cnt in counts.items():
        for _ in range(cnt):
            imgs.append(_synthesize_target_patch(cls_id, rng))
            labels.append(cls_id)

    imgs_np = np.stack(imgs, axis=0)
    labels_np = np.array(labels, dtype=np.int64)

    perm = rng.permutation(len(labels_np))
    imgs_np = imgs_np[perm]
    labels_np = labels_np[perm]

    split = int(0.8 * len(labels_np))
    x_train = torch.from_numpy(_normalize_patches(imgs_np[:split]))
    y_train = torch.from_numpy(labels_np[:split])
    x_val = torch.from_numpy(_normalize_patches(imgs_np[split:]))
    y_val = torch.from_numpy(labels_np[split:])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False)

    model = SARVisionMicroNet(num_classes=len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)

    best_acc = -1.0
    best_state = None
    final_train_loss = 0.0

    for _ in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(by)
        scheduler.step()
        final_train_loss = running_loss / len(x_train)

        model.eval()
        correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                preds = model(bx).argmax(dim=1)
                correct += int((preds == by).sum().item())
        val_acc = correct / len(x_val)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for bx, _ in val_loader:
            all_preds.append(model(bx).argmax(dim=1).numpy())
    val_preds = np.concatenate(all_preds, axis=0)
    val_true = y_val.numpy()

    per_class_val: dict[str, dict[str, float]] = {}
    for cid, cname in enumerate(CLASSES):
        tp = int(np.sum((val_preds == cid) & (val_true == cid)))
        fp = int(np.sum((val_preds == cid) & (val_true != cid)))
        fn = int(np.sum((val_preds != cid) & (val_true == cid)))
        per_class_val[cname] = {
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "support": int(np.sum(val_true == cid)),
        }

    dummy = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    sample_batch = x_val[:16]
    with torch.no_grad():
        pt_out = model(sample_batch).numpy()
    ort_out = ort_sess.run(["logits"], {"input": sample_batch.numpy()})[0]
    max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
    if max_abs_diff > 1e-4:
        raise RuntimeError(f"ONNX Runtime parity mismatch: max_abs_diff={max_abs_diff}")

    param_count = sum(p.numel() for p in model.parameters())
    meta = {
        "model_name": "SARVisionMicroNet-v1",
        "architecture": "sarvision_micro_cnn",
        "onnx_file": ONNX_PATH.name,
        "opset_version": 17,
        "parameter_count": int(param_count),
        "model_size_kb": round(ONNX_PATH.stat().st_size / 1024.0, 2),
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [1, 3, 64, 64],
        "color_format": "RGB",
        "normalization": {"mean": NORM_MEAN, "std": NORM_STD},
        "classes": CLASSES,
        "foreground_classes": {str(i): c for i, c in enumerate(CLASSES) if i > 0},
        "training": {
            "synthetic_dataset": True,
            "samples_total": int(len(labels_np)),
            "train_samples": int(len(x_train)),
            "val_samples": int(len(x_val)),
            "epochs": epochs,
            "final_train_loss": round(final_train_loss, 4),
            "best_val_accuracy": round(best_acc, 4),
            "per_class_validation": per_class_val,
        },
        "onnx_parity_max_abs_diff": max_abs_diff,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    info = train_and_export()
    print(
        f"Exported {info['onnx_file']} ({info['model_size_kb']} KB, {info['parameter_count']} params) | "
        f"Val Acc={info['training']['best_val_accuracy']:.4f} | "
        f"ORT parity max_diff={info['onnx_parity_max_abs_diff']:.2e}",
        flush=True,
    )
