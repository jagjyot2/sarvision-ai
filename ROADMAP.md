# SARVision AI — Engineering Roadmap

## Completed Milestones (v1.0)
- [x] **Offline Neural Detector Pipeline:** Lightweight 4-stage PyTorch CNN (`SARVisionMicroNet-v1`) trained on multi-terrain aerial SAR target patches + hard clutter negatives, exported to self-contained ONNX (`models/sarvision_micro_detector.onnx`) with numerical ORT parity verification.
- [x] **YOLO ONNX Plug-and-Play Support:** Automatic shape-aware adapter supporting drop-in YOLOv8 / YOLOv11 `.onnx` weights alongside built-in ONNX micro-detector.
- [x] **High-Resolution Tiler & Cross-Tile Fusion:** Overlapping `640×640` tiling, internal-seam fragment suppression, and Weighted Box Fusion (WBF).
- [x] **Dual-Stage CV Engine:** OpenCV HSV high-visibility rescue-color + LAB CLAHE saliency fused with ONNX Runtime inference (`fusion`, `onnx`, and `opencv` modes).
- [x] **EXIF GPS & Pinhole Raycasting:** Real EXIF IFD parsing (`0x8825` / `0x8769`) and 3D NED pinhole projection to WGS84 coordinates (supporting nadir and oblique gimbal pitch + heading).
- [x] **Synthetic Evaluation Benchmark:** Procedural 1920×1280 wilderness, maritime, and alpine scenes with embedded GPS EXIF and COCO `ground_truth.json`.
- [x] **FastAPI + SQLite + React/TypeScript UI:** Full REST API, human-in-the-loop verification queue, batch processing, interactive tile/bbox canvas, tactical WGS84 ground plot, Precision/Recall evaluator, and COCO/CSV/Markdown/HTML reporting.
- [x] **Automated Test Suite:** Unit and integration tests covering tiling/NMS, ONNX/OpenCV inference, pinhole geolocation math, COCO metrics, and FastAPI endpoints.

## Future Enhancements (v1.1+)
- [ ] **Thermal / LWIR Radiometric Channel Fusion:** Support 16-bit radiometric TIFF (FLIR/DJI M30T RJPEG) thermal hotspot alignment with RGB orthotiles.
- [ ] **DEM Terrain-Corrected Raycasting:** Replace flat ground-plane intersection with local SRTM/ASTER elevation ray-marching for steep mountain ridges.
- [ ] **ONNX INT8 Quantization & TensorRT/DirectML EP:** Accelerated edge deployment on NVIDIA Jetson Orin / tactical field laptops.
- [ ] **Live RTSP / WebRTC UAV Stream Ingest:** Real-time frame sampling, de-duplication across consecutive patrol frames, and KML/TAK Cursor-on-Target (CoT) export.
