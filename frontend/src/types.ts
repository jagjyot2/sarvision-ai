export type DetectorMode = "fusion" | "onnx" | "opencv";
export type VerificationStatus = "pending" | "confirmed" | "false_positive" | "escalated";
export type SarClass = "person" | "life_raft" | "vehicle" | "distress_marker";

export interface TileRect {
  tile_id: number;
  row: number;
  col: number;
  x_offset: number;
  y_offset: number;
  width: number;
  height: number;
}

export interface DetectionOut {
  id: number;
  analysis_id: string;
  class_id: number;
  class_name: string;
  confidence: number;
  onnx_score: number;
  opencv_score: number;
  detector_source: string;
  color_signature: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  tile_id: number | null;
  wgs84_lat: number | null;
  wgs84_lon: number | null;
  north_offset_m: number | null;
  east_offset_m: number | null;
  slant_range_m: number | null;
  verification_status: VerificationStatus;
  verified_class_name: string | null;
  operator_notes: string | null;
  updated_at: string;
}

export interface AnalysisDetailOut {
  id: string;
  filename: string;
  image_width: number;
  image_height: number;
  tile_size: number;
  tile_overlap: number;
  tile_count: number;
  tiles: TileRect[];
  conf_threshold: number;
  nms_iou_threshold: number;
  detector_mode: DetectorMode;
  model_name: string;
  processing_time_ms: number;
  raw_tile_detections_count: number;
  has_gps_exif: boolean;
  uav_lat: number | null;
  uav_lon: number | null;
  uav_alt_m: number;
  gimbal_pitch_deg: number;
  heading_deg: number;
  hfov_deg: number;
  gsd_cm_px: number | null;
  swath_width_m: number | null;
  swath_height_m: number | null;
  synthetic_flag: boolean;
  batch_id: string | null;
  created_at: string;
  image_url: string;
  detections: DetectionOut[];
}

export interface AnalysisSummaryOut {
  id: string;
  filename: string;
  image_width: number;
  image_height: number;
  detector_mode: string;
  model_name: string;
  processing_time_ms: number;
  has_gps_exif: boolean;
  uav_lat: number | null;
  uav_lon: number | null;
  detection_count: number;
  confirmed_count: number;
  escalated_count: number;
  false_positive_count: number;
  pending_count: number;
  batch_id: string | null;
  created_at: string;
}

export interface HealthOut {
  status: string;
  app_name: string;
  opencv_version: string;
  onnxruntime_version: string;
  model_loaded: boolean;
  active_model: string | null;
  model_architecture: string;
  classes: string[];
}

export interface SampleSceneInfo {
  file_name: string;
  width?: number;
  height?: number;
  synthetic: boolean;
  gt_targets?: number;
  exif_gps?: {
    latitude: number;
    longitude: number;
    altitude_agl_m: number;
    heading_deg: number;
  };
}

export interface ModelInfo {
  file_name: string;
  size_kb: number;
  architecture: string;
  classes: string[];
}

export interface EvaluationClassMetrics {
  gt_count: number;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
  ap: number;
  mean_iou: number;
}

export interface EvaluationResponse {
  iou_threshold: number;
  summary: {
    images_evaluated: number;
    total_ground_truth: number;
    total_predictions: number;
    tp: number;
    fp: number;
    fn: number;
    precision: number;
    recall: number;
    f1_score: number;
    mAP: number;
    mean_iou: number;
    class_agnostic_precision: number;
    class_agnostic_recall: number;
  };
  per_class: Record<string, EvaluationClassMetrics>;
  per_image: Array<{
    file_name: string;
    gt_count: number;
    pred_count: number;
    tp: number;
    fp: number;
    fn: number;
    precision: number;
    recall: number;
    f1: number;
    mean_iou: number;
  }>;
}
