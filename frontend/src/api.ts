import type {
  AnalysisDetailOut,
  AnalysisSummaryOut,
  DetectorMode,
  DetectionOut,
  EvaluationResponse,
  HealthOut,
  ModelInfo,
  SampleSceneInfo,
  SarClass,
  VerificationStatus,
} from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export interface InferenceParams {
  tile_size: number;
  tile_overlap: number;
  conf_threshold: number;
  nms_iou_threshold: number;
  detector_mode: DetectorMode;
  model_name?: string | null;
}

export const api = {
  health: () => requestJson<HealthOut>("/api/health"),

  models: () => requestJson<{ active_model: string | null; models: ModelInfo[] }>("/api/models"),

  samples: () =>
    requestJson<{
      has_ground_truth: boolean;
      dataset_info: Record<string, unknown>;
      samples: SampleSceneInfo[];
    }>("/api/samples"),

  listAnalyses: () => requestJson<AnalysisSummaryOut[]>("/api/analyses"),

  getAnalysis: (id: string) => requestJson<AnalysisDetailOut>(`/api/analyses/${id}`),

  deleteAnalysis: (id: string) =>
    requestJson<{ deleted: boolean }>(`/api/analyses/${id}`, { method: "DELETE" }),

  analyzeFile: (file: File, params: InferenceParams) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("tile_size", String(params.tile_size));
    fd.append("tile_overlap", String(params.tile_overlap));
    fd.append("conf_threshold", String(params.conf_threshold));
    fd.append("nms_iou_threshold", String(params.nms_iou_threshold));
    fd.append("detector_mode", params.detector_mode);
    if (params.model_name) {
      fd.append("model_name", params.model_name);
    }
    return requestJson<AnalysisDetailOut>("/api/analyze", { method: "POST", body: fd });
  },

  analyzeBatch: (files: File[], params: InferenceParams) => {
    const fd = new FormData();
    for (const f of files) {
      fd.append("files", f);
    }
    fd.append("tile_size", String(params.tile_size));
    fd.append("tile_overlap", String(params.tile_overlap));
    fd.append("conf_threshold", String(params.conf_threshold));
    fd.append("nms_iou_threshold", String(params.nms_iou_threshold));
    fd.append("detector_mode", params.detector_mode);
    if (params.model_name) {
      fd.append("model_name", params.model_name);
    }
    return requestJson<{
      batch_id: string;
      total_files: number;
      total_detections: number;
      analyses: AnalysisDetailOut[];
    }>("/api/analyze/batch", { method: "POST", body: fd });
  },

  analyzeSample: (filename: string, params: InferenceParams) => {
    const qs = new URLSearchParams({
      tile_size: String(params.tile_size),
      tile_overlap: String(params.tile_overlap),
      conf_threshold: String(params.conf_threshold),
      nms_iou_threshold: String(params.nms_iou_threshold),
      detector_mode: params.detector_mode,
    });
    if (params.model_name) {
      qs.set("model_name", params.model_name);
    }
    return requestJson<AnalysisDetailOut>(`/api/analyze/sample/${encodeURIComponent(filename)}?${qs.toString()}`, {
      method: "POST",
    });
  },

  reanalyze: (analysisId: string, params: InferenceParams) =>
    requestJson<AnalysisDetailOut>(`/api/analyses/${analysisId}/reanalyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  verifyDetection: (
    detectionId: number,
    verification_status: VerificationStatus,
    verified_class_name?: SarClass | null,
    operator_notes?: string | null
  ) =>
    requestJson<DetectionOut>(`/api/detections/${detectionId}/verify`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        verification_status,
        verified_class_name: verified_class_name ?? null,
        operator_notes: operator_notes ?? null,
      }),
    }),

  evaluateSampleDataset: (params: InferenceParams & { iou_match_threshold: number }) =>
    requestJson<EvaluationResponse>("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),

  exportUrl: (analysisId: string, format: "coco" | "csv" | "markdown" | "html") =>
    `/api/analyses/${analysisId}/export?format=${format}`,
};
