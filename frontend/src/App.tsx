import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Crosshair,
  Download,
  FileText,
  Grid,
  Layers,
  MapPin,
  RefreshCw,
  ShieldAlert,
  Sliders,
  Trash2,
  Upload,
  XCircle,
} from "lucide-react";
import { api, type InferenceParams } from "./api";
import { DetectionCanvas } from "./components/DetectionCanvas";
import { EvaluationPanel } from "./components/EvaluationPanel";
import { GeoTacticalPlot } from "./components/GeoTacticalPlot";
import type {
  AnalysisDetailOut,
  AnalysisSummaryOut,
  DetectorMode,
  HealthOut,
  ModelInfo,
  SampleSceneInfo,
  SarClass,
  VerificationStatus,
} from "./types";

type TabView = "inspector" | "geotag" | "metrics";

const SAR_CLASSES: SarClass[] = ["person", "life_raft", "vehicle", "distress_marker"];

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthOut | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [samples, setSamples] = useState<SampleSceneInfo[]>([]);
  const [history, setHistory] = useState<AnalysisSummaryOut[]>([]);
  const [activeAnalysis, setActiveAnalysis] = useState<AnalysisDetailOut | null>(null);
  const [selectedDetId, setSelectedDetId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<TabView>("inspector");

  const [params, setParams] = useState<InferenceParams>({
    tile_size: 640,
    tile_overlap: 0.2,
    conf_threshold: 0.45,
    nms_iou_threshold: 0.4,
    detector_mode: "fusion",
    model_name: null,
  });

  const [showTiles, setShowTiles] = useState(true);
  const [showFalsePositives, setShowFalsePositives] = useState(true);
  const [previewConfFilter, setPreviewConfFilter] = useState(0.0);
  const [dragOver, setDragOver] = useState(false);
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [noteDraft, setNoteDraft] = useState("");
  const [reclassDraft, setReclassDraft] = useState<SarClass>("person");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshBootstrap = useCallback(async () => {
    try {
      const [h, m, s, hist] = await Promise.all([
        api.health(),
        api.models(),
        api.samples(),
        api.listAnalyses(),
      ]);
      setHealth(h);
      setModels(m.models);
      setSamples(s.samples);
      setHistory(hist);
      if (m.active_model) {
        setParams((prev) => (prev.model_name ? prev : { ...prev, model_name: m.active_model }));
      }
      if (hist.length > 0 && !activeAnalysis) {
        const detail = await api.getAnalysis(hist[0].id);
        setActiveAnalysis(detail);
        setSelectedDetId(detail.detections[0]?.id ?? null);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to connect to SARVision backend.");
    }
  }, [activeAnalysis]);

  useEffect(() => {
    void refreshBootstrap();
  }, [refreshBootstrap]);

  const selectedDetection = useMemo(
    () => activeAnalysis?.detections.find((d) => d.id === selectedDetId) ?? null,
    [activeAnalysis, selectedDetId]
  );

  useEffect(() => {
    if (selectedDetection) {
      setNoteDraft(selectedDetection.operator_notes || "");
      setReclassDraft((selectedDetection.verified_class_name || selectedDetection.class_name) as SarClass);
    }
  }, [selectedDetection]);

  const loadAnalysisDetail = async (id: string) => {
    setErrorMsg(null);
    const detail = await api.getAnalysis(id);
    setActiveAnalysis(detail);
    setSelectedDetId(detail.detections[0]?.id ?? null);
  };

  const handleFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList);
    setErrorMsg(null);
    setBusyMessage(files.length > 1 ? `Batch analyzing ${files.length} aerial frames...` : `Analyzing ${files[0].name}...`);
    try {
      let latest: AnalysisDetailOut;
      if (files.length === 1) {
        latest = await api.analyzeFile(files[0], params);
      } else {
        const batch = await api.analyzeBatch(files, params);
        latest = batch.analyses[0];
      }
      setActiveAnalysis(latest);
      setSelectedDetId(latest.detections[0]?.id ?? null);
      setHistory(await api.listAnalyses());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setBusyMessage(null);
    }
  };

  const handleAnalyzeSample = async (filename: string) => {
    setErrorMsg(null);
    setBusyMessage(`Running tiled inference on ${filename}...`);
    try {
      const detail = await api.analyzeSample(filename, params);
      setActiveAnalysis(detail);
      setSelectedDetId(detail.detections[0]?.id ?? null);
      setHistory(await api.listAnalyses());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Sample analysis failed.");
    } finally {
      setBusyMessage(null);
    }
  };

  const handleReanalyzeActive = async () => {
    if (!activeAnalysis) return;
    setErrorMsg(null);
    setBusyMessage("Re-running inference with updated thresholds...");
    try {
      const detail = await api.reanalyze(activeAnalysis.id, params);
      setActiveAnalysis(detail);
      setSelectedDetId(detail.detections[0]?.id ?? null);
      setHistory(await api.listAnalyses());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Re-analysis failed.");
    } finally {
      setBusyMessage(null);
    }
  };

  const handleVerify = async (status: VerificationStatus, overrideClass?: SarClass) => {
    if (!selectedDetection || !activeAnalysis) return;
    setErrorMsg(null);
    try {
      const updated = await api.verifyDetection(
        selectedDetection.id,
        status,
        overrideClass ?? reclassDraft,
        noteDraft
      );
      setActiveAnalysis({
        ...activeAnalysis,
        detections: activeAnalysis.detections.map((d) => (d.id === updated.id ? updated : d)),
      });
      setHistory(await api.listAnalyses());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Verification update failed.");
    }
  };

  const handleDeleteRun = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await api.deleteAnalysis(id);
    const updated = await api.listAnalyses();
    setHistory(updated);
    if (activeAnalysis?.id === id) {
      if (updated.length > 0) {
        await loadAnalysisDetail(updated[0].id);
      } else {
        setActiveAnalysis(null);
        setSelectedDetId(null);
      }
    }
  };

  return (
    <div className="app-shell">
      {/* Top Navigation & System Telemetry Bar */}
      <header className="topbar">
        <div className="brand-block">
          <Crosshair size={22} className="brand-icon" />
          <div>
            <h1>SARVision AI</h1>
            <span>Aerial Search &amp; Rescue Intelligence · Offline ONNX + OpenCV</span>
          </div>
        </div>

        <nav className="tab-nav">
          <button
            className={activeTab === "inspector" ? "active" : ""}
            onClick={() => setActiveTab("inspector")}
          >
            <Layers size={15} /> Visual Inspector &amp; Queue
          </button>
          <button
            className={activeTab === "geotag" ? "active" : ""}
            onClick={() => setActiveTab("geotag")}
          >
            <MapPin size={15} /> WGS84 Geotag Plot
          </button>
          <button
            className={activeTab === "metrics" ? "active" : ""}
            onClick={() => setActiveTab("metrics")}
          >
            <Activity size={15} /> Precision / Recall Metrics
          </button>
        </nav>

        <div className="health-badges">
          <span className="pill-badge">
            ORT {health?.onnxruntime_version ?? "—"} · CV {health?.opencv_version ?? "—"}
          </span>
          <span className={`pill-badge ${health?.model_loaded ? "pill-ok" : "pill-warn"}`}>
            {health?.active_model ?? "OpenCV-only"}
          </span>
        </div>
      </header>

      <div className="workspace-body">
        {/* Left Control Rail */}
        <aside className="sidebar">
          <section
            className={`dropzone ${dragOver ? "drag-active" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              void handleFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(e) => void handleFiles(e.target.files)}
            />
            <Upload size={22} />
            <strong>Upload Aerial Frame(s)</strong>
            <span>Drag &amp; drop JPEG/TIFF/PNG or click for batch queue</span>
          </section>

          <section className="panel-card">
            <div className="panel-title">
              <Grid size={14} /> Synthetic Sample Scenarios
            </div>
            <div className="sample-list">
              {samples.map((s) => (
                <button
                  key={s.file_name}
                  className="btn-sample"
                  onClick={() => void handleAnalyzeSample(s.file_name)}
                  disabled={Boolean(busyMessage)}
                >
                  <span>{s.file_name}</span>
                  <small>{s.gt_targets ?? "?"} GT targets</small>
                </button>
              ))}
            </div>
          </section>

          <section className="panel-card">
            <div className="panel-title">
              <Sliders size={14} /> Detector &amp; Tiling Controls
            </div>

            <label className="ctrl-row">
              <span>Detector Pipeline</span>
              <select
                value={params.detector_mode}
                onChange={(e) =>
                  setParams({ ...params, detector_mode: e.target.value as DetectorMode })
                }
              >
                <option value="fusion">Dual Fusion (OpenCV + ONNX CNN)</option>
                <option value="onnx">ONNX Neural Only</option>
                <option value="opencv">OpenCV HSV + CLAHE Only</option>
              </select>
            </label>

            {models.length > 0 && (
              <label className="ctrl-row">
                <span>ONNX Weights</span>
                <select
                  value={params.model_name || models[0].file_name}
                  onChange={(e) => setParams({ ...params, model_name: e.target.value })}
                >
                  {models.map((m) => (
                    <option key={m.file_name} value={m.file_name}>
                      {m.file_name} ({m.size_kb} KB)
                    </option>
                  ))}
                </select>
              </label>
            )}

            <label className="ctrl-row">
              <span>Confidence Threshold: <strong>{params.conf_threshold.toFixed(2)}</strong></span>
              <input
                type="range"
                min={0.2}
                max={0.9}
                step={0.05}
                value={params.conf_threshold}
                onChange={(e) => setParams({ ...params, conf_threshold: Number(e.target.value) })}
              />
            </label>

            <label className="ctrl-row">
              <span>Global NMS IoU: <strong>{params.nms_iou_threshold.toFixed(2)}</strong></span>
              <input
                type="range"
                min={0.15}
                max={0.8}
                step={0.05}
                value={params.nms_iou_threshold}
                onChange={(e) => setParams({ ...params, nms_iou_threshold: Number(e.target.value) })}
              />
            </label>

            <label className="ctrl-row">
              <span>Tile Overlap: <strong>{Math.round(params.tile_overlap * 100)}%</strong></span>
              <input
                type="range"
                min={0.0}
                max={0.4}
                step={0.05}
                value={params.tile_overlap}
                onChange={(e) => setParams({ ...params, tile_overlap: Number(e.target.value) })}
              />
            </label>

            <button
              className="btn-primary full-w"
              onClick={() => void handleReanalyzeActive()}
              disabled={!activeAnalysis || Boolean(busyMessage)}
            >
              <RefreshCw size={14} /> Apply &amp; Re-Analyze Frame
            </button>
          </section>

          <section className="panel-card history-card">
            <div className="panel-title">Analysis History ({history.length})</div>
            <div className="history-list">
              {history.map((item) => (
                <div
                  key={item.id}
                  className={`history-item ${activeAnalysis?.id === item.id ? "active" : ""}`}
                  onClick={() => void loadAnalysisDetail(item.id)}
                >
                  <div>
                    <strong>{item.filename}</strong>
                    <small>
                      {item.detection_count} dets · {item.confirmed_count}✓ · {item.detector_mode}
                    </small>
                  </div>
                  <button
                    className="btn-icon-del"
                    title="Delete analysis"
                    onClick={(e) => void handleDeleteRun(item.id, e)}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
              {history.length === 0 && (
                <div className="empty-hint">No runs stored yet. Pick a sample scene or upload imagery.</div>
              )}
            </div>
          </section>
        </aside>

        {/* Main Viewport */}
        <main className="main-content">
          {busyMessage && <div className="info-banner">{busyMessage}</div>}
          {errorMsg && <div className="error-banner">{errorMsg}</div>}

          {activeTab === "metrics" ? (
            <EvaluationPanel params={params} />
          ) : !activeAnalysis ? (
            <div className="empty-stage">
              <Crosshair size={40} />
              <h2>Ready for Aerial SAR Target Detection</h2>
              <p>
                Drop a high-resolution JPEG/TIFF aerial image on the left or click one of the 3
                geotagged synthetic evaluation scenes to run tiled ONNX + OpenCV inference.
              </p>
            </div>
          ) : activeTab === "geotag" ? (
            <GeoTacticalPlot
              analysis={activeAnalysis}
              selectedDetectionId={selectedDetId}
              onSelectDetection={setSelectedDetId}
            />
          ) : (
            <div className="inspector-layout">
              {/* Top Action & Export Strip */}
              <div className="scene-toolbar">
                <div className="scene-meta">
                  <strong>{activeAnalysis.filename}</strong>
                  <span>
                    {activeAnalysis.image_width}×{activeAnalysis.image_height} px ·{" "}
                    {activeAnalysis.tile_count} tiles · {activeAnalysis.processing_time_ms.toFixed(0)} ms
                    {activeAnalysis.gsd_cm_px ? ` · GSD ${activeAnalysis.gsd_cm_px} cm/px` : ""}
                  </span>
                  {activeAnalysis.synthetic_flag && (
                    <span className="synth-tag">SYNTHETIC EVALUATION DATA</span>
                  )}
                </div>

                <div className="toolbar-controls">
                  <label className="chk-label">
                    <input
                      type="checkbox"
                      checked={showTiles}
                      onChange={(e) => setShowTiles(e.target.checked)}
                    />
                    Tile Grid
                  </label>
                  <label className="chk-label">
                    <input
                      type="checkbox"
                      checked={showFalsePositives}
                      onChange={(e) => setShowFalsePositives(e.target.checked)}
                    />
                    Show Rejected
                  </label>
                  <label className="inline-label">
                    Preview Filter ≥ {(previewConfFilter * 100).toFixed(0)}%
                    <input
                      type="range"
                      min={0}
                      max={0.9}
                      step={0.05}
                      value={previewConfFilter}
                      onChange={(e) => setPreviewConfFilter(Number(e.target.value))}
                    />
                  </label>

                  <div className="export-group">
                    <a href={api.exportUrl(activeAnalysis.id, "coco")} className="btn-export">
                      <Download size={13} /> COCO
                    </a>
                    <a href={api.exportUrl(activeAnalysis.id, "csv")} className="btn-export">
                      <Download size={13} /> CSV
                    </a>
                    <a href={api.exportUrl(activeAnalysis.id, "markdown")} className="btn-export">
                      <FileText size={13} /> SAR Report (.md)
                    </a>
                    <a
                      href={api.exportUrl(activeAnalysis.id, "html")}
                      target="_blank"
                      rel="noreferrer"
                      className="btn-export"
                    >
                      <FileText size={13} /> HTML
                    </a>
                  </div>
                </div>
              </div>

              <div className="inspector-split">
                <DetectionCanvas
                  analysis={activeAnalysis}
                  selectedDetectionId={selectedDetId}
                  onSelectDetection={setSelectedDetId}
                  showTiles={showTiles}
                  showFalsePositives={showFalsePositives}
                  previewConfFilter={previewConfFilter}
                />

                {/* Right Human-in-the-Loop Verification Queue */}
                <div className="verification-panel">
                  <div className="card-header">
                    <span>Verification Queue ({activeAnalysis.detections.length})</span>
                  </div>

                  <div className="det-queue-list">
                    {activeAnalysis.detections.map((d) => (
                      <div
                        key={d.id}
                        className={`det-item ${d.id === selectedDetId ? "selected" : ""}`}
                        onClick={() => setSelectedDetId(d.id)}
                      >
                        <div className="det-item-top">
                          <strong>
                            #{d.id} {d.verified_class_name || d.class_name}
                          </strong>
                          <span className={`status-pill status-${d.verification_status}`}>
                            {d.verification_status}
                          </span>
                        </div>
                        <div className="det-item-sub">
                          <span>Conf {(d.confidence * 100).toFixed(1)}%</span>
                          <span>ONNX {(d.onnx_score * 100).toFixed(0)}% / CV {(d.opencv_score * 100).toFixed(0)}%</span>
                        </div>
                        {d.wgs84_lat !== null && (
                          <div className="det-item-geo">
                            {d.wgs84_lat.toFixed(6)}°, {d.wgs84_lon?.toFixed(6)}°
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  {selectedDetection && (
                    <div className="operator-box">
                      <h4>Operator Triage — Detection #{selectedDetection.id}</h4>
                      <div className="triage-meta">
                        <div>Source: <code>{selectedDetection.detector_source}</code> ({selectedDetection.color_signature})</div>
                        <div>BBox: <code>[{Math.round(selectedDetection.x1)}, {Math.round(selectedDetection.y1)}, {Math.round(selectedDetection.x2)}, {Math.round(selectedDetection.y2)}]</code></div>
                        <div>
                          WGS84: <code>{selectedDetection.wgs84_lat !== null ? `${selectedDetection.wgs84_lat.toFixed(6)}, ${selectedDetection.wgs84_lon?.toFixed(6)}` : "No EXIF GPS"}</code>
                        </div>
                      </div>

                      <label className="ctrl-row">
                        <span>Reclassify Target</span>
                        <select
                          value={reclassDraft}
                          onChange={(e) => setReclassDraft(e.target.value as SarClass)}
                        >
                          {SAR_CLASSES.map((c) => (
                            <option key={c} value={c}>{c}</option>
                          ))}
                        </select>
                      </label>

                      <label className="ctrl-row">
                        <span>Operator Notes</span>
                        <textarea
                          rows={2}
                          placeholder="Visual confirmation notes, rescue sector info..."
                          value={noteDraft}
                          onChange={(e) => setNoteDraft(e.target.value)}
                        />
                      </label>

                      <div className="triage-actions">
                        <button
                          className="btn-confirm"
                          onClick={() => void handleVerify("confirmed")}
                        >
                          <CheckCircle2 size={14} /> Confirm
                        </button>
                        <button
                          className="btn-escalate"
                          onClick={() => void handleVerify("escalated")}
                        >
                          <AlertTriangle size={14} /> Escalate
                        </button>
                        <button
                          className="btn-reject"
                          onClick={() => void handleVerify("false_positive")}
                        >
                          <XCircle size={14} /> False Pos
                        </button>
                      </div>
                      <button
                        className="btn-secondary full-w"
                        onClick={() => void handleVerify(selectedDetection.verification_status, reclassDraft)}
                      >
                        <ShieldAlert size={13} /> Save Reclassification &amp; Notes
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
