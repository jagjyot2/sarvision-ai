import React from "react";
import type { AnalysisDetailOut, DetectionOut } from "../types";

const CLASS_COLORS: Record<string, string> = {
  person: "#fb923c",
  life_raft: "#38bdf8",
  vehicle: "#a78bfa",
  distress_marker: "#f43f5e",
};

interface Props {
  analysis: AnalysisDetailOut;
  selectedDetectionId: number | null;
  onSelectDetection: (id: number) => void;
  showTiles: boolean;
  showFalsePositives: boolean;
  previewConfFilter: number;
}

export const DetectionCanvas: React.FC<Props> = ({
  analysis,
  selectedDetectionId,
  onSelectDetection,
  showTiles,
  showFalsePositives,
  previewConfFilter,
}) => {
  const visibleDetections = analysis.detections.filter((d) => {
    if (d.confidence < previewConfFilter) return false;
    if (!showFalsePositives && d.verification_status === "false_positive") return false;
    return true;
  });

  return (
    <div className="canvas-frame">
      <div className="canvas-viewport">
        <img
          src={analysis.image_url}
          alt={analysis.filename}
          className="canvas-base-img"
        />
        <svg
          className="canvas-svg-overlay"
          viewBox={`0 0 ${analysis.image_width} ${analysis.image_height}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {showTiles &&
            analysis.tiles.map((tile) => (
              <g key={tile.tile_id}>
                <rect
                  x={tile.x_offset}
                  y={tile.y_offset}
                  width={tile.width}
                  height={tile.height}
                  fill="rgba(56, 189, 248, 0.03)"
                  stroke="rgba(56, 189, 248, 0.45)"
                  strokeWidth={2}
                  strokeDasharray="10 6"
                />
                <text
                  x={tile.x_offset + 10}
                  y={tile.y_offset + 24}
                  fill="rgba(125, 211, 252, 0.85)"
                  fontSize="18"
                  fontFamily="monospace"
                  fontWeight="600"
                >
                  T{tile.tile_id} ({tile.row},{tile.col})
                </text>
              </g>
            ))}

          {visibleDetections.map((det: DetectionOut) => {
            const displayClass = det.verified_class_name || det.class_name;
            const baseColor = CLASS_COLORS[displayClass] || "#22d3ee";
            const isSelected = det.id === selectedDetectionId;
            const isFP = det.verification_status === "false_positive";
            const strokeColor =
              det.verification_status === "confirmed"
                ? "#10b981"
                : det.verification_status === "escalated"
                ? "#f59e0b"
                : isFP
                ? "#64748b"
                : baseColor;

            const w = Math.max(4, det.x2 - det.x1);
            const h = Math.max(4, det.y2 - det.y1);
            const statusMark =
              det.verification_status === "confirmed"
                ? " ✓"
                : det.verification_status === "escalated"
                ? " !"
                : isFP
                ? " ✕"
                : "";
            const label = `#${det.id} ${displayClass} ${(det.confidence * 100).toFixed(0)}%${statusMark}`;
            const labelY = det.y1 > 28 ? det.y1 - 6 : det.y2 + 22;

            return (
              <g
                key={det.id}
                onClick={() => onSelectDetection(det.id)}
                style={{ cursor: "pointer", opacity: isFP ? 0.52 : 1 }}
              >
                {isSelected && (
                  <rect
                    x={det.x1 - 5}
                    y={det.y1 - 5}
                    width={w + 10}
                    height={h + 10}
                    fill="none"
                    stroke="#ffffff"
                    strokeWidth={3}
                    strokeDasharray="6 4"
                  />
                )}
                <rect
                  x={det.x1}
                  y={det.y1}
                  width={w}
                  height={h}
                  fill={isSelected ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.06)"}
                  stroke={strokeColor}
                  strokeWidth={isSelected ? 4 : 3}
                />
                <rect
                  x={det.x1}
                  y={labelY - 19}
                  width={Math.max(120, label.length * 10.2)}
                  height={24}
                  rx={4}
                  fill="rgba(9, 14, 24, 0.88)"
                  stroke={strokeColor}
                  strokeWidth={1.5}
                />
                <text
                  x={det.x1 + 6}
                  y={labelY - 2}
                  fill={strokeColor}
                  fontSize="16"
                  fontFamily="ui-monospace, Consolas, monospace"
                  fontWeight="700"
                >
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="canvas-legend">
        <span>
          <strong>{visibleDetections.length}</strong> / {analysis.detections.length} detections shown
        </span>
        <span className="legend-dots">
          <i style={{ background: "#fb923c" }} /> person
          <i style={{ background: "#38bdf8" }} /> life_raft
          <i style={{ background: "#a78bfa" }} /> vehicle
          <i style={{ background: "#f43f5e" }} /> distress_marker
          <i style={{ background: "#10b981" }} /> confirmed
          <i style={{ background: "#f59e0b" }} /> escalated
        </span>
      </div>
    </div>
  );
};
