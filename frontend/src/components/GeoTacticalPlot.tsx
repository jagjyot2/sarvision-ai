import React from "react";
import type { AnalysisDetailOut } from "../types";

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
}

export const GeoTacticalPlot: React.FC<Props> = ({
  analysis,
  selectedDetectionId,
  onSelectDetection,
}) => {
  const geoDets = analysis.detections.filter(
    (d) => d.north_offset_m !== null && d.east_offset_m !== null && d.wgs84_lat !== null && d.wgs84_lon !== null
  );

  const halfW = (analysis.swath_width_m || 140) / 2;
  const halfH = (analysis.swath_height_m || 100) / 2;
  const maxExt = Math.max(
    60,
    halfW * 1.15,
    halfH * 1.15,
    ...geoDets.map((d) => Math.max(Math.abs(d.east_offset_m || 0), Math.abs(d.north_offset_m || 0)) * 1.25)
  );

  const size = 460;
  const center = size / 2;
  const scale = (size * 0.42) / maxExt;
  const ringStep = maxExt > 160 ? 50 : 25;
  const rings: number[] = [];
  for (let r = ringStep; r <= maxExt * 1.05; r += ringStep) {
    rings.push(r);
  }

  return (
    <div className="geo-grid-layout">
      <div className="geo-plot-card">
        <div className="card-header">
          <span>Tactical WGS84 Ground Projection (Local NED Swath)</span>
          <span className="mono-badge">
            {analysis.has_gps_exif
              ? `UAV ${analysis.uav_lat?.toFixed(5)}°, ${analysis.uav_lon?.toFixed(5)}°`
              : "NO GPS EXIF IN IMAGE"}
          </span>
        </div>

        <svg className="geo-svg" viewBox={`0 0 ${size} ${size}`}>
          <rect x={0} y={0} width={size} height={size} fill="#09111e" rx={8} />
          {/* Axis lines */}
          <line x1={center} y1={16} x2={center} y2={size - 16} stroke="#1e293b" strokeWidth={1} />
          <line x1={16} y1={center} x2={size - 16} y2={center} stroke="#1e293b" strokeWidth={1} />

          {/* Concentric range rings */}
          {rings.map((r) => (
            <g key={r}>
              <circle
                cx={center}
                cy={center}
                r={r * scale}
                fill="none"
                stroke="#1e293b"
                strokeDasharray="4 4"
              />
              <text x={center + 4} y={center - r * scale + 12} fill="#64748b" fontSize="10" fontFamily="monospace">
                {r}m
              </text>
            </g>
          ))}

          {/* Camera ground swath rotated by heading */}
          {analysis.swath_width_m && analysis.swath_height_m && (
            <rect
              x={center - (analysis.swath_width_m / 2) * scale}
              y={center - (analysis.swath_height_m / 2) * scale}
              width={analysis.swath_width_m * scale}
              height={analysis.swath_height_m * scale}
              fill="rgba(56, 189, 248, 0.06)"
              stroke="rgba(56, 189, 248, 0.45)"
              strokeWidth={1.5}
              transform={`rotate(${analysis.heading_deg} ${center} ${center})`}
            />
          )}

          {/* Heading vector */}
          <g transform={`rotate(${analysis.heading_deg} ${center} ${center})`}>
            <line x1={center} y1={center} x2={center} y2={center - 38} stroke="#38bdf8" strokeWidth={2} />
            <polygon points={`${center},${center - 44} ${center - 5},${center - 34} ${center + 5},${center - 34}`} fill="#38bdf8" />
          </g>

          {/* UAV Nadir Crosshair */}
          <circle cx={center} cy={center} r={5} fill="#38bdf8" />
          <text x={center + 8} y={center + 14} fill="#94a3b8" fontSize="10" fontFamily="monospace">
            UAV Nadir ({analysis.uav_alt_m}m AGL)
          </text>

          {/* Plotted Detections */}
          {geoDets.map((d) => {
            const x = center + (d.east_offset_m || 0) * scale;
            const y = center - (d.north_offset_m || 0) * scale;
            const cls = d.verified_class_name || d.class_name;
            const color = d.verification_status === "false_positive" ? "#64748b" : CLASS_COLORS[cls] || "#22d3ee";
            const isSel = d.id === selectedDetectionId;
            return (
              <g key={d.id} onClick={() => onSelectDetection(d.id)} style={{ cursor: "pointer" }}>
                {isSel && <circle cx={x} cy={y} r={12} fill="none" stroke="#ffffff" strokeWidth={2} />}
                <circle cx={x} cy={y} r={6} fill={color} stroke="#09111e" strokeWidth={1.5} />
                <text x={x + 9} y={y + 4} fill={color} fontSize="11" fontFamily="monospace" fontWeight="700">
                  #{d.id} {cls}
                </text>
              </g>
            );
          })}

          {/* Compass labels */}
          <text x={center - 4} y={14} fill="#94a3b8" fontSize="11" fontWeight="700">N</text>
          <text x={center - 4} y={size - 6} fill="#94a3b8" fontSize="11" fontWeight="700">S</text>
          <text x={size - 16} y={center + 4} fill="#94a3b8" fontSize="11" fontWeight="700">E</text>
          <text x={6} y={center + 4} fill="#94a3b8" fontSize="11" fontWeight="700">W</text>
        </svg>
      </div>

      <div className="geo-table-card">
        <div className="card-header">
          <span>EXIF Telemetry &amp; WGS84 Target Coordinates</span>
        </div>
        <div className="telemetry-kv-grid">
          <div><span>UAV GPS:</span><strong>{analysis.uav_lat?.toFixed(6) ?? "—"}, {analysis.uav_lon?.toFixed(6) ?? "—"}</strong></div>
          <div><span>Altitude (AGL):</span><strong>{analysis.uav_alt_m} m</strong></div>
          <div><span>Gimbal Pitch:</span><strong>{analysis.gimbal_pitch_deg}°</strong></div>
          <div><span>Heading:</span><strong>{analysis.heading_deg}° True</strong></div>
          <div><span>HFOV / GSD:</span><strong>{analysis.hfov_deg}° / {analysis.gsd_cm_px ?? "—"} cm/px</strong></div>
          <div><span>Ground Swath:</span><strong>{analysis.swath_width_m ?? "—"} × {analysis.swath_height_m ?? "—"} m</strong></div>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Class</th>
                <th>WGS84 Lat / Lon</th>
                <th>Offset (N, E)</th>
                <th>Slant</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {analysis.detections.map((d) => (
                <tr
                  key={d.id}
                  className={d.id === selectedDetectionId ? "selected-row" : ""}
                  onClick={() => onSelectDetection(d.id)}
                >
                  <td>#{d.id}</td>
                  <td>{d.verified_class_name || d.class_name}</td>
                  <td className="mono-cell">
                    {d.wgs84_lat !== null ? `${d.wgs84_lat.toFixed(6)}, ${d.wgs84_lon?.toFixed(6)}` : "No EXIF GPS"}
                  </td>
                  <td className="mono-cell">
                    {d.north_offset_m !== null ? `${d.north_offset_m >= 0 ? "+" : ""}${d.north_offset_m}m, ${d.east_offset_m! >= 0 ? "+" : ""}${d.east_offset_m}m` : "—"}
                  </td>
                  <td className="mono-cell">{d.slant_range_m ? `${d.slant_range_m} m` : "—"}</td>
                  <td><span className={`status-pill status-${d.verification_status}`}>{d.verification_status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
