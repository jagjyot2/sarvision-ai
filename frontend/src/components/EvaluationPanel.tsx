import React, { useState } from "react";
import { api, type InferenceParams } from "../api";
import type { EvaluationResponse } from "../types";

interface Props {
  params: InferenceParams;
}

export const EvaluationPanel: React.FC<Props> = ({ params }) => {
  const [iouMatch, setIouMatch] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluationResponse | null>(null);

  const runEval = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.evaluateSampleDataset({ ...params, iou_match_threshold: iouMatch });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="eval-panel">
      <div className="card-header">
        <div>
          <h3>COCO Ground-Truth Benchmark Evaluator</h3>
          <p className="muted-sub">
            Runs live inference across <code>sample_data/</code> against <code>ground_truth.json</code> using active detector settings ({params.detector_mode}, conf ≥ {params.conf_threshold.toFixed(2)}).
          </p>
        </div>
        <div className="eval-controls">
          <label className="inline-label">
            IoU Match: {iouMatch.toFixed(2)}
            <input
              type="range"
              min={0.25}
              max={0.75}
              step={0.05}
              value={iouMatch}
              onChange={(e) => setIouMatch(Number(e.target.value))}
            />
          </label>
          <button className="btn-primary" onClick={runEval} disabled={loading}>
            {loading ? "Evaluating..." : "Run Evaluation Benchmark"}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <>
          <div className="kpi-grid">
            <div className="kpi-card">
              <span>Precision</span>
              <strong>{(result.summary.precision * 100).toFixed(1)}%</strong>
              <small>TP {result.summary.tp} / FP {result.summary.fp}</small>
            </div>
            <div className="kpi-card">
              <span>Recall</span>
              <strong>{(result.summary.recall * 100).toFixed(1)}%</strong>
              <small>TP {result.summary.tp} / GT {result.summary.total_ground_truth}</small>
            </div>
            <div className="kpi-card">
              <span>F1 Score</span>
              <strong>{(result.summary.f1_score * 100).toFixed(1)}%</strong>
              <small>FN {result.summary.fn}</small>
            </div>
            <div className="kpi-card">
              <span>mAP @ {result.iou_threshold.toFixed(2)}</span>
              <strong>{(result.summary.mAP * 100).toFixed(1)}%</strong>
              <small>Mean IoU {(result.summary.mean_iou * 100).toFixed(1)}%</small>
            </div>
          </div>

          <div className="eval-tables">
            <div className="geo-plot-card">
              <div className="card-header"><span>Per-Class Detection Metrics</span></div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>GT</th>
                    <th>TP</th>
                    <th>FP</th>
                    <th>FN</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>AP</th>
                    <th>Mean IoU</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.per_class).map(([cls, m]) => (
                    <tr key={cls}>
                      <td><strong>{cls}</strong></td>
                      <td>{m.gt_count}</td>
                      <td>{m.tp}</td>
                      <td>{m.fp}</td>
                      <td>{m.fn}</td>
                      <td>{(m.precision * 100).toFixed(1)}%</td>
                      <td>{(m.recall * 100).toFixed(1)}%</td>
                      <td>{(m.f1 * 100).toFixed(1)}%</td>
                      <td>{(m.ap * 100).toFixed(1)}%</td>
                      <td>{(m.mean_iou * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="geo-plot-card">
              <div className="card-header"><span>Per-Scene Evaluation Breakdown</span></div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Scene Image</th>
                    <th>GT</th>
                    <th>Pred</th>
                    <th>TP</th>
                    <th>FP</th>
                    <th>FN</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Mean IoU</th>
                  </tr>
                </thead>
                <tbody>
                  {result.per_image.map((im) => (
                    <tr key={im.file_name}>
                      <td className="mono-cell">{im.file_name}</td>
                      <td>{im.gt_count}</td>
                      <td>{im.pred_count}</td>
                      <td>{im.tp}</td>
                      <td>{im.fp}</td>
                      <td>{im.fn}</td>
                      <td>{(im.precision * 100).toFixed(1)}%</td>
                      <td>{(im.recall * 100).toFixed(1)}%</td>
                      <td>{(im.mean_iou * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
