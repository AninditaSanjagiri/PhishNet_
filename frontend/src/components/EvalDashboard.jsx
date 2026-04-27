import { useState, useEffect } from 'react'
import { getEvalSummary } from '../utils/api.js'
import './EvalDashboard.css'

function MetricCell({ value, highlight }) {
  return (
    <td className={`metric-cell mono ${highlight ? 'best' : ''}`}>
      {typeof value === 'number' ? value.toFixed(4) : (value ?? '—')}
    </td>
  )
}

function SystemRow({ data, systems, bestF1 }) {
  const isBest = data.f1 === bestF1
  return (
    <tr className={`system-row ${isBest ? 'best-row' : ''}`}>
      <td className="sys-label mono">{data.system}{isBest ? ' ★' : ''}</td>
      <MetricCell value={data.accuracy}  highlight={data.accuracy  === Math.max(...systems.map(s => s.accuracy))} />
      <MetricCell value={data.precision} highlight={data.precision === Math.max(...systems.map(s => s.precision))} />
      <MetricCell value={data.recall}    highlight={data.recall    === Math.max(...systems.map(s => s.recall))} />
      <MetricCell value={data.f1}        highlight={isBest} />
      <MetricCell value={data.roc_auc}   highlight={data.roc_auc   === Math.max(...systems.map(s => s.roc_auc))} />
      <td className="mono lat-col">
        {data.avg_latency_ms != null ? `${data.avg_latency_ms.toFixed(0)}ms` : '—'}
      </td>
    </tr>
  )
}

function StatCard({ label, value, sub, color }) {
  return (
    <div className="stat-card">
      <span className={`stat-val mono ${color || ''}`}>{value}</span>
      <span className="stat-label">{label}</span>
      {sub && <span className="stat-sub mono">{sub}</span>}
    </div>
  )
}

export default function EvalDashboard() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  useEffect(() => {
    getEvalSummary()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="eval-loading">
      <div className="eval-spinner" />
      <span className="mono">Loading performance data…</span>
    </div>
  )

  if (error || data?.error) return (
    <div className="eval-placeholder">
      <div className="ph-icon">📊</div>
      <h3>System Performance Dashboard</h3>
      <p>No performance reports generated yet.</p>
      <p className="ph-sub">
        Run the backend evaluation once to populate charts and metrics.
      </p>
      <pre className="eval-cmd">python evaluation/run_evaluation.py --fast</pre>
      <p className="ph-note mono">
        Evaluation typically takes 1–2 minutes and tests the system against
        a sample of real URLs.
      </p>
    </div>
  )

  const systems = [data.url_only, data.text_only, data.fused].filter(Boolean)
  const bestF1  = Math.max(...systems.map(s => s.f1))
  const fused   = data.fused || {}

  return (
    <div className="eval-dashboard">
      <div className="eval-header">
        <h2 className="eval-title">System Performance Dashboard</h2>
        <span className="eval-meta mono">
          {data.n_test} URLs tested · {data.n_phishing} phishing · {data.n_legit} legitimate
        </span>
      </div>

      {/* Top stat cards */}
      <div className="stat-grid">
        <StatCard
          label="Detection Accuracy"
          value={fused.accuracy != null ? `${(fused.accuracy * 100).toFixed(1)}%` : '—'}
          sub="fused model"
          color="safe"
        />
        <StatCard
          label="F1 Score"
          value={fused.f1 != null ? fused.f1.toFixed(4) : '—'}
          sub="fused model"
          color="accent"
        />
        <StatCard
          label="ROC-AUC"
          value={fused.roc_auc != null ? fused.roc_auc.toFixed(4) : '—'}
          sub="fused model"
          color="purple"
        />
        <StatCard
          label="Avg Latency"
          value={fused.avg_latency_ms != null ? `${fused.avg_latency_ms.toFixed(0)}ms` : '—'}
          sub="per request"
          color=""
        />
      </div>

      {/* Comparison table */}
      <div className="eval-section">
        <div className="section-header">
          <h3 className="section-title">Component Comparison</h3>
          <span className="section-sub mono">★ = best performing configuration</span>
        </div>
        <div className="table-wrap">
          <table className="eval-table">
            <thead>
              <tr>
                <th className="mono">Configuration</th>
                <th className="mono">Accuracy</th>
                <th className="mono">Precision</th>
                <th className="mono">Recall</th>
                <th className="mono">F1</th>
                <th className="mono">AUC</th>
                <th className="mono">Latency</th>
              </tr>
            </thead>
            <tbody>
              {systems.map(s => (
                <SystemRow key={s.system} data={s} systems={systems} bestF1={bestF1} />
              ))}
            </tbody>
          </table>
        </div>

        {data.fusion_gain_f1 != null && (
          <div className={`fusion-gain ${data.fusion_gain_f1 >= 0 ? 'pos' : 'neg'}`}>
            <span className="mono">
              Multimodal fusion improvement over best single component:&nbsp;
              <strong>
                {data.fusion_gain_f1 > 0 ? '+' : ''}{data.fusion_gain_f1.toFixed(4)} F1
              </strong>
            </span>
          </div>
        )}
      </div>

    </div>
  )
}
