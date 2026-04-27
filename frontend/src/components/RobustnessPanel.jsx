/**
 * RobustnessPanel
 * Live demonstration of all URL adversarial attacks on the analyzed URL.
 * Shows per-attack score, evasion status, and overall evasion rate.
 * Connects to /robustness/url endpoint.
 */
import { useState } from 'react'
import { runUrlRobustness } from '../utils/api.js'
import './RobustnessPanel.css'

function AttackRow({ atk, baseline }) {
  const evades = atk.evades
  const drop   = atk.score_drop
  const ps     = atk.perturbed_score
  const dropPct = baseline > 0 ? (drop / baseline) * 100 : 0

  return (
    <div className={`attack-row ${evades ? 'evades' : ''}`}>
      <div className="atk-name-wrap">
        <span className={`atk-badge ${evades ? 'danger' : 'safe'}`}>
          {evades ? '✗ EVADES' : '✓ CAUGHT'}
        </span>
        <span className="atk-name mono">{atk.attack_name}</span>
      </div>

      <div className="atk-scores">
        <div className="score-pair">
          <span className="score-label mono">before</span>
          <span className="score-val mono accent">{(baseline * 100).toFixed(0)}%</span>
        </div>
        <span className="score-arrow mono">→</span>
        <div className="score-pair">
          <span className="score-label mono">after</span>
          <span className={`score-val mono ${ps < 0.5 ? 'danger' : 'safe'}`}>
            {(ps * 100).toFixed(0)}%
          </span>
        </div>
        <div className="drop-bar-wrap" title={`Score drop: ${(drop*100).toFixed(1)}%`}>
          <div className="drop-bar" style={{ width: `${Math.min(dropPct, 100)}%` }} />
        </div>
        <span className="drop-label mono">−{(drop * 100).toFixed(0)}%</span>
      </div>

      {atk.perturbed_url && atk.perturbed_url !== atk.original_url && (
        <div className="atk-url mono truncate">{atk.perturbed_url}</div>
      )}
      {atk.description && (
        <div className="atk-desc">{atk.description}</div>
      )}
    </div>
  )
}

export default function RobustnessPanel({ url, screenshotB64 }) {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  async function handleRun() {
    setLoading(true); setError(null); setData(null)
    try {
      const result = await runUrlRobustness(url)
      setData(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const evasionColor = data
    ? data.evasion_rate >= 0.5 ? 'danger'
    : data.evasion_rate >= 0.25 ? 'warn' : 'safe'
    : 'text3'

  return (
    <div className="robustness-panel">
      <div className="rp-header">
        <div className="rp-title-wrap">
          <span className="rp-title">Adversarial Robustness</span>
          <span className="rp-sub mono">7 URL mutation attacks · live on this URL</span>
        </div>
        <button
          className={`rp-run-btn ${loading ? 'loading' : ''}`}
          onClick={handleRun}
          disabled={loading || !url}
        >
          {loading
            ? <><span className="btn-spin"/>Running attacks…</>
            : '▶ Run Attack Suite'}
        </button>
      </div>

      {error && (
        <div className="rp-error mono">⚠ {error}</div>
      )}

      {data && (
        <div className="rp-results" style={{ animation: 'fade-up 0.3s ease' }}>
          {/* Summary row */}
          <div className="rp-summary">
            <div className="rp-stat">
              <span className="rp-stat-val mono" style={{ color:`var(--${evasionColor})` }}>
                {(data.evasion_rate * 100).toFixed(0)}%
              </span>
              <span className="rp-stat-label mono">evasion rate</span>
            </div>
            <div className="rp-stat">
              <span className="rp-stat-val mono danger">{data.n_evaded}</span>
              <span className="rp-stat-label mono">attacks evaded</span>
            </div>
            <div className="rp-stat">
              <span className="rp-stat-val mono safe">{data.n_attacks - data.n_evaded}</span>
              <span className="rp-stat-label mono">attacks blocked</span>
            </div>
            <div className="rp-stat">
              <span className="rp-stat-val mono accent">
                {(data.baseline_score * 100).toFixed(0)}%
              </span>
              <span className="rp-stat-label mono">clean score</span>
            </div>
          </div>

          {/* Attack rows */}
          <div className="rp-attacks">
            {data.attacks
              .sort((a, b) => b.score_drop - a.score_drop)
              .map((atk, i) => (
                <AttackRow key={i} atk={atk} baseline={data.baseline_score} />
              ))}
          </div>

          {/* Research note */}
          <div className="rp-research-note mono">
            Evasion rate = fraction of attacks that drop score below detection threshold (0.5).
            Fused multimodal architecture provides complementary signal when URL agent is evaded.
          </div>
        </div>
      )}

      {!data && !loading && (
        <div className="rp-idle">
          <p className="mono">
            Click <strong>Run Attack Suite</strong> to test how the URL agent responds
            to 7 adversarial URL mutations on this specific URL.
          </p>
        </div>
      )}
    </div>
  )
}
