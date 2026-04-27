/**
 * VisualAttackDemo
 * If the analysis has a screenshot, runs all 5 visual perturbation
 * attacks and shows the original + perturbed images with scores.
 * Connects to /robustness/visual endpoint.
 */
import { useState } from 'react'
import { runVisualRobustness } from '../utils/api.js'
import './VisualAttackDemo.css'

const ATTACK_LABELS = {
  gaussian_noise:      'Gaussian Noise',
  jpeg_compression:    'JPEG Compression',
  brightness_shift:    'Brightness Shift',
  pixel_block_mask:    'Pixel Masking',
  fgsm_approximation:  'FGSM Approx.',
}

function ScoreDelta({ baseline, perturbed }) {
  const drop   = Math.max(0, baseline - perturbed)
  const evades = perturbed < 0.5 && baseline >= 0.5
  return (
    <div className={`score-delta ${evades ? 'evades' : 'caught'}`}>
      <span className="sd-badge mono">{evades ? '✗ EVADES' : '✓ CAUGHT'}</span>
      <span className="sd-scores mono">
        <span className="accent">{(baseline * 100).toFixed(0)}%</span>
        &nbsp;→&nbsp;
        <span className={perturbed < 0.5 ? 'danger' : 'safe'}>
          {(perturbed * 100).toFixed(0)}%
        </span>
      </span>
      {drop > 0.02 && (
        <span className="sd-drop mono">−{(drop * 100).toFixed(0)}%</span>
      )}
    </div>
  )
}

export default function VisualAttackDemo({ screenshotB64 }) {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [selected, setSelected] = useState(null)

  if (!screenshotB64) return null

  async function handleRun() {
    setLoading(true); setError(null); setData(null); setSelected(null)
    try {
      const result = await runVisualRobustness(screenshotB64)
      setData(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="visual-attack-demo">
      <div className="vad-header">
        <div className="vad-title-wrap">
          <span className="vad-title">Visual Perturbation Attacks</span>
          <span className="vad-sub mono">5 image-level adversarial strategies on the captured screenshot</span>
        </div>
        <button
          className={`vad-run-btn ${loading ? 'loading' : ''}`}
          onClick={handleRun}
          disabled={loading}
        >
          {loading
            ? <><span className="btn-spin" />Perturbing…</>
            : '▶ Run Visual Attacks'}
        </button>
      </div>

      {error && <div className="vad-error mono">⚠ {error}</div>}

      {data && (
        <div className="vad-results" style={{ animation: 'fade-up 0.3s ease' }}>
          {/* Summary */}
          <div className="vad-summary">
            <div className="vs-item">
              <span className="vs-val mono accent">{(data.baseline_score * 100).toFixed(0)}%</span>
              <span className="vs-label mono">clean score</span>
            </div>
            <div className="vs-item">
              <span className={`vs-val mono ${data.evasion_rate >= 0.5 ? 'danger' : data.evasion_rate >= 0.25 ? 'warn' : 'safe'}`}>
                {(data.evasion_rate * 100).toFixed(0)}%
              </span>
              <span className="vs-label mono">evasion rate</span>
            </div>
            <div className="vs-item">
              <span className="vs-val mono danger">{data.n_evaded}</span>
              <span className="vs-label mono">attacks evaded</span>
            </div>
            <div className="vs-item">
              <span className="vs-val mono safe">{data.n_attacks - data.n_evaded}</span>
              <span className="vs-label mono">blocked</span>
            </div>
          </div>

          {/* Attack score grid */}
          <div className="vad-attack-grid">
            {data.attacks.map((atk, i) => (
              <div
                key={i}
                className={`vad-atk-card ${atk.evades ? 'evades' : ''} ${selected === i ? 'selected' : ''}`}
                onClick={() => setSelected(selected === i ? null : i)}
                role="button"
                tabIndex={0}
              >
                <span className="vad-atk-name mono">
                  {ATTACK_LABELS[atk.attack_name] || atk.attack_name}
                </span>
                <ScoreDelta baseline={data.baseline_score} perturbed={atk.perturbed_score} />
                <span className="vad-desc">{atk.description}</span>
                <span className="vad-pixel mono">Δpx: {atk.mean_pixel_delta?.toFixed(1)}</span>
              </div>
            ))}
          </div>

          <p className="vad-note mono">
            Pixel delta = mean absolute per-channel pixel change.
            FGSM approximation uses finite-difference gradient sign (transfer attack, no model access required).
          </p>
        </div>
      )}

      {!data && !loading && (
        <div className="vad-idle mono">
          Screenshot captured — click <strong>Run Visual Attacks</strong> to apply
          Gaussian noise, JPEG compression, brightness shift, pixel masking, and FGSM approximation.
        </div>
      )}
    </div>
  )
}
