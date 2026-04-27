/**
 * TextRobustnessDemo
 * Lets the user type or paste phishing text and runs all 6 text
 * adversarial attacks on it live, showing per-attack score changes.
 * Connects to /robustness/text endpoint.
 */
import { useState } from 'react'
import { runTextRobustness } from '../utils/api.js'
import './TextRobustnessDemo.css'

const DEMO_TEXTS = [
  "Your account has been suspended. Click here to verify your identity immediately or it will be closed.",
  "Urgent: Confirm your banking credentials to avoid unauthorized access to your account.",
  "You have won a free prize! Enter your social security number to claim your winner reward.",
  "Dear customer, your login password needs to be updated. Failure to verify will result in suspension.",
]

function AttackChip({ name, baseScore, pertScore, nChanges }) {
  const drop   = Math.max(0, baseScore - pertScore)
  const evades = pertScore < 0.5 && baseScore >= 0.5
  const dropPct = baseScore > 0 ? (drop / baseScore) * 100 : 0

  return (
    <div className={`text-attack-chip ${evades ? 'evades' : 'caught'}`}>
      <div className="tac-top">
        <span className={`tac-badge mono ${evades ? 'danger' : 'safe'}`}>
          {evades ? '✗' : '✓'}
        </span>
        <span className="tac-name mono">{name.replace(/_/g, ' ')}</span>
        {nChanges > 0 && (
          <span className="tac-changes mono">{nChanges} edits</span>
        )}
      </div>
      <div className="tac-bar-row">
        <div className="tac-scores">
          <span className="tac-score mono accent">{(baseScore * 100).toFixed(0)}%</span>
          <span className="tac-arrow mono">→</span>
          <span className={`tac-score mono ${pertScore < 0.5 ? 'danger' : 'safe'}`}>
            {(pertScore * 100).toFixed(0)}%
          </span>
        </div>
        <div className="tac-drop-track">
          <div
            className={`tac-drop-fill ${evades ? 'danger' : 'warn'}`}
            style={{ width: `${Math.min(dropPct, 100)}%` }}
          />
        </div>
      </div>
    </div>
  )
}

export default function TextRobustnessDemo() {
  const [text,    setText]    = useState(DEMO_TEXTS[0])
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  async function handleRun() {
    if (!text.trim()) return
    setLoading(true); setError(null); setData(null)
    try {
      const result = await runTextRobustness(text)
      setData(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="text-robustness-demo">
      <div className="trd-header">
        <span className="trd-title">Text Adversarial Attack Demo</span>
        <span className="trd-sub mono">6 linguistic evasion strategies · live scoring</span>
      </div>

      <div className="trd-input-row">
        <textarea
          className="trd-textarea mono"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste phishing email or webpage text here…"
          rows={3}
          disabled={loading}
        />
        <div className="trd-controls">
          <div className="demo-text-pills">
            {DEMO_TEXTS.map((t, i) => (
              <button
                key={i}
                className="demo-pill mono"
                onClick={() => setText(t)}
                disabled={loading}
              >
                Sample {i + 1}
              </button>
            ))}
          </div>
          <button
            className={`trd-run-btn ${loading ? 'loading' : ''}`}
            onClick={handleRun}
            disabled={loading || !text.trim()}
          >
            {loading
              ? <><span className="btn-spin" />Attacking…</>
              : '▶ Run Text Attacks'}
          </button>
        </div>
      </div>

      {error && <div className="trd-error mono">⚠ {error}</div>}

      {data && (
        <div className="trd-results" style={{ animation: 'fade-up 0.3s ease' }}>
          <div className="trd-summary">
            <span className="trd-sum-item mono">
              Baseline: <strong className="accent">{(data.baseline_score * 100).toFixed(0)}%</strong>
            </span>
            <span className="trd-sum-item mono">
              Evaded: <strong className="danger">{data.n_evaded}/{data.n_attacks}</strong>
            </span>
            <span className="trd-sum-item mono">
              Evasion rate: <strong className={data.evasion_rate >= 0.5 ? 'danger' : data.evasion_rate >= 0.25 ? 'warn' : 'safe'}>
                {(data.evasion_rate * 100).toFixed(0)}%
              </strong>
            </span>
          </div>

          <div className="trd-chips">
            {data.attacks
              .sort((a, b) => b.score_drop - a.score_drop)
              .map((atk, i) => (
                <AttackChip
                  key={i}
                  name={atk.attack_name}
                  baseScore={data.baseline_score}
                  pertScore={atk.perturbed_score}
                  nChanges={atk.n_changes}
                />
              ))}
          </div>

          <p className="trd-note mono">
            Gap addressed: semantic text manipulation attacks are untested in Phan The Duy et al.
            (AWG, Table 19). PhishNet is the first to benchmark DistilBERT against
            these strategies.
          </p>
        </div>
      )}
    </div>
  )
}
