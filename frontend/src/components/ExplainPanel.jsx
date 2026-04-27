/**
 * ExplainPanel
 * Shows SHAP (URL) and LIME (Text) explainability outputs side by side.
 * This is the core research contribution display component.
 */
import './ExplainPanel.css'

function ShapBar({ feature, value, maxVal }) {
  const pct   = maxVal > 0 ? Math.abs(value) / maxVal * 100 : 0
  const pos   = value > 0  // positive = toward phishing
  return (
    <div className="shap-row">
      <span className="shap-feat mono">{feature}</span>
      <div className="shap-bar-wrap">
        <div
          className={`shap-bar ${pos ? 'phish' : 'legit'}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span className={`shap-val mono ${pos ? 'phish' : 'legit'}`}>
        {value > 0 ? '+' : ''}{value.toFixed(4)}
      </span>
    </div>
  )
}

function LimeToken({ token, weight }) {
  const danger = weight > 0
  const opacity = Math.min(Math.abs(weight) * 4 + 0.3, 1)
  return (
    <span
      className={`lime-token ${danger ? 'phish' : 'legit'}`}
      style={{ opacity }}
      title={`LIME weight: ${weight.toFixed(4)}`}
    >
      {token}
      <span className="lime-w mono">{weight > 0 ? '+' : ''}{weight.toFixed(3)}</span>
    </span>
  )
}

export default function ExplainPanel({ result }) {
  const topShap    = result.url_agent?.top_shap || {}
  const limeTokens = result.text_agent?.lime_tokens || []
  const dominant   = result.dominant_modality || 'unknown'
  const contribs   = result.modality_contributions || {}

  const shapEntries = Object.entries(topShap)
  const maxShap     = shapEntries.length
    ? Math.max(...shapEntries.map(([,v]) => Math.abs(v)))
    : 1

  const domColor = {
    url: 'var(--accent)', text: 'var(--purple)',
    image: 'var(--warn)', balanced: 'var(--safe)', none: 'var(--text3)'
  }[dominant] || 'var(--text2)'

  return (
    <div className="explain-panel">
      <div className="explain-header">
        <span className="explain-title">Explainability (XAI)</span>
        <div className="dominant-badge" style={{ borderColor: domColor, color: domColor }}>
          <span className="dom-dot" style={{ background: domColor }} />
          <span className="mono">dominant: {dominant}</span>
        </div>
      </div>

      <div className="explain-grid">

        {/* SHAP - URL Agent */}
        <div className="explain-card">
          <div className="ec-header">
            <span className="ec-label accent">URL Agent — SHAP</span>
            <span className="ec-sub mono">top feature attributions</span>
          </div>
          {shapEntries.length > 0 ? (
            <div className="shap-list">
              {shapEntries.map(([feat, val]) => (
                <ShapBar key={feat} feature={feat} value={val} maxVal={maxShap} />
              ))}
              <p className="explain-note mono">
                Positive values push toward phishing · Negative toward legit
              </p>
            </div>
          ) : (
            <p className="explain-empty mono">SHAP not available (model not loaded)</p>
          )}
        </div>

        {/* LIME - Text Agent */}
        <div className="explain-card">
          <div className="ec-header">
            <span className="ec-label purple">Text Agent — LIME</span>
            <span className="ec-sub mono">token-level importance</span>
          </div>
          {limeTokens.length > 0 ? (
            <div className="lime-list">
              {limeTokens.map((t, i) => (
                <LimeToken key={i} token={t.token} weight={t.weight} />
              ))}
              <p className="explain-note mono">
                Red tokens increase phishing probability · Blue decrease it
              </p>
            </div>
          ) : (
            <p className="explain-empty mono">LIME not available (text unclassified)</p>
          )}
        </div>

      </div>

      {/* Modality contributions */}
      {Object.keys(contribs).length > 0 && (
        <div className="contrib-row">
          <span className="contrib-label mono">Modality contributions:</span>
          {Object.entries(contribs).map(([k, v]) => (
            <span key={k} className={`contrib-chip mono ${k}`}>
              {k}: {typeof v === 'number' ? (v * 100).toFixed(1) + '%' : v}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
