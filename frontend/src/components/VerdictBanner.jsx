import './VerdictBanner.css'

const VERDICT_CONFIG = {
  phishing: {
    label: 'HIGH RISK',
    sub:   'Likely phishing — do not proceed',
    color: 'danger',
    icon:  '⚠',
    glow:  'var(--glow-danger)',
    borderColor: 'rgba(255,56,96,0.4)',
    bg:    'rgba(255,56,96,0.06)',
  },
  suspicious: {
    label: 'SUSPICIOUS',
    sub:   'Proceed with caution',
    color: 'warn',
    icon:  '⬡',
    glow:  'var(--glow-warn)',
    borderColor: 'rgba(255,176,32,0.35)',
    bg:    'rgba(255,176,32,0.05)',
  },
  safe: {
    label: 'SAFE',
    sub:   'No phishing indicators detected',
    color: 'safe',
    icon:  '✓',
    glow:  'var(--glow-safe)',
    borderColor: 'rgba(0,230,118,0.3)',
    bg:    'rgba(0,230,118,0.04)',
  },
}

function ProbabilityRing({ value, color }) {
  const r      = 42
  const circ   = 2 * Math.PI * r
  const filled = (value / 100) * circ

  return (
    <svg className="prob-ring" width="110" height="110" viewBox="0 0 110 110">
      <circle cx="55" cy="55" r={r} fill="none" stroke="var(--border)" strokeWidth="5" />
      <circle
        cx="55" cy="55" r={r} fill="none"
        stroke={`var(--${color})`} strokeWidth="5"
        strokeDasharray={`${filled} ${circ}`}
        strokeLinecap="round"
        transform="rotate(-90 55 55)"
        style={{ transition: 'stroke-dasharray 1s cubic-bezier(0.4,0,0.2,1)' }}
      />
      <text x="55" y="51" textAnchor="middle" fill={`var(--${color})`}
        style={{ fontFamily: 'var(--font-head)', fontWeight: 800, fontSize: 22 }}>
        {Math.round(value)}%
      </text>
      <text x="55" y="65" textAnchor="middle" fill="var(--text3)"
        style={{ fontFamily: 'var(--font-mono)', fontSize: 9, letterSpacing: '0.08em' }}>
        RISK SCORE
      </text>
    </svg>
  )
}

export default function VerdictBanner({ result }) {
  const cfg = VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.suspicious
  const c   = cfg.color
  const lb  = result.latency_breakdown || {}
  const ms  = lb.total_wall_ms ?? result.processing_time_ms

  return (
    <div
      className="verdict-banner"
      style={{
        border:     `1px solid ${cfg.borderColor}`,
        background: cfg.bg,
        boxShadow:  cfg.glow,
      }}
    >
      <div className="verdict-left">
        <div className={`verdict-icon-wrap ${c}`}>
          <span className="verdict-icon">{cfg.icon}</span>
        </div>
        <div className="verdict-text">
          <span className={`verdict-label mono ${c}`}>{cfg.label}</span>
          <span className="verdict-sub">{cfg.sub}</span>
          <span className="verdict-url mono truncate">{result.url}</span>
          <div className="verdict-meta">
            {ms != null && (
              <span className="meta-chip mono">⏱ {Math.round(ms)}ms</span>
            )}
            {result.dominant_modality && result.dominant_modality !== 'unknown' && (
              <span className="meta-chip mono">
                dominant: {result.dominant_modality}
              </span>
            )}
          </div>
        </div>
      </div>

      <ProbabilityRing value={result.phishing_probability} color={c} />
    </div>
  )
}
