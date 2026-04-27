/**
 * LatencyPanel
 * Displays per-agent inference latency and memory delta.
 * Key metric for the deployment-viability research claim.
 */
import './LatencyPanel.css'

function LatBar({ label, ms, maxMs, color }) {
  const pct = maxMs > 0 ? (ms / maxMs) * 100 : 0
  return (
    <div className="lat-row">
      <span className="lat-label mono">{label}</span>
      <div className="lat-track">
        <div
          className="lat-fill"
          style={{
            width: `${pct}%`,
            background: `var(--${color})`,
            animation: 'bar-fill 0.8s cubic-bezier(0.4,0,0.2,1)',
          }}
        />
      </div>
      <span className="lat-val mono" style={{ color: `var(--${color})` }}>
        {ms.toFixed(1)}ms
      </span>
    </div>
  )
}

export default function LatencyPanel({ result }) {
  const lb = result.latency_breakdown || {}
  const {
    url_ms   = 0,
    text_ms  = 0,
    image_ms = 0,
    total_wall_ms = 0,
  } = lb

  const maxMs = Math.max(url_ms, text_ms, image_ms, 1)
  const mem   = result.memory_delta_mb ?? 0

  return (
    <div className="latency-panel">
      <div className="panel-header">
        <span className="panel-title">Performance Metrics</span>
        <span className="lat-total mono">
          ⏱ {total_wall_ms.toFixed(0)}ms wall-clock
        </span>
      </div>

      <div className="lat-bars">
        <LatBar label="URL agent"   ms={url_ms}   maxMs={maxMs} color="accent" />
        <LatBar label="Text agent"  ms={text_ms}  maxMs={maxMs} color="purple" />
        <LatBar label="Image agent" ms={image_ms} maxMs={maxMs} color="warn"   />
      </div>

      <div className="lat-note-row">
        <span className="lat-chip mono">
          Agents ran in parallel (max, not sum)
        </span>
        {mem !== 0 && (
          <span className="lat-chip mono">
            RSS Δ {mem > 0 ? '+' : ''}{mem.toFixed(1)} MB
          </span>
        )}
      </div>

      <p className="lat-research-note mono">
        Parallel dispatch via asyncio.gather() · Comparable to SAHF-PD's latency
        goal ({'<'}200ms vs LLM systems at 5–15s)
      </p>
    </div>
  )
}
