import { useState, useEffect, useCallback } from 'react'
import { getHistory, clearHistory } from '../utils/api.js'
import './HistoryView.css'

function VerdictBadge({ verdict }) {
  const cfg = {
    phishing:   { color: 'danger', label: 'HIGH RISK'   },
    suspicious: { color: 'warn',   label: 'SUSPICIOUS'  },
    safe:       { color: 'safe',   label: 'SAFE'        },
  }[verdict] || { color: 'text3', label: (verdict || '').toUpperCase() }

  return <span className={`verdict-badge mono ${cfg.color}`}>{cfg.label}</span>
}

function ScoreMini({ value, color }) {
  if (value === null || value === undefined)
    return <span className="mini-na mono">—</span>
  return (
    <div className="score-mini">
      <div className="score-mini-track">
        <div className={`score-mini-fill ${color}`}
             style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className={`score-mini-label mono ${color}`}>
        {Math.round(value * 100)}%
      </span>
    </div>
  )
}

function formatTime(iso) {
  const d = new Date(iso)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatDate(iso) {
  const d   = new Date(iso)
  const now = new Date()
  const diffMs  = now - d
  const diffMin = Math.floor(diffMs / 60000)
  const diffH   = Math.floor(diffMin / 60)
  const diffD   = Math.floor(diffH / 24)

  if (diffMin < 1)  return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffH < 24)   return `${diffH}h ago`
  if (diffD === 1)  return 'yesterday'
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export default function HistoryView() {
  const [rows,    setRows]    = useState([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [clearing, setClearing] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getHistory(50)
      setRows(data)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  async function handleClear() {
    if (!window.confirm('Clear all analysis history?')) return
    setClearing(true)
    try {
      await clearHistory()
      setRows([])
    } catch (err) {
      setError(err.message)
    } finally {
      setClearing(false)
    }
  }

  if (loading) return (
    <div className="history-loading">
      <div className="history-spinner" />
      <span className="mono">Loading history…</span>
    </div>
  )

  if (error) return (
    <div className="history-error">
      <span>⚠ {error}</span>
      <button className="retry-btn mono" onClick={load}>retry</button>
    </div>
  )

  if (rows.length === 0) return (
    <div className="history-empty">
      <span className="empty-icon">📋</span>
      <p className="mono">No analyses yet.</p>
      <p className="mono" style={{ color: 'var(--text3)', fontSize: 13 }}>
        Run the Analyzer to see results here.
      </p>
    </div>
  )

  return (
    <div className="history-view">
      <div className="history-header">
        <div className="history-title-row">
          <h2 className="history-title">Analysis History</h2>
          <span className="history-count mono">{rows.length} scans</span>
        </div>
        <button
          className="clear-btn mono"
          onClick={handleClear}
          disabled={clearing}
        >
          {clearing ? 'Clearing…' : '× Clear all'}
        </button>
      </div>

      <div className="history-table-wrap">
        <table className="history-table">
          <thead>
            <tr>
              <th className="mono">When</th>
              <th className="mono">URL</th>
              <th className="mono">Verdict</th>
              <th className="mono">Score</th>
              <th className="mono">URL</th>
              <th className="mono">Text</th>
              <th className="mono">Time</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const verdict    = row.verdict
              const scoreColor = row.phishing_probability >= 50 ? 'danger'
                               : row.phishing_probability >= 20 ? 'warn' : 'safe'
              return (
                <tr key={row.id} className={`history-row ${verdict}`}>
                  <td className="td-time">
                    <span className="mono time-ago">{formatDate(row.created_at)}</span>
                    <span className="mono time-clock">{formatTime(row.created_at)}</span>
                  </td>
                  <td className="td-url">
                    <span className="url-text mono truncate" title={row.url}>
                      {row.url.replace(/^https?:\/\//, '')}
                    </span>
                  </td>
                  <td><VerdictBadge verdict={verdict} /></td>
                  <td>
                    <span className={`mono score-overall ${scoreColor}`}>
                      {row.phishing_probability.toFixed(1)}%
                    </span>
                  </td>
                  <td><ScoreMini value={row.url_score}  color="accent" /></td>
                  <td><ScoreMini value={row.text_score} color="purple" /></td>
                  <td className="mono td-latency">
                    {row.processing_time_ms != null
                      ? `${Math.round(row.processing_time_ms)}ms`
                      : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
