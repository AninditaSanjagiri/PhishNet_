import { useState } from 'react'
import { analyzeUrl } from '../utils/api.js'
import URLInput from './URLInput.jsx'
import VerdictBanner from './VerdictBanner.jsx'
import AgentCard from './AgentCard.jsx'
import ScreenshotPanel from './ScreenshotPanel.jsx'
import FusionPanel from './FusionPanel.jsx'
import ExplainPanel from './ExplainPanel.jsx'
import LatencyPanel from './LatencyPanel.jsx'
import LoadingState from './LoadingState.jsx'
import './AnalyzerView.css'

export default function AnalyzerView({ onResult, lastResult }) {
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [result,  setResult]  = useState(lastResult)

  async function handleAnalyze(url) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeUrl(url)
      setResult(data)
      onResult(data)
    } catch (err) {
      setError(err.message || 'Analysis failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="analyzer">
      <div className="analyzer-hero">
        <h1 className="hero-title">
          Phishing Detection
          <span className="hero-accent"> Powered by Multimodal AI</span>
        </h1>
        <p className="hero-sub">
          URL analysis · Page content · Visual scan · Explainable results
        </p>
      </div>

      <URLInput onSubmit={handleAnalyze} loading={loading} />

      {error && (
        <div className="error-bar">
          <span className="error-icon">⚠</span>{error}
        </div>
      )}

      {loading && <LoadingState />}

      {result && !loading && (
        <div className="results" style={{ animation: 'fade-up 0.4s ease' }}>

          <VerdictBanner result={result} />

          <div className="results-grid">
            <AgentCard
              title="URL Analysis"
              subtitle="Structure · Features · SHAP"
              icon="🔗"
              agentResult={result.url_agent}
              color="accent"
            />
            <AgentCard
              title="Content Analysis"
              subtitle="Language model · LIME"
              icon="📄"
              agentResult={result.text_agent}
              color="purple"
            />
            <AgentCard
              title="Visual Scan"
              subtitle="Screenshot · Layout"
              icon="🖼"
              agentResult={result.image_agent}
              color="warn"
              isOptional
            />
          </div>

          <ExplainPanel result={result} />

          <div className="results-bottom">
            <FusionPanel  result={result} />
            <LatencyPanel result={result} />
          </div>

          {(result.screenshot_base64 || result.gradcam_base64) && (
            <ScreenshotPanel result={result} />
          )}

        </div>
      )}

      {!result && !loading && !error && (
        <div className="empty-state">
          <div className="empty-grid">
            {['URL structure', 'Page content', 'Visual layout', 'AI fusion'].map((l, i) => (
              <div key={i} className="empty-chip">
                <span className="empty-chip-dot" />{l}
              </div>
            ))}
          </div>
          <p className="empty-hint">Enter any URL above to begin</p>
        </div>
      )}
    </div>
  )
}
