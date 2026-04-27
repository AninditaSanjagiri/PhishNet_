import { useState } from 'react'
import Header from './components/Header.jsx'
import AnalyzerView from './components/AnalyzerView.jsx'
import HistoryView from './components/HistoryView.jsx'
import EvalDashboard from './components/EvalDashboard.jsx'
import './App.css'

export default function App() {
  const [view,       setView]       = useState('analyzer')
  const [lastResult, setLastResult] = useState(null)

  return (
    <div className="app-shell">
      <Header activeView={view} onNavigate={setView} />
      <main className="app-main">
        {view === 'analyzer'   && <AnalyzerView onResult={setLastResult} lastResult={lastResult} />}
        {view === 'evaluation' && <EvalDashboard />}
        {view === 'history'    && <HistoryView />}
      </main>
      <footer className="app-footer">
        <span className="mono" style={{ color: 'var(--text3)', fontSize: 12 }}>
          PhishNet — Multimodal Phishing Detection System
        </span>
      </footer>
    </div>
  )
}
