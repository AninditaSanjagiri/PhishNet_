import './Header.css'

const NAV = [
  { id: 'analyzer',   icon: '⬡', label: 'Analyzer'   },
  { id: 'evaluation', icon: '📊', label: 'Performance' },
  { id: 'history',    icon: '≡',  label: 'History'     },
]

export default function Header({ activeView, onNavigate }) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-logo">
          <div className="logo-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <path d="M14 2L3 8v12l11 6 11-6V8L14 2z"
                stroke="var(--accent)" strokeWidth="1.5" fill="none"/>
              <path d="M14 2L3 8l11 6 11-6L14 2z"
                fill="var(--accent)" fillOpacity="0.12"/>
              <circle cx="14" cy="14" r="3" fill="var(--accent)" fillOpacity="0.9"/>
            </svg>
          </div>
          <div className="logo-text">
            <span className="logo-name">PhishNet</span>
            <span className="logo-tag">Multimodal Detection</span>
          </div>
        </div>

        <nav className="header-nav">
          {NAV.map(({ id, icon, label }) => (
            <button
              key={id}
              className={`nav-btn ${activeView === id ? 'active' : ''}`}
              onClick={() => onNavigate(id)}
            >
              <span className="nav-icon">{icon}</span> {label}
            </button>
          ))}
        </nav>

        <div className="header-status">
          <span className="status-dot"/>
          <span className="status-label mono">LIVE</span>
        </div>
      </div>
    </header>
  )
}
