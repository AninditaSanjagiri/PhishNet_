import { useEffect, useState } from 'react'
import './LoadingState.css'

const STEPS = [
  { icon: '🔗', label: 'Scanning URL structure…'       },
  { icon: '📄', label: 'Analyzing page content…'        },
  { icon: '🖼', label: 'Running visual scan…'           },
  { icon: '🔀', label: 'Generating verdict…'            },
]

export default function LoadingState() {
  const [active, setActive] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setActive(a => Math.min(a + 1, STEPS.length - 1)), 1400)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="loading-state">
      <div className="loading-header">
        <div className="loading-scanner">
          <div className="scanner-ring" />
          <div className="scanner-dot" />
        </div>
        <span className="loading-title mono">{STEPS[active].label}</span>
      </div>

      <div className="loading-steps">
        {STEPS.map((step, i) => (
          <div
            key={i}
            className={`loading-step ${i <= active ? 'active' : ''} ${i < active ? 'done' : ''}`}
          >
            <span className="step-icon">{i < active ? '✓' : step.icon}</span>
            <span className="step-bar">
              <span className="step-fill" style={{ animationDelay: `${i * 0.1}s` }} />
            </span>
            <span className="step-label mono">{step.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
