/**
 * RobustnessDashboard
 * Displays full robustness benchmark results from /evaluation/robustness.
 * Shows evasion tables for all three agents + fusion resilience comparison.
 */
import { useState, useEffect } from 'react'
import { getRobustnessSummary } from '../utils/api.js'
import './RobustnessDashboard.css'

function EvasionTable({ title, agentColor, data, attackKey }) {
  if (!data) return null
  const perAttack = data.per_attack_evasion || {}
  const perDrop   = data.per_attack_drop    || {}
  const rows = Object.entries(perAttack)
    .map(([name, evasion]) => ({ name, evasion, drop: perDrop[name] || 0 }))
    .sort((a, b) => b.evasion - a.evasion)

  return (
    <div className="evasion-card">
      <div className="ecard-header">
        <span className="ecard-title" style={{ color: `var(--${agentColor})` }}>
          {title}
        </span>
        <div className="ecard-stats">
          <span className="estat mono">
            Mean evasion: <strong style={{ color:`var(--${agentColor})` }}>
              {(data.mean_evasion_rate * 100).toFixed(1)}%
            </strong>
          </span>
          <span className="estat mono">
            Clean acc: <strong style={{ color:'var(--safe)' }}>
              {(data.clean_detection_acc * 100).toFixed(1)}%
            </strong>
          </span>
        </div>
      </div>

      <div className="ecard-worst mono">
        ⚠ Worst attack: <span className="worst-name">{data.worst_attack}</span>
        &nbsp;({(data.worst_evasion_rate * 100).toFixed(1)}% evasion)
      </div>

      <table className="evasion-table">
        <thead>
          <tr>
            <th className="mono">Attack</th>
            <th className="mono">Evasion</th>
            <th className="mono">Score Drop</th>
            <th className="mono">Bar</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.name}
                className={row.evasion >= 0.5 ? 'high-risk' : row.evasion >= 0.25 ? 'med-risk' : ''}>
              <td className="mono atk-td">{row.name}</td>
              <td className={`mono pct-td ${row.evasion >= 0.5 ? 'danger' : row.evasion >= 0.25 ? 'warn' : 'safe'}`}>
                {(row.evasion * 100).toFixed(1)}%
              </td>
              <td className="mono pct-td" style={{ color:'var(--text3)' }}>
                −{(row.drop * 100).toFixed(1)}%
              </td>
              <td className="bar-td">
                <div className="inline-bar-track">
                  <div className={`inline-bar ${row.evasion >= 0.5 ? 'danger' : row.evasion >= 0.25 ? 'warn' : 'safe'}`}
                       style={{ width: `${row.evasion * 100}%` }}/>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function FusionResilienceCard({ fusion, url, text, image }) {
  if (!fusion) return null
  const agents = [
    { name:'URL',   evasion: url?.mean_evasion_rate   || 0, color:'accent' },
    { name:'Text',  evasion: text?.mean_evasion_rate  || 0, color:'purple' },
    { name:'Image', evasion: image?.mean_evasion_rate || 0, color:'warn'   },
    { name:'Fused', evasion: fusion.combined_attack_evasion_rate || 0, color:'safe' },
  ]
  const maxE = Math.max(...agents.map(a => a.evasion), 0.01)

  return (
    <div className="fusion-resilience-card">
      <div className="fr-header">
        <span className="fr-title">Fusion Resilience vs Single Agents</span>
        <span className="fr-gain mono">
          Robustness gain: &nbsp;
          <strong style={{ color: fusion.robustness_gain_vs_worst_agent >= 0
              ? 'var(--safe)' : 'var(--danger)' }}>
            {fusion.robustness_gain_vs_worst_agent >= 0 ? '+' : ''}
            {(fusion.robustness_gain_vs_worst_agent * 100).toFixed(1)}%
          </strong>
        </span>
      </div>

      <div className="fr-bars">
        {agents.map(a => (
          <div key={a.name} className="fr-row">
            <span className={`fr-label mono ${a.color}`}>{a.name}</span>
            <div className="fr-track">
              <div
                className={`fr-fill ${a.name === 'Fused' ? 'safe' : a.color}`}
                style={{ width: `${(a.evasion / maxE) * 100}%` }}
              />
            </div>
            <span className={`fr-pct mono ${a.evasion >= 0.5 ? 'danger' : a.evasion >= 0.25 ? 'warn' : 'safe'}`}>
              {(a.evasion * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      <div className="fr-insight">
        <span className="fi-label mono">Research claim:</span>
        <p>
          Multimodal fusion reduces evasion rate versus the worst single-modality
          agent by {(fusion.robustness_gain_vs_worst_agent * 100).toFixed(1)} percentage
          points — consistent with Phan The Duy et al. (2024) finding that multimodal
          models are the most adversarially robust architecture class.
        </p>
      </div>
    </div>
  )
}

export default function RobustnessDashboard() {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  useEffect(() => {
    getRobustnessSummary()
      .then(setData).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="rb-loading">
      <div className="rb-spinner"/><span className="mono">Loading robustness results…</span>
    </div>
  )

  if (error || data?.error) return (
    <div className="rb-placeholder">
      <div className="ph-icon">🛡</div>
      <h3>No Robustness Data Yet</h3>
      <p>Run the robustness benchmark to generate results:</p>
      <pre className="rb-cmd">cd backend{"\n"}python robustness/run_robustness.py --fast</pre>
      <div className="attack-preview">
        <span className="ap-label mono">Attacks included:</span>
        <div className="ap-chips">
          {['URL homograph', 'Subdomain inject', 'TLD swap', 'Path noise',
            'HTTPS spoof', 'Keyword dilution', 'Entropy reduction',
            'Synonym swap', 'Urgency paraphrase', 'Whitespace inject',
            'Leet substitution', 'Sentence dilution',
            'Gaussian noise', 'JPEG compression', 'Brightness shift',
            'Pixel masking', 'FGSM approx'].map(a => (
            <span key={a} className="ap-chip mono">{a}</span>
          ))}
        </div>
      </div>
    </div>
  )

  return (
    <div className="robustness-dashboard">
      <div className="rb-header">
        <h2 className="rb-title">Robustness Evaluation</h2>
        <span className="rb-meta mono">
          n={data.n_urls} URLs · 18 attack strategies across 3 modalities
        </span>
      </div>

      <FusionResilienceCard
        fusion={data.fusion}
        url={data.url_agent}
        text={data.text_agent}
        image={data.image_agent}
      />

      <div className="evasion-grid">
        <EvasionTable
          title="URL Agent — 7 URL Mutation Attacks"
          agentColor="accent"
          data={data.url_agent}
        />
        <EvasionTable
          title="Text Agent — 6 Text Adversarial Attacks"
          agentColor="purple"
          data={data.text_agent}
        />
        <EvasionTable
          title="Image Agent — 5 Visual Perturbations"
          agentColor="warn"
          data={data.image_agent}
        />
      </div>

      <div className="rb-plots-note">
        <span className="mono">
          📊 Paper figures generated in <code>evaluation/robustness/</code>:&nbsp;
          url_evasion_bar.png · text_evasion_bar.png · visual_evasion_bar.png ·
          robustness_radar.png · *_delta_heatmap.png
        </span>
      </div>
    </div>
  )
}

// NOTE: TextRobustnessDemo and VisualAttackDemo are also accessible
// inline in the Analyzer tab after each analysis (via AnalyzerView).
// The RobustnessDashboard shows the batch benchmark results from the
// run_robustness.py script. See the Analyzer tab for live per-URL demos.
