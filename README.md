# PhishNet v3 🛡️
### An Explainable Multimodal Phishing Detection Framework with Robustness Evaluation Against Adaptive Attacks

A college-level research prototype that directly justifies every word of the paper title:

| Title Component | Implementation |
|---|---|
| **Explainable** | SHAP (URL agent) + LIME (Text agent) + GradCAM (Image agent) |
| **Multimodal** | URL features + HTML text + Screenshot — fused late |
| **Phishing Detection** | Binary classifier: phishing / suspicious / safe |
| **Robustness Evaluation** | 18 attacks across 3 modalities, evasion rates, score-drop heatmaps |
| **Adaptive Attacks** | 7 URL mutations + 6 text evasion + 5 visual perturbations |

---

## Architecture

```
User submits URL
      │
      ▼
FastAPI Backend  ─── asyncio.gather() ───────────────┐
      │                                               │
   URL Agent          Text Agent           Image Agent
   RandomForest        DistilBERT          MobileNetV3
   + SHAP              + LIME              + GradCAM
   ~2ms                ~80ms               ~120ms+screenshot
      │                    │                    │
      └──────── FusionAgent (weighted late fusion) ───┘
                           │
                    Final Verdict
           phishing_probability, dominant_modality,
           top_shap, lime_tokens, modality_contributions
                           │
              ┌────────────┼─────────────┐
         /analyze    /robustness/*   /evaluation/*
```

---

## Robustness: 18 Attack Strategies

### URL Mutations (7)
| Attack | Description |
|---|---|
| `homograph_swap` | Replace ASCII letters with Unicode lookalikes (а vs a) |
| `subdomain_inject` | Prepend `brand-secure.` subdomain |
| `tld_substitution` | Swap `.tk` → `.com` |
| `path_noise` | Append benign path segments |
| `https_spoof` | Force HTTPS scheme |
| `keyword_dilution` | Add benign query parameters |
| `entropy_reduction` | Replace high-entropy chars to lower URL entropy |

### Text Adversarial Attacks (6)
| Attack | Description |
|---|---|
| `synonym_swap` | Replace phishing keywords with synonyms |
| `paraphrase_urgency` | Soften urgency trigger phrases |
| `negation_inject` | Insert negation into non-critical verbs |
| `whitespace_inject` | Zero-width space inside trigger words |
| `leet_substitution` | p@ssw0rd-style character replacements |
| `sentence_dilution` | Pad with benign boilerplate sentences |

### Visual Perturbations (5)
| Attack | Description |
|---|---|
| `gaussian_noise` | Add pixel noise (σ=15) |
| `jpeg_compression` | Re-encode at quality=10 |
| `brightness_shift` | Increase brightness ×1.8 |
| `pixel_block_mask` | Blank 20 random 10×10 blocks |
| `fgsm_approximation` | Finite-difference gradient sign perturbation (ε=8) |

---

## Quick Start

```bash
# 1. Restore/clone
git clone https://github.com/yourname/phishnet
cd phishnet/backend

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium && playwright install-deps chromium  # Linux

# 3. Start backend (auto-creates synthetic baseline model)
python main.py                  # → http://localhost:8000

# 4. Start frontend
cd ../frontend && npm install && npm run dev   # → http://localhost:5173
```

---

## Full Research Workflow

### Step 1: Train real models

```bash
cd backend

# Train URL Random Forest on PhishTank + Tranco (auto-downloads ~5k URLs)
python training/train_url_model.py --auto-download --limit 5000

# Fine-tune DistilBERT on public phishing email datasets
python training/train_text_model.py --epochs 3
# Quick smoke test:
python training/train_text_model.py --fast
```

### Step 2: Run ablation evaluation

```bash
# Quick (auto-downloads ~200 test URLs)
python evaluation/run_evaluation.py --fast

# Full (1000 URLs)
python evaluation/run_evaluation.py --n-samples 1000

# Outputs: evaluation/latest_metrics.json, ablation_table.md,
#          roc_comparison.png, confusion_*.png, latency_bar.png
```

### Step 3: Run robustness benchmark

```bash
# Quick (10 URLs, all 18 attacks)
python robustness/run_robustness.py --fast

# Full (20 URLs)
python robustness/run_robustness.py --n 20

# Outputs: evaluation/robustness/robustness_summary.json,
#          url/text/visual_evasion_bar.png, *_delta_heatmap.png,
#          robustness_radar.png, url/text/visual_evasion_table.md
```

### Step 4: Generate all paper figures

```bash
python robustness/generate_paper_figures.py

# Outputs: evaluation/paper_figures/
#   fig2_ablation.png          (grouped bar: Acc/Prec/Rec/F1/AUC × 3 systems)
#   fig3_url_evasion.png       (URL attack evasion bar chart)
#   fig4_text_evasion.png      (text attack evasion bar chart)
#   fig5_visual_evasion.png    (visual attack evasion bar chart)
#   fig6_robustness_radar.png  (radar: clean vs worst-attack per agent)
#   fig7_score_drop_heatmap.png (agents × attacks heatmap)
#   fig8_latency_comparison.png (PhishNet vs SAHF-PD latency)
#   table1_ablation.md + .tex
#   table2_robustness.md + .tex
```

---

## API Reference

### `POST /analyze` — Full multimodal analysis
```json
{
  "url": "http://paypa1-secure.tk/verify"
}
```
Returns:
- `verdict`, `phishing_probability`
- `dominant_modality` — which agent drove the decision
- `top_shap` — SHAP feature attributions (URL agent)
- `lime_tokens` — LIME token weights (Text agent)
- `latency_breakdown` — per-agent + wall-clock ms
- `screenshot_base64`, `gradcam_base64`

### `POST /robustness/url` — Live URL attack demo
```json
{ "url": "http://example-phish.tk/verify" }
```
Returns per-attack scores for all 7 URL mutations.

### `POST /robustness/text` — Live text attack demo
```json
{ "text": "Your account has been suspended. Verify immediately." }
```
Returns per-attack scores for all 6 text adversarial attacks.

### `POST /robustness/visual` — Live visual attack demo
```json
{ "image_base64": "<base64-encoded PNG>" }
```
Returns per-attack scores for all 5 visual perturbations.

### `GET /evaluation/summary` — Ablation results JSON
### `GET /evaluation/robustness` — Robustness results JSON
### `GET /history` — Recent analysis log

---

## File Tree

```
phishnet/
├── backend/
│   ├── main.py                          # FastAPI app (v3 — robustness endpoints)
│   ├── database.py
│   ├── requirements.txt
│   ├── agents/
│   │   ├── orchestrator.py              # asyncio.gather + memory tracking
│   │   ├── url_agent.py                 # RF + SHAP
│   │   ├── text_agent.py                # DistilBERT + LIME
│   │   ├── image_agent.py               # MobileNetV3 + GradCAM
│   │   └── fusion_agent.py              # Late fusion + dominant_modality
│   ├── robustness/
│   │   ├── run_robustness.py            # Master benchmark runner
│   │   ├── generate_paper_figures.py    # All publication figures
│   │   └── attacks/
│   │       ├── url_attacks.py           # 7 URL mutation strategies
│   │       ├── text_attacks.py          # 6 text adversarial attacks
│   │       └── visual_attacks.py        # 5 visual perturbation attacks
│   ├── training/
│   │   ├── train_url_model.py           # RF on PhishTank + Tranco
│   │   └── train_text_model.py          # DistilBERT fine-tune
│   ├── evaluation/
│   │   └── run_evaluation.py            # Ablation study runner
│   └── utils/
│       └── url_validator.py
│
├── frontend/
│   └── src/
│       ├── App.jsx                      # 4 views: Analyzer / Robustness / Eval / History
│       ├── components/
│       │   ├── AnalyzerView.jsx         # Main analysis + attack toggle
│       │   ├── VerdictBanner.jsx
│       │   ├── AgentCard.jsx
│       │   ├── ExplainPanel.jsx         # SHAP bars + LIME tokens
│       │   ├── FusionPanel.jsx
│       │   ├── LatencyPanel.jsx
│       │   ├── ScreenshotPanel.jsx      # Screenshot + GradCAM toggle
│       │   ├── RobustnessPanel.jsx      # Live URL attack suite
│       │   ├── TextRobustnessDemo.jsx   # Live text attack demo
│       │   ├── VisualAttackDemo.jsx     # Live visual attack demo
│       │   ├── RobustnessDashboard.jsx  # Full benchmark results page
│       │   ├── EvalDashboard.jsx        # Ablation table page
│       │   └── HistoryView.jsx
│       └── utils/api.js                 # All 8 API functions
```

---

## Research Paper Claims Supported

| Claim | Evidence |
|---|---|
| Multimodal fusion improves over single modality | `ablation_table.md` F1 gain row, `fig2_ablation.png` |
| Sub-200ms inference (vs 5–15s LLM systems) | `fig8_latency_comparison.png`, `latency_breakdown` in API |
| Per-decision explainability | `top_shap` + `lime_tokens` in JSON, `ExplainPanel` screenshots |
| Dominant modality attribution | `dominant_modality` field in every `/analyze` response |
| URL attack robustness quantified | `fig3_url_evasion.png`, `table2_robustness.md` |
| Text attack robustness (novel — not in prior work) | `fig4_text_evasion.png` — first benchmark of DistilBERT against semantic attacks |
| Visual attack robustness quantified | `fig5_visual_evasion.png` |
| Fusion more robust than any single agent | `fig6_robustness_radar.png`, `fusion.robustness_gain_vs_worst_agent` |

---

## Datasets (Free, No Login)

| Dataset | Use | Link |
|---|---|---|
| PhishTank feed | URL + screenshot (phishing) | phishtank.org/developer_info.php |
| Tranco Top-1M | URL + screenshot (legit) | tranco-list.eu |
| HuggingFace `zefang-liu/phishing-email-dataset` | DistilBERT fine-tune | huggingface.co |
| HuggingFace `ealvaradob/phishing-dataset-multilingual` | DistilBERT fine-tune | huggingface.co |

---

## Deployment (Free)

```bash
# Frontend → Vercel
cd frontend && vercel --prod
# Set: VITE_API_BASE = https://your-backend.onrender.com

# Backend → Render
# Build: pip install -r requirements.txt && playwright install chromium
# Start: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

*PhishNet v3 — Built as a college-level research prototype.*
*Not intended for production security use without additional hardening.*
