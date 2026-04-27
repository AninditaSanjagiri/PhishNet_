"""
PhishNet API  — v3 production
================================
Key fix in this version: Windows asyncio + Playwright compatibility.
See agents/image_agent.py for full explanation.

The event-loop policy is set at module level (before uvicorn starts)
using the standard pattern recommended by the asyncio docs:
  sys.platform == "win32"  ->  ProactorEventLoop
This ensures that IF Playwright's async_api is ever called (it won't
be after our refactor, but defensive programming is good), it will
not crash.  The real fix is using sync_playwright in a thread pool.
"""
import asyncio
import logging
import platform
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

# -------------------------------------------------------------------
# Windows event-loop policy
# Must be set BEFORE any asyncio.run() / uvicorn.run() call.
# ProactorEventLoop supports subprocess spawning on Windows.
# On Linux/macOS this block is a no-op.
# -------------------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# -------------------------------------------------------------------
# Structured logging  (replaces bare print() calls)
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phishnet.main")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn

from agents.orchestrator import OrchestratorAgent
from database import init_db, log_analysis
from utils.url_validator import normalize_url

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PhishNet v3 starting — platform: %s", platform.system())
    await init_db()
    app.state.orchestrator = OrchestratorAgent()
    await app.state.orchestrator.initialize()
    logger.info("All agents ready.")
    yield
    logger.info("PhishNet shutdown.")


app = FastAPI(
    title="PhishNet API",
    description="Multimodal Phishing Detection System",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        v = v.strip()
        if not v.startswith(("http://","https://")):
            v = "http://" + v
        return v


class LIMEToken(BaseModel):
    token:  str
    weight: float


class AgentResult(BaseModel):
    score:       Optional[float]
    confidence:  float
    explanation: str
    latency_ms:  float = 0.0
    features:    dict
    shap_values: list  = []
    top_shap:    dict  = {}
    lime_tokens: list  = []
    top_tokens:  list  = []


class LatencyBreakdown(BaseModel):
    url_ms:        float
    text_ms:       float
    image_ms:      float
    total_wall_ms: float


class AnalyzeResponse(BaseModel):
    url:                     str
    verdict:                 str
    phishing_probability:    float
    dominant_modality:       str = "unknown"
    modality_contributions:  dict = {}
    url_agent:               AgentResult
    text_agent:              AgentResult
    image_agent:             AgentResult
    fusion_weights:          dict
    latency_breakdown:       LatencyBreakdown
    memory_delta_mb:         float = 0.0
    screenshot_base64:       Optional[str]
    gradcam_base64:          Optional[str]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status":"ok","version":"2.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, bg: BackgroundTasks):
    url = normalize_url(req.url)
    try:
        result = await app.state.orchestrator.analyze(url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    resp = AnalyzeResponse(
        url=url,
        verdict=result["verdict"],
        phishing_probability=round(result["phishing_probability"], 2),
        dominant_modality=result.get("dominant_modality","unknown"),
        modality_contributions=result.get("modality_contributions",{}),
        url_agent=AgentResult(**result["url_agent"]),
        text_agent=AgentResult(**result["text_agent"]),
        image_agent=AgentResult(**result["image_agent"]),
        fusion_weights=result["fusion_weights"],
        latency_breakdown=LatencyBreakdown(**result["latency_breakdown"]),
        memory_delta_mb=result.get("memory_delta_mb",0.0),
        screenshot_base64=result.get("screenshot_base64"),
        gradcam_base64=result.get("gradcam_base64"),
    )
    bg.add_task(log_analysis, url, resp.model_dump())
    return resp


@app.get("/history")
async def history(limit: int = 50):
    from database import get_recent_analyses
    return await get_recent_analyses(limit)


@app.delete("/history")
async def clear_history():
    from database import clear_all_analyses
    await clear_all_analyses()
    return {"status": "cleared"}


@app.get("/evaluation/summary")
async def eval_summary():
    """Return latest evaluation metrics from file if available."""
    import json
    from pathlib import Path
    p = Path("evaluation/latest_metrics.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"error": "No evaluation run yet. Run: python evaluation/run_evaluation.py"}


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Correct startup command for Windows + Linux
    #
    # Windows:  DO NOT use reload=True with ProactorEventLoop —
    #           the file-watcher spawns child processes which conflict.
    #           Use reload=False in development, or run via the CLI:
    #               uvicorn main:app --host 0.0.0.0 --port 8000
    #
    # Linux:    reload=True is safe.
    #
    # Both platforms: loop="asyncio" is explicit and safe.
    #   DO NOT use loop="uvloop" on Windows (uvloop is Linux-only).
    # ------------------------------------------------------------------
    is_windows = sys.platform == "win32"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=not is_windows,   # reload off on Windows, on on Linux
        loop="asyncio",          # explicit — never "auto" which may pick uvloop
        log_level="info",
    )


# ── Robustness endpoints (added in v3) ────────────────────────────────────────

class URLRobustnessRequest(BaseModel):
    url: str

class TextRobustnessRequest(BaseModel):
    text: str

class VisualRobustnessRequest(BaseModel):
    image_base64: str   # base64-encoded PNG


@app.post("/robustness/url")
async def robustness_url(req: URLRobustnessRequest):
    """
    Apply all 7 URL adversarial attacks to a URL and return per-attack scores.
    Used by the frontend RobustnessPanel live demo.
    """
    from robustness.attacks.url_attacks import apply_all_attacks
    url       = normalize_url(req.url)
    baseline  = await app.state.orchestrator.url_agent.analyze(url)
    base_score = baseline.get("score", 0.5)

    attacked   = apply_all_attacks(url)
    results    = []
    for atk in attacked:
        if atk["is_noop"]:
            results.append({**atk, "perturbed_score": base_score, "score_drop": 0.0})
            continue
        try:
            r = await app.state.orchestrator.url_agent.analyze(atk["perturbed_url"])
            ps = r.get("score", 0.5)
        except Exception:
            ps = 0.5
        results.append({
            **{k: v for k, v in atk.items() if k != "perturbed_img"},
            "baseline_score":   round(base_score, 4),
            "perturbed_score":  round(ps, 4),
            "score_drop":       round(max(0.0, base_score - ps), 4),
            "evades":           ps < 0.5 and base_score >= 0.5,
        })

    n_evade = sum(1 for r in results if r.get("evades"))
    return {
        "url":            url,
        "baseline_score": round(base_score, 4),
        "n_attacks":      len(results),
        "n_evaded":       n_evade,
        "evasion_rate":   round(n_evade / max(len(results), 1), 4),
        "attacks":        results,
    }


@app.post("/robustness/text")
async def robustness_text(req: TextRobustnessRequest):
    """Apply all 6 text adversarial attacks and return per-attack scores."""
    from robustness.attacks.text_attacks import apply_all_text_attacks
    import asyncio

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    fake_html  = f"<html><body><p>{text}</p></body></html>"
    loop       = asyncio.get_event_loop()
    base_result = await loop.run_in_executor(
        None, app.state.orchestrator.text_agent._classify, fake_html)
    base_score  = base_result.get("score", 0.5)

    attacked = apply_all_text_attacks(text)
    results  = []
    for atk in attacked:
        fake = f"<html><body><p>{atk['perturbed_text']}</p></body></html>"
        r    = await loop.run_in_executor(
            None, app.state.orchestrator.text_agent._classify, fake)
        ps   = r.get("score", 0.5)
        results.append({
            "attack_name":    atk["attack_name"],
            "n_changes":      atk["n_changes"],
            "char_delta":     atk["char_delta"],
            "baseline_score": round(base_score, 4),
            "perturbed_score":round(ps, 4),
            "score_drop":     round(max(0.0, base_score - ps), 4),
            "evades":         ps < 0.5 and base_score >= 0.5,
        })

    n_evade = sum(1 for r in results if r.get("evades"))
    return {
        "baseline_score": round(base_score, 4),
        "n_attacks":      len(results),
        "n_evaded":       n_evade,
        "evasion_rate":   round(n_evade / max(len(results), 1), 4),
        "attacks":        results,
    }


@app.post("/robustness/visual")
async def robustness_visual(req: VisualRobustnessRequest):
    """Apply all 5 visual perturbation attacks and return per-attack scores."""
    from robustness.attacks.visual_attacks import (
        apply_all_visual_attacks, base64_to_pil)
    import asyncio

    try:
        img = base64_to_pil(req.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}")

    loop       = asyncio.get_event_loop()
    base_score = await app.state.orchestrator.image_agent._classify.__func__(
        app.state.orchestrator.image_agent, img
    ) if False else 0.5  # placeholder — see note below

    # Proper async call
    base_r  = await loop.run_in_executor(
        None, app.state.orchestrator.image_agent._classify, img)
    base_score = base_r.get("score", 0.5)

    attacked = apply_all_visual_attacks(img)
    results  = []
    for atk in attacked:
        r  = await loop.run_in_executor(
            None, app.state.orchestrator.image_agent._classify,
            atk["perturbed_img"])
        ps = r.get("score", 0.5)
        results.append({
            "attack_name":      atk["attack_name"],
            "description":      atk["description"],
            "mean_pixel_delta": atk["mean_pixel_delta"],
            "baseline_score":   round(base_score, 4),
            "perturbed_score":  round(ps, 4),
            "score_drop":       round(max(0.0, base_score - ps), 4),
            "evades":           ps < 0.5 and base_score >= 0.5,
        })

    n_evade = sum(1 for r in results if r.get("evades"))
    return {
        "baseline_score": round(base_score, 4),
        "n_attacks":      len(results),
        "n_evaded":       n_evade,
        "evasion_rate":   round(n_evade / max(len(results), 1), 4),
        "attacks":        results,
    }


@app.get("/evaluation/robustness")
async def get_robustness_summary():
    """Return latest robustness benchmark results."""
    import json
    from pathlib import Path
    p = Path("evaluation/robustness/robustness_summary.json")
    if p.exists():
        return json.loads(p.read_text())
    return {
        "error": "No robustness run yet.",
        "hint":  "Run: python robustness/run_robustness.py --fast"
    }
