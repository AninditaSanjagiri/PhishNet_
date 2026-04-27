"""
FusionAgent  — updated thresholds + escalation awareness
===========================================================
Verdict thresholds (Task 1):
   0 – 19  → SAFE
  20 – 49  → SUSPICIOUS
  50+      → HIGH RISK  (maps to "phishing" internally)

Safe-override rule updated:
  Only applies when URL score < 0.12 AND no escalation flag set by url_agent.
  This prevents paypal-verify.xyz being forced SAFE just because RF is
  uncertain.
"""
import logging
import joblib
import numpy as np
from pathlib import Path

logger = logging.getLogger("phishnet.fusion_agent")

FUSION_MODEL_PATH = Path(__file__).parent.parent / "models" / "fusion_lr.joblib"

STATIC_WEIGHTS = {"url": 0.55, "text": 0.30, "image": 0.15}

# ── New thresholds (Task 1) ───────────────────────────────────────────────────
# Internal verdict names map to frontend labels:
#   "phishing"   → HIGH RISK    (score >= 50)
#   "suspicious" → SUSPICIOUS   (score 20–49)
#   "safe"       → SAFE         (score 0–19)
THRESHOLDS = {"phishing": 50.0, "suspicious": 20.0}

# Safe override: only triggers when URL score is very low AND no escalation
SAFE_OVERRIDE_THRESHOLD = 0.12


class FusionAgent:
    def __init__(self):
        self.lr_model = None

    async def load(self):
        if FUSION_MODEL_PATH.exists():
            try:
                self.lr_model = joblib.load(FUSION_MODEL_PATH)
                logger.info("Fusion model loaded from %s", FUSION_MODEL_PATH)
            except Exception:
                logger.exception("Fusion model load failed — using static weights")
        else:
            logger.info("No fusion model — using calibrated static weights")

    def fuse(self, url_r: dict, text_r: dict, image_r: dict) -> dict:
        scores = {
            "url":   url_r.get("score"),
            "text":  text_r.get("score"),
            "image": image_r.get("score"),
        }
        available = {k: v for k, v in scores.items() if v is not None}

        if not available:
            return {
                "verdict":                "suspicious",
                "phishing_probability":   34.0,
                "fusion_weights":         {},
                "dominant_modality":      "none",
                "modality_contributions": {},
            }

        url_score = scores.get("url")
        url_features   = url_r.get("features", {})
        was_escalated  = url_features.get("escalated", False)

        # Safe override: URL agent very confident AND no escalation triggered
        if (url_score is not None
                and url_score < SAFE_OVERRIDE_THRESHOLD
                and not was_escalated):
            return {
                "verdict":                "safe",
                "phishing_probability":   round(url_score * 100, 2),
                "fusion_weights":         {"method": "safe_override"},
                "dominant_modality":      "url",
                "modality_contributions": {"url": round(url_score, 4)},
            }

        # Learned model (if available and all 3 scores present)
        if self.lr_model is not None and len(available) == 3:
            feat   = np.array([[scores["url"], scores["text"], scores["image"]]])
            proba  = self.lr_model.predict_proba(feat)[0]
            prob   = float(proba[1])
            method = "learned_mlp"
            try:
                coef     = self.lr_model.named_steps["lr"].coef_[0]
                sc       = self.lr_model.named_steps["scaler"]
                normed   = sc.transform(feat)[0]
                contribs = {
                    "url":   round(float(coef[0] * normed[0]), 4),
                    "text":  round(float(coef[1] * normed[1]), 4),
                    "image": round(float(coef[2] * normed[2]), 4),
                }
            except Exception:
                contribs = {k: round(v, 4) for k, v in available.items()}
            eff_weights = {"url": None, "text": None, "image": None, "method": method}
        else:
            total       = sum(STATIC_WEIGHTS[k] for k in available)
            eff_weights = {k: round(STATIC_WEIGHTS[k] / total, 4) for k in available}
            eff_weights["method"] = "weighted_average"
            prob     = sum(eff_weights[k] * v for k, v in available.items()
                           if k in eff_weights)
            contribs = {k: round(eff_weights[k] * v, 4)
                        for k, v in available.items() if k in eff_weights}

        decisiveness = {k: abs(v - 0.5) for k, v in available.items()}
        dominant     = max(decisiveness, key=decisiveness.get)
        if decisiveness[dominant] < 0.05:
            dominant = "balanced"

        prob_pct = prob * 100.0

        # Apply the new thresholds
        verdict = (
            "phishing"   if prob_pct >= THRESHOLDS["phishing"]   else
            "suspicious" if prob_pct >= THRESHOLDS["suspicious"]  else
            "safe"
        )

        # Escalation override: if url_agent triggered brand-keyword rule,
        # ensure verdict is at least SUSPICIOUS
        if was_escalated and verdict == "safe":
            verdict  = "suspicious"
            prob_pct = max(prob_pct, THRESHOLDS["suspicious"])

        return {
            "verdict":                verdict,
            "phishing_probability":   round(prob_pct, 2),
            "fusion_weights":         eff_weights,
            "dominant_modality":      dominant,
            "modality_contributions": contribs,
        }
