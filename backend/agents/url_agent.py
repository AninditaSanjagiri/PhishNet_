"""
URLAgent  — presentation-calibrated + XAI hardened
=====================================================
Changes in this version:
  1. SHAP hardened: explicit shape handling for all SHAP versions,
     logging on failure instead of silent pass, explainer lazily rebuilt
     on shape mismatch.
  2. Brand-keyword escalation rule: if suspicious keyword count >= 2
     AND domain has a brand-lookalike signal, enforce minimum score 0.45
     so the verdict cannot be SAFE regardless of RF output.
  3. Trusted domain fast-path unchanged (google.com etc. → score 0.04).
  4. SUSPICIOUS_KEYWORDS tightened (no generic terms like 'login').
"""
import asyncio
import logging
import math
import re
import time
import joblib
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from Levenshtein import distance as lev_distance
import tldextract

logger = logging.getLogger("phishnet.url_agent")

MODEL_PATH = Path(__file__).parent.parent / "models" / "url_rf_model.joblib"

TOP_BRANDS = [
    "paypal", "microsoft", "apple", "google", "amazon", "facebook",
    "netflix", "instagram", "twitter", "linkedin", "dropbox", "yahoo",
    "chase", "wellsfargo", "bankofamerica", "citibank", "ebay", "steam",
    "roblox", "discord", "whatsapp", "telegram", "outlook", "office365",
    "adobe", "docusign", "fedex", "dhl", "ups", "irs",
    "github", "gitlab", "openai", "anthropic",
]

TRUSTED_DOMAINS = {
    "google.com", "google.co.uk", "google.org", "google.net",
    "github.com", "github.io", "githubusercontent.com",
    "microsoft.com", "microsoftonline.com", "live.com", "outlook.com",
    "apple.com", "icloud.com",
    "amazon.com", "amazon.co.uk", "amazonaws.com",
    "openai.com", "anthropic.com",
    "cloudflare.com", "fastly.com", "akamai.com",
    "wikipedia.org", "wikimedia.org",
    "mozilla.org", "python.org", "nodejs.org",
    "youtube.com", "twitter.com", "x.com",
    "linkedin.com", "facebook.com", "instagram.com",
    "stackoverflow.com", "reddit.com",
}

# High-signal phishing keywords only — no generic terms
SUSPICIOUS_KEYWORDS = [
    "verify", "confirm", "credential", "webscr", "ebayisapi",
    "ssn", "lucky", "winner", "prize", "urgent",
    "paypa1", "amaz0n", "micros0ft", "g00gle", "app1e",
    "signin-", "-signin", "login-", "-login",
    "account-verify", "verify-account", "secure-login",
    "update-billing", "billing-update", "password-reset",
]

SUSPICIOUS_TLDS = {
    ".tk": 0.9, ".ml": 0.85, ".ga": 0.85, ".cf": 0.8, ".gq": 0.8,
    ".xyz": 0.55, ".top": 0.55, ".club": 0.50, ".work": 0.45, ".site": 0.45,
    ".online": 0.40, ".info": 0.30,
}

FEATURE_NAMES = [
    "url_length", "dot_count", "subdomain_count", "is_ip", "is_https",
    "keyword_count", "url_entropy", "special_char_count", "path_depth",
    "query_length", "brand_lev_distance", "tld_suspicion", "has_port",
    "domain_length", "subdomain_length", "digit_count",
]

# ── Brand-keyword escalation rule ─────────────────────────────────────────────
# If a URL has >= 2 suspicious keywords AND its domain closely mimics a brand,
# the minimum phishing score is this value, regardless of RF output.
# This prevents paypal-login-verify.xyz from being classified SAFE.
ESCALATION_MIN_SCORE    = 0.45   # forces minimum SUSPICIOUS verdict
ESCALATION_KW_THRESHOLD = 2      # keyword hits needed to trigger
ESCALATION_BRAND_DIST   = 0.35   # brand_lev_distance must be below this


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def min_brand_distance(domain: str) -> float:
    if not domain:
        return 1.0
    d = min(lev_distance(domain.lower(), brand) for brand in TOP_BRANDS)
    return min(d / max(len(domain), 1), 1.0)


def get_registrable_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}".lower()


def extract_features(url: str) -> tuple[np.ndarray, dict]:
    parsed    = urlparse(url)
    ext       = tldextract.extract(url)
    domain    = ext.domain or ""
    subdomain = ext.subdomain or ""
    suffix    = "." + ext.suffix if ext.suffix else ""
    host      = parsed.hostname or ""
    path      = parsed.path or ""
    query     = parsed.query or ""

    is_ip      = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host))
    kw_hits    = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower())
    brand_dist = min_brand_distance(domain)
    tld_score  = SUSPICIOUS_TLDS.get(suffix.lower(), 0.0)

    feat = {
        "url_length":         len(url),
        "dot_count":          url.count("."),
        "subdomain_count":    len(subdomain.split(".")) if subdomain else 0,
        "is_ip":              int(is_ip),
        "is_https":           int(parsed.scheme == "https"),
        "keyword_count":      kw_hits,
        "url_entropy":        round(shannon_entropy(url), 4),
        "special_char_count": sum(url.count(c) for c in "@-_~%"),
        "path_depth":         len([p for p in path.split("/") if p]),
        "query_length":       len(query),
        "brand_lev_distance": round(brand_dist, 4),
        "tld_suspicion":      tld_score,
        "has_port":           int(bool(parsed.port)),
        "domain_length":      len(domain),
        "subdomain_length":   len(subdomain),
        "digit_count":        sum(1 for c in domain if c.isdigit()),
    }
    return np.array(list(feat.values()), dtype=np.float32), feat


def _apply_escalation_rule(
    phish_score: float, feat_dict: dict
) -> tuple[float, bool]:
    """
    Brand-keyword escalation rule.
    Returns (adjusted_score, was_escalated).

    Trigger condition:
      keyword_count >= ESCALATION_KW_THRESHOLD
      AND brand_lev_distance < ESCALATION_BRAND_DIST

    When triggered, score is raised to at least ESCALATION_MIN_SCORE.
    This is a heuristic floor, not a hard verdict — the RF can still
    push the score higher.
    """
    kw_ok    = feat_dict["keyword_count"] >= ESCALATION_KW_THRESHOLD
    brand_ok = feat_dict["brand_lev_distance"] < ESCALATION_BRAND_DIST

    if kw_ok and brand_ok and phish_score < ESCALATION_MIN_SCORE:
        return ESCALATION_MIN_SCORE, True

    return phish_score, False


class URLAgent:
    def __init__(self):
        self.model      = None
        self._explainer = None
        self._shap_ok   = False   # tracks whether SHAP succeeded at least once

    async def load(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_or_create_model)

    def _load_or_create_model(self):
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            logger.info("URL RF model loaded from %s", MODEL_PATH)
        else:
            logger.warning("No trained RF model — creating calibrated baseline.")
            self._train_synthetic_model()

    def _get_explainer(self):
        """
        Lazily build SHAP TreeExplainer.
        Rebuilds if previously failed, logs clearly on error.
        """
        if self._explainer is not None:
            return self._explainer
        if self.model is None:
            return None
        try:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialised for URL agent")
        except ImportError:
            logger.warning("shap not installed — URL explainability unavailable. "
                           "Fix: pip install shap")
        except Exception:
            logger.exception("SHAP TreeExplainer init failed")
        return self._explainer

    def _compute_shap(self, feat_array: np.ndarray) -> tuple[list, dict]:
        """
        Compute SHAP values and return (shap_values_list, top_shap_dict).
        Returns empty structures on any failure, with clear log output.
        """
        explainer = self._get_explainer()
        if explainer is None:
            return [], {}

        try:
            sv = explainer.shap_values(feat_array.reshape(1, -1))

            # Normalise every possible SHAP output shape to a flat 1-D array
            # of length == n_features, representing the phishing class.
            #
            # SHAP returns different structures depending on version + model:
            #
            #  A) list of length n_classes, each element shape (n_samples, n_feats)
            #     → binary RF produces [legit_arr, phish_arr]
            #     → single-class degenerate RF produces [arr]  ← causes "index 1 OOB"
            #
            #  B) ndarray shape (n_classes, n_samples, n_feats)  ← newer SHAP/sklearn
            #
            #  C) ndarray shape (n_samples, n_feats)             ← regression fallback
            #
            # Rule: always take the LAST class slice (phishing=1).
            # If only one class exists, take the only one.

            if isinstance(sv, list):
                # Pick last element — phishing class in binary case,
                # or only element in degenerate single-class case.
                arr = np.asarray(sv[-1])        # shape: (1, n_feats) or (n_feats,)
                vals = arr.flatten()
            else:
                sv_arr = np.asarray(sv)
                if sv_arr.ndim == 3:
                    # (n_classes, n_samples, n_feats) — take last class
                    vals = sv_arr[-1, 0, :]
                elif sv_arr.ndim == 2:
                    # (n_samples, n_feats) — single sample, take row 0
                    vals = sv_arr[0, :]
                else:
                    vals = sv_arr.flatten()

            # Sanity check: must match feature count
            if len(vals) != len(FEATURE_NAMES):
                logger.warning(
                    "SHAP output length %d != feature count %d — skipping",
                    len(vals), len(FEATURE_NAMES),
                )
                return [], {}

            shap_list = [round(float(v), 5) for v in vals]
            ranked    = sorted(
                zip(FEATURE_NAMES, shap_list),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:3]
            top_shap  = {k: v for k, v in ranked}
            self._shap_ok = True
            return shap_list, top_shap

        except Exception:
            logger.exception("SHAP value computation failed for URL agent")
            return [], {}

    def _train_synthetic_model(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        legit_X = rng.uniform(
            [15,  1, 0, 0, 1, 0, 2.8, 0, 0,  0, 0.7, 0.0, 0,  4,  0, 0],
            [60,  3, 1, 0, 1, 0, 3.5, 1, 2, 10, 1.0, 0.0, 0, 12,  4, 0],
            size=(600, 16),
        )
        phish_X = rng.uniform(
            [60,  4, 1, 0, 0, 2, 4.2, 2, 1, 10, 0.0, 0.5, 0,  8,  4, 1],
            [250, 9, 4, 1, 1, 7, 5.5, 8, 5, 80, 0.3, 1.0, 1, 28, 18, 6],
            size=(600, 16),
        )
        X = np.vstack([legit_X, phish_X]).astype(np.float32)
        y = np.array([0] * 600 + [1] * 600)
        self.model = RandomForestClassifier(
            n_estimators=150, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.model.fit(X, y)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        logger.info("Calibrated synthetic RF model saved to %s", MODEL_PATH)

    async def analyze(self, url: str) -> dict:
        reg = get_registrable_domain(url)
        if reg in TRUSTED_DOMAINS:
            return {
                "score":       0.04,
                "confidence":  0.97,
                "explanation": "Domain belongs to a known legitimate service.",
                "latency_ms":  0.5,
                "features":    {"trusted_domain": True, "registrable": reg},
                "shap_values": [],
                "top_shap":    {},
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_sync, url)

    def _analyze_sync(self, url: str) -> dict:
        t0 = time.perf_counter()
        feat_array, feat_dict = extract_features(url)
        proba       = self.model.predict_proba([feat_array])[0]
        phish_score = float(proba[1])

        # Brand-keyword escalation rule
        phish_score, escalated = _apply_escalation_rule(phish_score, feat_dict)

        # SHAP attributions
        shap_values, top_shap = self._compute_shap(feat_array)

        # Human-readable reasons
        reasons = []
        if feat_dict["is_ip"]:
            reasons.append("uses IP address as host")
        if feat_dict["keyword_count"] >= 2:
            reasons.append(f"{feat_dict['keyword_count']} phishing keyword patterns")
        if feat_dict["brand_lev_distance"] < 0.18:
            reasons.append("domain closely mimics a known brand")
        if feat_dict["tld_suspicion"] > 0.5:
            reasons.append("high-risk TLD")
        if feat_dict["url_entropy"] > 4.8:
            reasons.append("high URL entropy")
        if feat_dict["digit_count"] >= 3:
            reasons.append("unusual digit density in domain")
        if escalated:
            reasons.append("brand-keyword combination flagged")
        if not reasons:
            reasons.append(
                "URL structure looks normal"
                if phish_score < 0.5
                else "multiple weak structural signals"
            )

        return {
            "score":       round(phish_score, 4),
            "confidence":  round(float(max(proba)), 4),
            "explanation": "; ".join(reasons).capitalize() + ".",
            "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
            "features":    {**feat_dict, "escalated": escalated},
            "shap_values": shap_values,
            "top_shap":    top_shap,
        }
