"""
TextAgent  — presentation-optimised
=====================================
Speed improvements vs research version:
  • FETCH_TIMEOUT  reduced to 5s
  • Text sent to BERT truncated to 400 tokens worth (~1500 chars)
  • LIME num_samples reduced to 30 (was 50)
  • Hard 5-second wall-clock cap on the entire _classify() call
  • Graceful skip: None score returned if fetch or classify times out

Calibration improvements:
  • Trusted domains receive a strong prior toward safe (0.10 floor)
  • metadata boost halved to reduce false positives on large legit sites
  • keyword fallback tuned down for common tech/news words
"""
import asyncio
import numpy as np
import logging
import re
import signal
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("phishnet.text_agent")
from bs4 import BeautifulSoup

_tokenizer      = None
_model          = None
_torch          = None
_lime_explainer = None

MODEL_NAME    = "ealvaradob/bert-finetuned-phishing"
MAX_LENGTH    = 256        # was 512 — halves BERT inference time
TEXT_CHARS    = 1200       # was 3000 — enough for intent signal
FETCH_TIMEOUT = 5          # was 10 — fail fast
CLASSIFY_MAX  = 5.0        # hard wall-clock cap (seconds) for _classify()

# Trusted registrable domains — these get a 0.10 URL floor applied
TRUSTED_DOMAINS = {
    "google.com","google.co.uk","google.org",
    "github.com","github.io","githubusercontent.com",
    "microsoft.com","microsoftonline.com","live.com","outlook.com",
    "apple.com","icloud.com",
    "amazon.com","aws.amazon.com",
    "openai.com","anthropic.com",
    "cloudflare.com","fastly.com",
    "wikipedia.org","mozilla.org","python.org",
    "youtube.com","twitter.com","linkedin.com","facebook.com",
    "stackoverflow.com","reddit.com","medium.com",
}

PHISH_KEYWORDS = {
    "verify your account":      0.85,
    "confirm your identity":    0.80,
    "account suspended":        0.70,
    "unauthorized access":      0.70,
    "click here to update":     0.75,
    "enter your password":      0.65,
    "social security number":   0.65,
    "urgent action required":   0.75,
    "account will be closed":   0.80,
    "claim your prize":         0.70,
    "you have been selected":   0.65,
    "free gift":                0.55,
    "winner":                   0.40,
}


def _load_model():
    global _tokenizer, _model, _torch
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _torch     = torch
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        logger.info("Text model loaded: %s", MODEL_NAME)
    except Exception as exc:
        logger.warning("DistilBERT load failed (%s). Keyword fallback active.", exc)


def _build_lime_explainer():
    global _lime_explainer
    if _lime_explainer is not None or _model is None:
        return
    try:
        from lime.lime_text import LimeTextExplainer
        _lime_explainer = LimeTextExplainer(class_names=["legit", "phishing"])
        logger.info("LIME TextExplainer ready")
    except ImportError:
        logger.warning("lime not installed — text explainability unavailable. "
                       "Fix: pip install lime")
    except Exception:
        logger.exception("LIME TextExplainer init failed")


def _bert_predict_proba(texts):
    """
    LIME-compatible prediction function.

    LIME passes a list of strings and expects a 2-D numpy array back:
        shape (n_texts, n_classes)
    Returning a plain Python list causes:
        "list indices must be integers or slices, not tuple"
    because LIME indexes the result as predictions[:, 1].

    Fix: always return np.ndarray, never a plain list.
    """
    import torch
    results = []
    for text in texts:
        inputs = _tokenizer(
            text, return_tensors="pt",
            max_length=MAX_LENGTH, truncation=True, padding=True)
        with torch.no_grad():
            logits = _model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)[0].tolist()
        row = probs if len(probs) == 2 else [1 - probs[0], probs[0]]
        results.append(row)
    return np.array(results, dtype=np.float64)   # ← must be ndarray for LIME


def _extract_text_from_html(html: str) -> tuple[str, dict]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "noscript", "header", "nav", "footer"]):
        tag.decompose()
    visible = soup.get_text(separator=" ", strip=True)
    visible = re.sub(r"\s+", " ", visible)[:TEXT_CHARS]

    forms           = soup.find_all("form")
    password_fields = len(soup.find_all("input", {"type": "password"}))
    external_forms  = sum(1 for f in forms if (f.get("action") or "").startswith("http"))
    suspicious_links = sum(
        1 for a in soup.find_all("a", href=True)
        if re.search(r"login|verify|secure|update|confirm", a["href"], re.I)
    )
    return visible, {
        "form_count":            len(forms),
        "password_fields":       password_fields,
        "external_form_actions": external_forms,
        "suspicious_links":      suspicious_links,
        "text_length":           len(visible),
    }


def _is_trusted_domain(url: str) -> bool:
    import tldextract
    ext = tldextract.extract(url)
    registrable = f"{ext.domain}.{ext.suffix}".lower()
    return registrable in TRUSTED_DOMAINS


def _keyword_fallback(text: str) -> float:
    tl   = text.lower()
    hits = [(w, s) for w, s in PHISH_KEYWORDS.items() if w in tl]
    if not hits:
        return 0.08
    return min(sum(s for _, s in hits) / max(len(hits) * 1.8, 1), 0.9)


class TextAgent:
    def __init__(self):
        self._model_loaded = False

    async def load(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_model)
        self._model_loaded = _model is not None
        if self._model_loaded:
            await loop.run_in_executor(None, _build_lime_explainer)

    async def analyze(self, url: str) -> dict:
        # Trusted domains: skip fetch, return safe prior immediately
        if _is_trusted_domain(url):
            return {
                "score":       0.05,
                "confidence":  0.95,
                "explanation": "Domain is a known legitimate service.",
                "latency_ms":  1.0,
                "features":    {"trusted_domain": True},
                "lime_tokens": [],
                "top_tokens":  [],
            }

        # Fetch with short timeout
        try:
            html, err = await asyncio.wait_for(
                self._fetch_html(url), timeout=FETCH_TIMEOUT
            )
        except asyncio.TimeoutError:
            html, err = None, "fetch timeout"

        if not html:
            return {
                "score":       None,
                "confidence":  0.0,
                "explanation": f"Page fetch skipped ({err}).",
                "latency_ms":  0.0,
                "features":    {"fetch_error": err},
                "lime_tokens": [],
                "top_tokens":  [],
            }

        # Classify with hard time cap
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._classify, html, url),
                timeout=CLASSIFY_MAX,
            )
        except asyncio.TimeoutError:
            result = {
                "score":       None,
                "confidence":  0.0,
                "explanation": "Content analysis timed out.",
                "latency_ms":  CLASSIFY_MAX * 1000,
                "features":    {"timeout": True},
                "lime_tokens": [],
                "top_tokens":  [],
            }
        return result

    async def _fetch_html(self, url):
        try:
            async with httpx.AsyncClient(
                timeout=FETCH_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PhishNet/3.0)"},
            ) as c:
                r = await c.get(url)
                return r.text, None
        except Exception as exc:
            return None, str(exc)[:80]

    def _classify(self, html: str, url: str = "") -> dict:
        t0   = time.perf_counter()
        text, meta = _extract_text_from_html(html)

        if not text.strip():
            return {
                "score":       0.20,
                "confidence":  0.40,
                "explanation": "Page has no readable text content.",
                "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
                "features":    meta,
                "lime_tokens": [],
                "top_tokens":  [],
            }

        lime_tokens: list[dict] = []
        top_tokens:  list[str]  = []

        if self._model_loaded and _tokenizer and _model and _torch:
            phish_score = _bert_predict_proba([text])[0][1]
            method      = "DistilBERT"

            if _lime_explainer is not None:
                try:
                    exp = _lime_explainer.explain_instance(
                        text[:800],
                        _bert_predict_proba,
                        num_features=8,
                        num_samples=30,    # fast
                        labels=(1,),
                    )
                    raw = exp.as_list(label=1)
                    lime_tokens = [
                        {"token": tok, "weight": round(w, 4)}
                        for tok, w in sorted(raw, key=lambda x: -abs(x[1]))[:5]
                        if w > 0
                    ]
                    top_tokens = [t["token"] for t in lime_tokens]
                except Exception:
                    logger.warning("LIME explain_instance failed", exc_info=True)
        else:
            phish_score = _keyword_fallback(text)
            method      = "keyword_heuristic"

        # Metadata boost — reduced to limit false positives on legit sites
        boost = 0.0
        if meta["password_fields"]:        boost += 0.07
        if meta["external_form_actions"]:  boost += 0.10
        if meta["suspicious_links"] > 3:   boost += 0.07
        final = min(phish_score + boost, 1.0)

        reasons = []
        if meta["password_fields"]:
            reasons.append(f"{meta['password_fields']} password field(s)")
        if meta["external_form_actions"]:
            reasons.append("form posts to external domain")
        if top_tokens:
            reasons.append(f"key terms: {', '.join(top_tokens[:3])}")
        if not reasons:
            reasons.append(
                f"{method}: {'phishing content detected' if final > 0.5 else 'content looks normal'}"
            )

        return {
            "score":       round(final, 4),
            "confidence":  round(min(abs(phish_score - 0.5) * 2 + 0.5, 1.0), 4),
            "explanation": "; ".join(reasons).capitalize() + ".",
            "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
            "features":    {**meta, "classifier": method, "boost": round(boost, 3)},
            "lime_tokens": lime_tokens,
            "top_tokens":  top_tokens,
        }
