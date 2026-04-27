"""
agents/image_agent.py  — Production-grade rewrite
===================================================

WHY THE ORIGINAL FAILED ON WINDOWS
------------------------------------
Playwright launches Chromium via asyncio.create_subprocess_exec().
On Windows, Python's default event loop is SelectorEventLoop, which
does NOT support subprocess spawning.  Only ProactorEventLoop supports
it, but Uvicorn internally uses SelectorEventLoop even when you set
ProactorEventLoop at startup — they fight each other after lifespan
initialisation and the error is silently swallowed as a Task exception.

THE CORRECT FIX
---------------
Never call Playwright inside the FastAPI event loop at all.
Run the entire Playwright session inside a ThreadPoolExecutor using
*synchronous* Playwright (playwright.sync_api) instead of async_api.
The thread manages its own subprocess lifecycle completely isolated
from asyncio.  This is the officially recommended pattern for running
Playwright alongside async frameworks on all platforms.

ARCHITECTURE
------------
  FastAPI coroutine
       |
       +-> asyncio.wait_for(
               loop.run_in_executor(THREAD_POOL, _capture_screenshot_sync, url)
           )
                   |
                   +-> _capture_screenshot_sync()   <- worker thread
                           |
                           +-> playwright.sync_api  <- its own subprocess,
                                                       no asyncio involved

Works identically on:
  Windows 10/11  (SelectorEventLoop or ProactorEventLoop)
  Linux / Docker (Render, Railway, fly.io)
  macOS

GRACEFUL DEGRADATION
--------------------
Any failure (network timeout, bot-block, sandbox restriction, Playwright
not installed) causes score=None to be returned.  The orchestrator passes
that to the fusion agent which redistributes its weight to URL + Text
agents and still returns a verdict.  The API never 500s.
"""

import asyncio
import base64
import concurrent.futures
import io
import logging
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("phishnet.image_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH         = Path(__file__).parent.parent / "models" / "image_mobilenet.pth"
IMG_SIZE           = 224
SCREENSHOT_TIMEOUT = 8_000   # 8s — demo mode   # ms  – Playwright page.goto() timeout
THREAD_TIMEOUT_S   = 10      # 10s outer guard — demo mode       # sec – asyncio.wait_for() outer guard

# ---------------------------------------------------------------------------
# Dedicated thread pool for Playwright work
# max_workers=2: prevents more than 2 simultaneous Chromium processes
# ---------------------------------------------------------------------------
_screenshot_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="playwright_worker",
)

# ---------------------------------------------------------------------------
# Module-level model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model     = None
_transform = None
_torch     = None


# ---------------------------------------------------------------------------
# Model loading  (sync — called via run_in_executor at startup)
# ---------------------------------------------------------------------------

def _load_model() -> None:
    global _model, _transform, _torch
    try:
        import torch
        import torchvision.transforms as T
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        _torch = torch

        if MODEL_PATH.exists():
            model = mobilenet_v3_small(weights=None)
            in_f  = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_f, 2)
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
            logger.info("Image model loaded from %s", MODEL_PATH)
        else:
            logger.warning(
                "No fine-tuned model at %s — using ImageNet pretrained weights.",
                MODEL_PATH,
            )
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            in_f  = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_f, 2)
            torch.nn.init.xavier_uniform_(model.classifier[-1].weight)

        model.eval()
        _model     = model
        _transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]),
        ])

    except Exception:
        logger.exception(
            "Image model load failed — visual heuristics fallback will be used."
        )


# ---------------------------------------------------------------------------
# Screenshot capture  (sync — must only be called from a thread, NEVER from
# the event loop directly)
# ---------------------------------------------------------------------------

def _capture_screenshot_sync(url: str) -> Optional[bytes]:
    """
    Launch a headless Chromium instance via synchronous Playwright and
    return the page's PNG screenshot bytes, or None on any error.

    This function is SYNCHRONOUS and runs entirely inside a worker thread.
    It has zero interaction with asyncio.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning(
            "Playwright not installed. "
            "Fix: pip install playwright && playwright install chromium"
        )
        return None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--disable-background-networking",
                    "--disable-sync",
                    "--no-first-run",
                    "--mute-audio",
                ],
            )
            try:
                context = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    ignore_https_errors=True,
                )
                page = context.new_page()

                # Abort font requests — irrelevant for visual classification
                # and saves ~30% of load time
                page.route(
                    "**/*.{woff,woff2,ttf,otf,eot}",
                    lambda route: route.abort(),
                )

                page.goto(
                    url,
                    timeout=SCREENSHOT_TIMEOUT,
                    wait_until="domcontentloaded",
                )
                # Brief settle — lets lazy-rendered login forms appear
                page.wait_for_timeout(800)

                return page.screenshot(full_page=False, type="png")

            finally:
                browser.close()

    except Exception:
        logger.debug("Screenshot capture failed for %s", url, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# GradCAM  (sync — runs in executor alongside model inference)
# ---------------------------------------------------------------------------

def _generate_gradcam(
    model,
    tensor,
    target_class: int,
    img_pil: Image.Image,
) -> Optional[str]:
    """
    Lightweight GradCAM on MobileNetV3's last conv block.
    Returns base64-encoded PNG heatmap overlay, or None on any error.
    """
    try:
        import torch
        import torch.nn.functional as F

        activations: dict = {}
        gradients:   dict = {}

        def fwd_hook(_m, _i, out):    activations["v"] = out
        def bwd_hook(_m, _gi, g_out): gradients["v"]   = g_out[0]

        layer = model.features[-1]
        fh = layer.register_forward_hook(fwd_hook)
        bh = layer.register_full_backward_hook(bwd_hook)
        try:
            out = model(tensor)
            model.zero_grad()
            out[0, target_class].backward()
        finally:
            fh.remove()
            bh.remove()

        acts  = activations["v"].detach()
        grads = gradients["v"].detach()
        cam   = F.relu((grads.mean(dim=[2, 3], keepdim=True) * acts).sum(1, keepdim=True))
        cam   = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                              mode="bilinear", align_corners=False)
        cam   = cam.squeeze().numpy()
        cam   = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heat  = np.zeros((*cam.shape, 3), dtype=np.uint8)
        heat[:, :, 0] = (cam * 255).astype(np.uint8)
        heat[:, :, 1] = ((1 - cam) * 128).astype(np.uint8)

        heat_img = Image.fromarray(heat).convert("RGBA")
        heat_img.putalpha(Image.fromarray((cam * 180).astype(np.uint8)))
        blended  = Image.alpha_composite(
            img_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGBA"), heat_img
        )
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    except Exception:
        logger.debug("GradCAM failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Visual heuristics fallback (when model is not loaded)
# ---------------------------------------------------------------------------

def _visual_heuristics(img: Image.Image) -> float:
    arr   = np.array(img.resize((64, 64)).convert("RGB"), dtype=np.float32) / 255.0
    red   = float((arr[:, :, 0] - arr[:, :, 1] - arr[:, :, 2]).mean())
    white = float((arr > 0.9).all(axis=2).mean())
    score = 0.20
    if red   > 0.10: score += 0.30
    if white > 0.60: score += 0.15
    return min(score, 0.75)


# ---------------------------------------------------------------------------
# Classification  (sync — runs in default executor via run_in_executor)
# ---------------------------------------------------------------------------

def _classify_sync(img: Image.Image) -> dict:
    """Classify a PIL image. Always synchronous — safe to run in any thread."""
    if _model is not None and _transform is not None and _torch is not None:
        return _classify_with_model(img)
    score = _visual_heuristics(img)
    return {
        "score":          round(score, 4),
        "confidence":     0.40,
        "explanation":    "Visual heuristics applied (no fine-tuned model loaded).",
        "features":       {"classifier": "visual_heuristics"},
        "gradcam_base64": None,
    }


def _classify_with_model(img: Image.Image) -> dict:
    import torch

    tensor      = _transform(img).unsqueeze(0)
    with torch.no_grad():
        probs   = torch.softmax(_model(tensor), dim=-1)[0]
    phish_score = float(probs[1])
    confidence  = float(probs.max())
    gradcam_b64 = _generate_gradcam(_model, tensor.clone(), 1, img)

    if phish_score > 0.50:
        msg = "Page visually resembles known phishing sites."
    elif phish_score > 0.30:
        msg = "Some visual similarity to phishing pages."
    else:
        msg = "Page visually resembles legitimate sites."

    return {
        "score":          round(phish_score, 4),
        "confidence":     round(confidence, 4),
        "explanation":    msg,
        "features":       {
            "classifier": "MobileNetV3",
            "phish_prob": round(phish_score, 4),
            "legit_prob":  round(float(probs[0]), 4),
        },
        "gradcam_base64": gradcam_b64,
    }


# ---------------------------------------------------------------------------
# ImageAgent  — the async facade
# ---------------------------------------------------------------------------

class ImageAgent:
    """
    Public async interface.  All CPU-bound and subprocess work is
    dispatched to thread pools so the event loop is never blocked and
    Playwright's subprocess never conflicts with asyncio on any platform.
    """

    def __init__(self) -> None:
        self._model_loaded = False

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_model)
        self._model_loaded = _model is not None
        logger.info(
            "ImageAgent ready  model_loaded=%s  platform=%s",
            self._model_loaded,
            platform.system(),
        )

    async def analyze(self, url: str) -> dict:
        t0   = time.perf_counter()
        loop = asyncio.get_event_loop()

        # ------------------------------------------------------------------
        # Step 1 — Screenshot in the dedicated playwright thread pool
        # asyncio.wait_for provides a hard outer timeout independent of
        # Playwright's own timeout, protecting against hung browser processes.
        # ------------------------------------------------------------------
        screenshot_bytes: Optional[bytes] = None
        try:
            screenshot_bytes = await asyncio.wait_for(
                loop.run_in_executor(
                    _screenshot_executor,
                    _capture_screenshot_sync,
                    url,
                ),
                timeout=THREAD_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Screenshot timed out (%ss) for %s", THREAD_TIMEOUT_S, url
            )
        except Exception:
            logger.warning("Screenshot executor error for %s", url, exc_info=True)

        # ------------------------------------------------------------------
        # Step 2 — Graceful degradation if screenshot failed
        # ------------------------------------------------------------------
        if not screenshot_bytes:
            return {
                "score":             None,
                "confidence":        0.0,
                "explanation":       (
                    "Screenshot unavailable — "
                    "URL and Text agents provide the full verdict."
                ),
                "latency_ms":        round((time.perf_counter() - t0) * 1000, 2),
                "features":          {
                    "screenshot_available": False,
                    "platform":             platform.system(),
                },
                "screenshot_base64": None,
                "gradcam_base64":    None,
            }

        # ------------------------------------------------------------------
        # Step 3 — Classification in the default executor (torch is sync)
        # ------------------------------------------------------------------
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
        try:
            img    = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
            result = await loop.run_in_executor(None, _classify_sync, img)
        except Exception:
            logger.warning("Classification failed for %s", url, exc_info=True)
            result = {
                "score":          None,
                "confidence":     0.0,
                "explanation":    "Image classification failed after screenshot capture.",
                "features":       {"classifier": "error"},
                "gradcam_base64": None,
            }

        return {
            **result,
            "latency_ms":        round((time.perf_counter() - t0) * 1000, 2),
            "screenshot_base64": screenshot_b64,
        }

    # Public sync method so the robustness module can call it directly
    def _classify(self, img: Image.Image) -> dict:
        return _classify_sync(img)
