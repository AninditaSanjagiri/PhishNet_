"""
agents/orchestrator.py  — production-grade
===========================================
Parallel async dispatch via asyncio.gather().
All agents run concurrently; individual failures are caught and
returned as graceful-degradation payloads so the API never 500s.
"""
import asyncio
import logging
import os
import time

import psutil

from agents.url_agent    import URLAgent
from agents.text_agent   import TextAgent
from agents.image_agent  import ImageAgent
from agents.fusion_agent import FusionAgent

logger = logging.getLogger("phishnet.orchestrator")


class OrchestratorAgent:

    def __init__(self) -> None:
        self.url_agent    = URLAgent()
        self.text_agent   = TextAgent()
        self.image_agent  = ImageAgent()
        self.fusion_agent = FusionAgent()

    async def initialize(self) -> None:
        logger.info("Loading URL agent (RF + SHAP)…")
        await self.url_agent.load()

        logger.info("Loading Text agent (DistilBERT + LIME)…")
        await self.text_agent.load()

        logger.info("Loading Image agent (MobileNetV3 + sync Playwright)…")
        await self.image_agent.load()

        logger.info("Loading Fusion agent…")
        await self.fusion_agent.load()

    async def analyze(self, url: str) -> dict:
        t_wall     = time.perf_counter()
        proc       = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / 1024 / 1024   # MB

        url_r, text_r, image_r = await asyncio.gather(
            self._safe_run(self.url_agent,   url),
            self._safe_run(self.text_agent,  url),
            self._safe_run(self.image_agent, url),
        )

        fusion   = self.fusion_agent.fuse(url_r, text_r, image_r)
        wall_ms  = round((time.perf_counter() - t_wall) * 1000, 1)
        mem_diff = round(proc.memory_info().rss / 1024 / 1024 - mem_before, 2)

        logger.info(
            "Analysis complete  url=%.4f  text=%.4f  image=%s  "
            "verdict=%s  wall=%.0fms",
            url_r.get("score") or 0,
            text_r.get("score") or 0,
            f"{image_r.get('score'):.4f}" if image_r.get("score") is not None else "N/A",
            fusion.get("verdict"),
            wall_ms,
        )

        return {
            **fusion,
            "url_agent":         url_r,
            "text_agent":        text_r,
            "image_agent":       image_r,
            "screenshot_base64": image_r.get("screenshot_base64"),
            "gradcam_base64":    image_r.get("gradcam_base64"),
            # XAI outputs
            "shap_values":       url_r.get("shap_values", []),
            "top_shap":          url_r.get("top_shap", {}),
            "lime_tokens":       text_r.get("lime_tokens", []),
            "top_tokens":        text_r.get("top_tokens", []),
            # Performance
            "latency_breakdown": {
                "url_ms":         url_r.get("latency_ms",   0),
                "text_ms":        text_r.get("latency_ms",  0),
                "image_ms":       image_r.get("latency_ms", 0),
                "total_wall_ms":  wall_ms,
            },
            "memory_delta_mb": mem_diff,
        }

    async def _safe_run(self, agent, url: str) -> dict:
        name = type(agent).__name__
        try:
            return await agent.analyze(url)
        except Exception:
            logger.warning("%s failed for %s", name, url, exc_info=True)
            return {
                "score":       None,
                "confidence":  0.0,
                "explanation": f"{name} unavailable.",
                "latency_ms":  0.0,
                "features":    {},
                "shap_values": [],
                "top_shap":    {},
                "lime_tokens": [],
                "top_tokens":  [],
            }
