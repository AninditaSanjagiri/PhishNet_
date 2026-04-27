"""
robustness/attacks/visual_attacks.py
======================================
Five visual perturbation attacks against the MobileNetV3 screenshot
classifier. These simulate pixel-level and layout-level manipulations
an attacker might apply to a phishing page screenshot.

Based on techniques from:
  - FGSM (Goodfellow et al. 2015) — gradient-based pixel perturbation
  - Phan The Duy et al. (2024) — practical visual evasion strategies

Attacks implemented:
  1. gaussian_noise        — add Gaussian pixel noise (σ-controlled)
  2. jpeg_compression      — degrade screenshot via JPEG artefacts
  3. brightness_shift      — shift brightness to confuse colour features
  4. pixel_block_mask      — blank out random 10×10 pixel blocks
  5. fgsm_approximation    — model-free signed gradient approximation
                             (finite-differences FGSM without true backprop)

All attacks operate on PIL.Image objects and return perturbed images.
"""
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Callable


# ── Attack implementations ────────────────────────────────────────────────────

def gaussian_noise(img: Image.Image, sigma: float = 15.0) -> tuple[Image.Image, str]:
    """Add Gaussian pixel noise with standard deviation sigma."""
    arr   = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy), f"Gaussian noise σ={sigma}"


def jpeg_compression(img: Image.Image, quality: int = 10) -> tuple[Image.Image, str]:
    """Re-encode as JPEG at low quality to introduce compression artefacts."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy(), f"JPEG compression quality={quality}"


def brightness_shift(img: Image.Image, factor: float = 1.8) -> tuple[Image.Image, str]:
    """Increase brightness to wash out visual features."""
    enhanced = ImageEnhance.Brightness(img).enhance(factor)
    return enhanced, f"Brightness ×{factor}"


def pixel_block_mask(img: Image.Image, n_blocks: int = 20,
                     block_size: int = 10) -> tuple[Image.Image, str]:
    """Blank out n random 10×10 pixel blocks with white patches."""
    arr    = np.array(img, dtype=np.uint8).copy()
    h, w   = arr.shape[:2]
    placed = 0
    for _ in range(n_blocks):
        y = random.randint(0, h - block_size)
        x = random.randint(0, w - block_size)
        arr[y:y+block_size, x:x+block_size] = 255
        placed += 1
    return Image.fromarray(arr), f"{placed} white blocks ({block_size}×{block_size}px)"


def fgsm_approximation(img: Image.Image,
                        epsilon: float = 8.0) -> tuple[Image.Image, str]:
    """
    Model-free FGSM approximation via finite differences.
    Perturbs each pixel in the direction that increases high-frequency
    content (a proxy for gradient sign without model access).
    This is the 'transfer attack' variant: perturb toward high-frequency
    adversarial noise regardless of the specific model weights.
    """
    arr    = np.array(img.convert("RGB"), dtype=np.float32)
    # Estimate local gradient sign via Laplacian of the image
    from PIL import ImageFilter
    grey   = np.array(img.convert("L"), dtype=np.float32)
    # Finite-difference Laplacian as gradient proxy
    pad    = np.pad(grey, 1, mode='edge')
    lap    = (pad[:-2, 1:-1] + pad[2:, 1:-1] +
              pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * grey)
    sign   = np.sign(lap).astype(np.float32)   # –1, 0, or +1
    # Apply perturbation to all channels
    for c in range(arr.shape[2]):
        arr[:, :, c] = np.clip(arr[:, :, c] + epsilon * sign, 0, 255)
    return Image.fromarray(arr.astype(np.uint8)), f"FGSM approx ε={epsilon}"


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_VISUAL_ATTACKS: list[Callable] = [
    gaussian_noise,
    jpeg_compression,
    brightness_shift,
    pixel_block_mask,
    fgsm_approximation,
]

VISUAL_ATTACK_NAMES = [f.__name__ for f in ALL_VISUAL_ATTACKS]


def apply_all_visual_attacks(img: Image.Image) -> list[dict]:
    """
    Apply every visual attack to a PIL image.
    Returns list of dicts with:
      attack_name, perturbed_img (PIL), description, pixel_delta_pct
    """
    orig_arr = np.array(img.convert("RGB"), dtype=np.float32)
    results  = []
    for fn in ALL_VISUAL_ATTACKS:
        try:
            perturbed, desc = fn(img.copy())
            pert_arr    = np.array(perturbed.convert("RGB"), dtype=np.float32)
            pixel_delta = float(np.mean(np.abs(pert_arr - orig_arr)))
            results.append({
                "attack_name":     fn.__name__,
                "perturbed_img":   perturbed,
                "description":     desc,
                "mean_pixel_delta": round(pixel_delta, 2),
            })
        except Exception as exc:
            results.append({
                "attack_name":    fn.__name__,
                "perturbed_img":  img,
                "description":    f"Attack failed: {exc}",
                "mean_pixel_delta": 0.0,
            })
    return results


def pil_to_base64(img: Image.Image) -> str:
    """Encode a PIL image to base64 PNG string for API transport."""
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def base64_to_pil(b64: str) -> Image.Image:
    """Decode a base64 PNG string to PIL image."""
    import base64
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")
