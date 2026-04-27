"""
robustness/attacks/url_attacks.py
===================================
Seven adversarial URL mutation strategies drawn from real attacker
playbooks documented in Phan The Duy et al. (IEEE Access 2024) and
the AWG paper. Each attack takes a phishing URL and produces a
perturbed version designed to evade the URL feature extractor.

Attacks implemented:
  1. homograph_swap      — replace ASCII letters with Unicode lookalikes
  2. subdomain_inject    — prepend legit-brand subdomain
  3. tld_substitution    — swap suspicious TLD for benign one
  4. path_noise          — append random benign path segments
  5. https_spoof         — force HTTPS scheme even on HTTP-only URL
  6. keyword_dilution    — dilute phishing keyword density with benign words
  7. entropy_reduction   — replace random chars to lower URL entropy

All attacks preserve the URL structure enough to remain plausible.
Each returns (perturbed_url, attack_name, description).
"""
import random
import re
import string
from urllib.parse import urlparse, urlunparse, quote

# Unicode homograph map (visually identical to ASCII)
HOMOGRAPHS: dict[str, str] = {
    'a': 'а',   # Cyrillic а
    'e': 'е',   # Cyrillic е
    'o': 'о',   # Cyrillic о
    'p': 'р',   # Cyrillic р
    'c': 'с',   # Cyrillic с
    'x': 'х',   # Cyrillic х
    'i': 'і',   # Ukrainian і
    'y': 'у',   # Cyrillic у
}

BENIGN_BRANDS   = ["google","microsoft","paypal","amazon","apple","secure","account"]
BENIGN_PATHS    = ["help","support","faq","login","home","index","about","contact"]
BENIGN_TLDS     = [".com", ".net", ".org", ".io"]
PHISH_KEYWORDS  = ["login","verify","secure","update","password","account","banking"]


# ── Individual attack functions ──────────────────────────────────────────────

def homograph_swap(url: str, n_chars: int = 3) -> tuple[str, str, str]:
    """Replace n ASCII letters with Unicode homoglyphs in the domain."""
    parsed  = urlparse(url)
    host    = parsed.hostname or ""
    swaps   = [(i, c) for i, c in enumerate(host) if c in HOMOGRAPHS]
    if not swaps:
        return url, "homograph_swap", "no swappable chars found (no-op)"
    chosen  = random.sample(swaps, min(n_chars, len(swaps)))
    chars   = list(host)
    for idx, ch in chosen:
        chars[idx] = HOMOGRAPHS[ch]
    new_host = "".join(chars)
    new_url  = urlunparse(parsed._replace(netloc=new_host))
    desc = f"Swapped {len(chosen)} char(s) to Unicode homoglyphs in '{host}'"
    return new_url, "homograph_swap", desc


def subdomain_inject(url: str) -> tuple[str, str, str]:
    """Prepend a trusted-brand subdomain: paypal.com.evil.tk → *.paypal.com.evil.tk"""
    parsed  = urlparse(url)
    host    = parsed.hostname or ""
    brand   = random.choice(BENIGN_BRANDS)
    new_host = f"{brand}-secure.{host}"
    new_url  = urlunparse(parsed._replace(netloc=new_host))
    desc = f"Injected '{brand}-secure.' subdomain before '{host}'"
    return new_url, "subdomain_inject", desc


def tld_substitution(url: str) -> tuple[str, str, str]:
    """Swap the TLD for a benign-looking one (.tk → .com)."""
    parsed = urlparse(url)
    host   = parsed.hostname or ""
    import tldextract
    ext = tldextract.extract(url)
    if not ext.suffix:
        return url, "tld_substitution", "no TLD found (no-op)"
    new_tld  = random.choice(BENIGN_TLDS)
    new_host = host.replace(f".{ext.suffix}", new_tld, 1)
    new_url  = urlunparse(parsed._replace(netloc=new_host))
    desc = f"Replaced .{ext.suffix} with {new_tld}"
    return new_url, "tld_substitution", desc


def path_noise(url: str, n_segments: int = 2) -> tuple[str, str, str]:
    """Append benign path segments to dilute phishing path signals."""
    parsed   = urlparse(url)
    extra    = "/" + "/".join(random.choices(BENIGN_PATHS, k=n_segments))
    new_path = parsed.path.rstrip("/") + extra
    new_url  = urlunparse(parsed._replace(path=new_path))
    desc = f"Appended '{extra}' to URL path"
    return new_url, "path_noise", desc


def https_spoof(url: str) -> tuple[str, str, str]:
    """Upgrade http:// to https:// — fools is_https feature."""
    if url.startswith("https://"):
        return url, "https_spoof", "already HTTPS (no-op)"
    new_url = "https://" + url[len("http://"):]
    return new_url, "https_spoof", "Upgraded scheme from http to https"


def keyword_dilution(url: str) -> tuple[str, str, str]:
    """Insert benign query params to dilute keyword density."""
    parsed  = urlparse(url)
    benign_params = "&".join(
        f"{random.choice(BENIGN_PATHS)}={random.randint(1,99)}"
        for _ in range(3)
    )
    sep      = "&" if parsed.query else ""
    new_q    = parsed.query + sep + benign_params
    new_url  = urlunparse(parsed._replace(query=new_q))
    desc = f"Appended benign query params: {benign_params[:40]}"
    return new_url, "keyword_dilution", desc


def entropy_reduction(url: str, n_replace: int = 4) -> tuple[str, str, str]:
    """Replace random path characters with lowercase letters to reduce entropy."""
    parsed = urlparse(url)
    path   = list(parsed.path)
    high_entropy_chars = [
        (i, c) for i, c in enumerate(path)
        if c in string.digits + string.punctuation and c not in "/.-_"
    ]
    if not high_entropy_chars:
        return url, "entropy_reduction", "no high-entropy chars found (no-op)"
    chosen  = random.sample(high_entropy_chars, min(n_replace, len(high_entropy_chars)))
    for idx, _ in chosen:
        path[idx] = random.choice(string.ascii_lowercase)
    new_path = "".join(path)
    new_url  = urlunparse(parsed._replace(path=new_path))
    desc = f"Replaced {len(chosen)} high-entropy path chars with letters"
    return new_url, "entropy_reduction", desc


# ── Attack registry ──────────────────────────────────────────────────────────

ALL_ATTACKS = [
    homograph_swap,
    subdomain_inject,
    tld_substitution,
    path_noise,
    https_spoof,
    keyword_dilution,
    entropy_reduction,
]

ATTACK_NAMES = [f.__name__ for f in ALL_ATTACKS]


def apply_all_attacks(url: str) -> list[dict]:
    """
    Apply every attack to a URL.
    Returns list of dicts with keys:
      attack_name, perturbed_url, description
    """
    results = []
    for fn in ALL_ATTACKS:
        try:
            perturbed, name, desc = fn(url)
            results.append({
                "attack_name":   name,
                "original_url":  url,
                "perturbed_url": perturbed,
                "description":   desc,
                "is_noop":       perturbed == url,
            })
        except Exception as exc:
            results.append({
                "attack_name":   fn.__name__,
                "original_url":  url,
                "perturbed_url": url,
                "description":   f"Attack failed: {exc}",
                "is_noop":       True,
            })
    return results


def apply_combined_attack(url: str) -> tuple[str, str]:
    """
    Apply 3 random attacks in sequence — simulates a sophisticated attacker.
    Returns (perturbed_url, combined_description).
    """
    chosen = random.sample(ALL_ATTACKS, min(3, len(ALL_ATTACKS)))
    desc_parts = []
    current = url
    for fn in chosen:
        try:
            current, _, desc = fn(current)
            desc_parts.append(desc)
        except Exception:
            pass
    return current, " → ".join(desc_parts)
