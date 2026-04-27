"""
robustness/run_robustness.py
==============================
Master robustness evaluation script for PhishNet.

For each test URL it:
  1. Scores with the clean (unperturbed) URL → baseline
  2. Applies all URL adversarial attacks → URL agent robustness
  3. Fetches page text, applies all text attacks → Text agent robustness
  4. Captures screenshot, applies all visual attacks → Image agent robustness
  5. Re-scores each perturbed version → evasion rate per attack

Outputs (all in evaluation/robustness/):
  robustness_summary.json         ← served at /evaluation/robustness
  url_evasion_table.md            ← paper Table 2
  text_evasion_table.md           ← paper Table 3
  visual_evasion_table.md         ← paper Table 4
  url_evasion_bar.png             ← Figure 3
  text_evasion_bar.png            ← Figure 4
  visual_evasion_bar.png          ← Figure 5
  delta_heatmap.png               ← Figure 6 (score drop heatmap)
  robustness_radar.png            ← Figure 7 (radar chart per agent)

Usage:
  python robustness/run_robustness.py --fast          # 20 URLs
  python robustness/run_robustness.py --n 100         # full run
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.url_agent    import URLAgent
from agents.text_agent   import TextAgent
from agents.image_agent  import ImageAgent
from robustness.attacks.url_attacks    import apply_all_attacks, ATTACK_NAMES
from robustness.attacks.text_attacks   import apply_all_text_attacks, TEXT_ATTACK_NAMES
from robustness.attacks.visual_attacks import apply_all_visual_attacks, VISUAL_ATTACK_NAMES

EVAL_DIR = Path(__file__).parent.parent / "evaluation" / "robustness"

# ── Sample phishing URLs for robustness testing ───────────────────────────────
DEMO_PHISH_URLS = [
    "http://paypa1-secure.login-verify.tk/account/update",
    "http://192.168.1.1/login?redirect=banking",
    "http://amazon-prize-winner.ml/claim/free-gift",
    "http://secure-microsoft-update.xyz/password/verify",
    "http://netflix-account-suspended.tk/reactivate",
    "http://apple-id-verify-urgent.cf/signin",
    "http://facebook-login-verify.ml/secure",
    "http://paypal-confirm-identity.ga/account",
    "http://ebay-prize-notification.tk/winner",
    "http://irs-refund-claim.xyz/verify/ssn",
    "http://bankofamerica-security-alert.ml/login",
    "http://google-account-suspended-alert.tk/recover",
    "http://dropbox-secure-share.xyz/document/verify",
    "http://microsoft-office365-password-reset.ml/update",
    "http://amazon-order-problem-verify.tk/help",
    "http://linkedin-account-review.xyz/confirm",
    "http://steam-account-hold-verify.ml/games",
    "http://discord-nitro-free-claim.tk/gift",
    "http://chase-bank-alert-urgent.xyz/signin",
    "http://wellsfargo-fraud-alert.ml/secure/login",
]


async def score_url(agent: URLAgent, url: str) -> float:
    try:
        r = await agent.analyze(url)
        return r.get("score") or 0.5
    except Exception:
        return 0.5


async def score_text(agent: TextAgent, url: str) -> tuple[float, str]:
    """Returns (score, fetched_text)."""
    try:
        html, _ = await agent._fetch_html(url)
        if not html:
            return 0.5, ""
        from robustness.attacks.text_attacks import apply_all_text_attacks
        from agents.text_agent import _extract_text_from_html
        text, _ = _extract_text_from_html(html)
        # Score on clean text using internal _classify
        result = agent._classify(html)
        return result.get("score") or 0.5, text
    except Exception:
        return 0.5, ""


async def score_text_direct(agent: TextAgent, text: str) -> float:
    """Score raw text directly (for perturbed versions)."""
    if not text.strip():
        return 0.5
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        # Wrap text in minimal HTML for _classify
        fake_html = f"<html><body><p>{text}</p></body></html>"
        result = await loop.run_in_executor(None, agent._classify, fake_html)
        return result.get("score") or 0.5
    except Exception:
        return 0.5


async def score_image_pil(agent: ImageAgent, img) -> float:
    """Score a PIL image directly."""
    if img is None or not agent._model_loaded:
        return 0.5
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent._classify, img)
        return result.get("score") or 0.5
    except Exception:
        return 0.5


def evasion_rate(baseline_scores: list[float],
                 perturbed_scores: list[float],
                 threshold: float = 0.5) -> float:
    """
    Fraction of samples that were correctly classified as phishing at baseline
    but are misclassified (score < threshold) after attack.
    """
    evaded = 0
    eligible = 0
    for b, p in zip(baseline_scores, perturbed_scores):
        if b >= threshold:          # correctly flagged at baseline
            eligible += 1
            if p < threshold:       # now below detection threshold
                evaded += 1
    return round(evaded / eligible, 4) if eligible else 0.0


def score_drop(baseline: list[float], perturbed: list[float]) -> float:
    """Mean absolute drop in phishing score caused by the attack."""
    drops = [max(0.0, b - p) for b, p in zip(baseline, perturbed)]
    return round(float(np.mean(drops)), 4)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def bar_chart(names: list[str], evasion_rates: list[float],
              score_drops: list[float], title: str, filename: str):
    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.1), 5))
    b1 = ax.bar(x - w/2, evasion_rates, w, label="Evasion rate",
                color="#ff3860", alpha=0.85)
    b2 = ax.bar(x + w/2, score_drops,   w, label="Mean score drop",
                color="#a78bfa", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate / Score Drop")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / filename, dpi=150)
    plt.close()


def delta_heatmap(attack_names: list[str], url_names: list[str],
                  delta_matrix: np.ndarray, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(max(8, len(attack_names)*1.2),
                                    max(4, len(url_names)*0.4)))
    sns.heatmap(delta_matrix, ax=ax,
                xticklabels=attack_names,
                yticklabels=[u[:30] for u in url_names],
                cmap="RdYlGn_r", vmin=0, vmax=0.6,
                annot=len(url_names) <= 15,
                fmt=".2f", linewidths=0.4, linecolor="#1e2a35")
    ax.set_title(title)
    ax.set_xlabel("Attack")
    ax.set_ylabel("URL")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / filename, dpi=150)
    plt.close()


def radar_chart(agent_names: list[str],
                clean_accs: list[float],
                attacked_accs: list[float],
                filename: str):
    """Radar showing clean vs post-attack accuracy per agent."""
    N      = len(agent_names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    clean_vals    = clean_accs + clean_accs[:1]
    attacked_vals = attacked_accs + attacked_accs[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, clean_vals,    'o-', lw=2, color='#00e5ff',  label='Clean')
    ax.fill(angles, clean_vals,          color='#00e5ff', alpha=0.15)
    ax.plot(angles, attacked_vals, 's-', lw=2, color='#ff3860',  label='Under attack')
    ax.fill(angles, attacked_vals,       color='#ff3860', alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(agent_names, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], size=8)
    ax.set_title("Agent Accuracy: Clean vs Worst-Case Attack", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(EVAL_DIR / filename, dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

async def run(args):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    test_urls = DEMO_PHISH_URLS[:args.n]
    print(f"Robustness benchmark — {len(test_urls)} phishing URLs\n")

    # Load agents
    url_agent   = URLAgent()
    text_agent  = TextAgent()
    image_agent = ImageAgent()
    print("Loading agents…")
    await url_agent.load()
    await text_agent.load()
    await image_agent.load()

    # ── 1. URL ROBUSTNESS ────────────────────────────────────────────────
    print("\n[1/3] URL Agent robustness…")
    url_baselines: list[float]                  = []
    url_attack_scores: dict[str, list[float]]   = {n: [] for n in ATTACK_NAMES}
    url_delta_rows: list[list[float]]           = []

    for url in test_urls:
        base = await score_url(url_agent, url)
        url_baselines.append(base)
        attacked = apply_all_attacks(url)
        row = []
        for atk in attacked:
            s = await score_url(url_agent, atk["perturbed_url"])
            url_attack_scores[atk["attack_name"]].append(s)
            row.append(max(0.0, base - s))
        url_delta_rows.append(row)

    url_evasion = {
        n: evasion_rate(url_baselines, url_attack_scores[n])
        for n in ATTACK_NAMES
    }
    url_drops = {
        n: score_drop(url_baselines, url_attack_scores[n])
        for n in ATTACK_NAMES
    }
    url_mean_evasion = round(float(np.mean(list(url_evasion.values()))), 4)

    print(f"  URL baseline mean score: {np.mean(url_baselines):.3f}")
    print(f"  Mean evasion rate: {url_mean_evasion:.3f}")
    for n in ATTACK_NAMES:
        print(f"    {n:<25} evasion={url_evasion[n]:.3f}  drop={url_drops[n]:.3f}")

    bar_chart(ATTACK_NAMES,
              [url_evasion[n] for n in ATTACK_NAMES],
              [url_drops[n]   for n in ATTACK_NAMES],
              "URL Agent — Attack Evasion Rates", "url_evasion_bar.png")
    delta_heatmap(ATTACK_NAMES, test_urls,
                  np.array(url_delta_rows),
                  "URL Agent — Score Drop Heatmap", "url_delta_heatmap.png")

    # ── 2. TEXT ROBUSTNESS ───────────────────────────────────────────────
    print("\n[2/3] Text Agent robustness…")
    text_baselines: list[float]                  = []
    text_attack_scores: dict[str, list[float]]   = {n: [] for n in TEXT_ATTACK_NAMES}
    text_delta_rows: list[list[float]]           = []

    for url in test_urls:
        base_score, raw_text = await score_text(text_agent, url)
        text_baselines.append(base_score)

        if not raw_text.strip():
            # Can't fetch — use dummy phishing text
            raw_text = (
                "Your account has been suspended. Verify your password immediately. "
                "Click here to confirm your identity and avoid account closure."
            )
            base_score = await score_text_direct(text_agent, raw_text)
            text_baselines[-1] = base_score

        attacked = apply_all_text_attacks(raw_text)
        row = []
        for atk in attacked:
            s = await score_text_direct(text_agent, atk["perturbed_text"])
            text_attack_scores[atk["attack_name"]].append(s)
            row.append(max(0.0, base_score - s))
        text_delta_rows.append(row)

    text_evasion = {
        n: evasion_rate(text_baselines, text_attack_scores[n])
        for n in TEXT_ATTACK_NAMES
    }
    text_drops = {
        n: score_drop(text_baselines, text_attack_scores[n])
        for n in TEXT_ATTACK_NAMES
    }
    text_mean_evasion = round(float(np.mean(list(text_evasion.values()))), 4)

    print(f"  Text baseline mean score: {np.mean(text_baselines):.3f}")
    print(f"  Mean evasion rate: {text_mean_evasion:.3f}")
    for n in TEXT_ATTACK_NAMES:
        print(f"    {n:<25} evasion={text_evasion[n]:.3f}  drop={text_drops[n]:.3f}")

    bar_chart(TEXT_ATTACK_NAMES,
              [text_evasion[n] for n in TEXT_ATTACK_NAMES],
              [text_drops[n]   for n in TEXT_ATTACK_NAMES],
              "Text Agent — Attack Evasion Rates", "text_evasion_bar.png")
    delta_heatmap(TEXT_ATTACK_NAMES, test_urls,
                  np.array(text_delta_rows),
                  "Text Agent — Score Drop Heatmap", "text_delta_heatmap.png")

    # ── 3. VISUAL ROBUSTNESS ─────────────────────────────────────────────
    print("\n[3/3] Image Agent robustness…")
    visual_baselines: list[float]                  = []
    visual_attack_scores: dict[str, list[float]]   = {n: [] for n in VISUAL_ATTACK_NAMES}
    visual_delta_rows: list[list[float]]           = []

    from robustness.attacks.visual_attacks import apply_all_visual_attacks
    from PIL import Image as PILImage
    import io, base64

    for url in test_urls:
        # Try to capture screenshot
        screenshot_bytes = None
        try:
            from agents.image_agent import _take_screenshot
            screenshot_bytes = await _take_screenshot(url)
        except Exception:
            pass

        if screenshot_bytes:
            img = PILImage.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        else:
            # Synthesise a white 224×224 test image (worst case)
            arr = np.ones((224, 224, 3), dtype=np.uint8) * 240
            img = PILImage.fromarray(arr)

        base_s = await score_image_pil(image_agent, img)
        visual_baselines.append(base_s)

        attacked = apply_all_visual_attacks(img)
        row = []
        for atk in attacked:
            s = await score_image_pil(image_agent, atk["perturbed_img"])
            visual_attack_scores[atk["attack_name"]].append(s)
            row.append(max(0.0, base_s - s))
        visual_delta_rows.append(row)

    visual_evasion = {
        n: evasion_rate(visual_baselines, visual_attack_scores[n])
        for n in VISUAL_ATTACK_NAMES
    }
    visual_drops = {
        n: score_drop(visual_baselines, visual_attack_scores[n])
        for n in VISUAL_ATTACK_NAMES
    }
    visual_mean_evasion = round(float(np.mean(list(visual_evasion.values()))), 4)

    print(f"  Visual baseline mean score: {np.mean(visual_baselines):.3f}")
    print(f"  Mean evasion rate: {visual_mean_evasion:.3f}")
    for n in VISUAL_ATTACK_NAMES:
        print(f"    {n:<25} evasion={visual_evasion[n]:.3f}  drop={visual_drops[n]:.3f}")

    bar_chart(VISUAL_ATTACK_NAMES,
              [visual_evasion[n] for n in VISUAL_ATTACK_NAMES],
              [visual_drops[n]   for n in VISUAL_ATTACK_NAMES],
              "Image Agent — Visual Attack Evasion Rates", "visual_evasion_bar.png")
    delta_heatmap(VISUAL_ATTACK_NAMES, test_urls,
                  np.array(visual_delta_rows),
                  "Image Agent — Score Drop Heatmap", "visual_delta_heatmap.png")

    # ── RADAR CHART ──────────────────────────────────────────────────────
    def detection_acc(scores, threshold=0.5):
        return round(sum(1 for s in scores if s >= threshold) / max(len(scores),1), 4)

    clean_accs   = [
        detection_acc(url_baselines),
        detection_acc(text_baselines),
        detection_acc(visual_baselines),
    ]
    # Worst-case attacked accuracy per agent
    worst_url   = min(detection_acc(url_attack_scores[n])   for n in ATTACK_NAMES)
    worst_text  = min(detection_acc(text_attack_scores[n])  for n in TEXT_ATTACK_NAMES)
    worst_vis   = min(detection_acc(visual_attack_scores[n]) for n in VISUAL_ATTACK_NAMES)
    attacked_accs = [worst_url, worst_text, worst_vis]

    radar_chart(["URL Agent","Text Agent","Image Agent"],
                clean_accs, attacked_accs, "robustness_radar.png")

    # ── COMBINED FUSION ROBUSTNESS ────────────────────────────────────────
    # Fuse perturbed scores to see if fusion is more robust than any single agent
    from agents.fusion_agent import FusionAgent
    fusion_agent = FusionAgent()
    await fusion_agent.load()

    fused_baselines, fused_combined = [], []
    for i in range(len(test_urls)):
        fu = fusion_agent.fuse(
            {"score": url_baselines[i]},
            {"score": text_baselines[i]},
            {"score": visual_baselines[i]},
        )
        fused_baselines.append(fu["phishing_probability"] / 100.0)

        # Apply combined attacks
        from robustness.attacks.url_attacks  import apply_combined_attack
        from robustness.attacks.text_attacks import apply_combined_text_attack
        pu, _  = apply_combined_attack(test_urls[i])
        su     = await score_url(url_agent, pu)
        # Combined text attack on demo text
        demo_text = "Verify your account password immediately or it will be suspended."
        pt, _  = apply_combined_text_attack(demo_text)
        st     = await score_text_direct(text_agent, pt)

        ff = fusion_agent.fuse(
            {"score": su},
            {"score": st},
            {"score": visual_baselines[i]},
        )
        fused_combined.append(ff["phishing_probability"] / 100.0)

    fusion_evasion = evasion_rate(fused_baselines, fused_combined)
    fusion_drop    = score_drop(fused_baselines, fused_combined)

    print(f"\n  Fused model — combined attack evasion: {fusion_evasion:.3f}  drop: {fusion_drop:.3f}")

    # ── MARKDOWN TABLES ───────────────────────────────────────────────────
    def md_table(title, names, evasion, drops):
        rows = sorted(zip(names, evasion.values(), drops.values()),
                      key=lambda x: -x[1])
        t  = f"## {title}\n\n"
        t += "| Attack | Evasion Rate | Mean Score Drop |\n"
        t += "|--------|-------------|----------------|\n"
        for n, e, d in rows:
            t += f"| `{n}` | {e:.4f} | {d:.4f} |\n"
        t += f"\n*Mean evasion rate: {np.mean(list(evasion.values())):.4f}*\n"
        return t

    url_md   = md_table("URL Agent — Adversarial Attack Robustness",
                        ATTACK_NAMES, url_evasion, url_drops)
    text_md  = md_table("Text Agent — Adversarial Text Attack Robustness",
                        TEXT_ATTACK_NAMES, text_evasion, text_drops)
    visual_md = md_table("Image Agent — Visual Perturbation Robustness",
                         VISUAL_ATTACK_NAMES, visual_evasion, visual_drops)

    (EVAL_DIR / "url_evasion_table.md").write_text(url_md)
    (EVAL_DIR / "text_evasion_table.md").write_text(text_md)
    (EVAL_DIR / "visual_evasion_table.md").write_text(visual_md)

    # ── SUMMARY JSON ──────────────────────────────────────────────────────
    summary = {
        "n_urls": len(test_urls),
        "url_agent": {
            "baseline_mean_score": round(float(np.mean(url_baselines)), 4),
            "clean_detection_acc": detection_acc(url_baselines),
            "mean_evasion_rate":   url_mean_evasion,
            "worst_attack":        max(url_evasion, key=url_evasion.get),
            "worst_evasion_rate":  max(url_evasion.values()),
            "per_attack_evasion":  url_evasion,
            "per_attack_drop":     url_drops,
        },
        "text_agent": {
            "baseline_mean_score": round(float(np.mean(text_baselines)), 4),
            "clean_detection_acc": detection_acc(text_baselines),
            "mean_evasion_rate":   text_mean_evasion,
            "worst_attack":        max(text_evasion, key=text_evasion.get),
            "worst_evasion_rate":  max(text_evasion.values()),
            "per_attack_evasion":  text_evasion,
            "per_attack_drop":     text_drops,
        },
        "image_agent": {
            "baseline_mean_score": round(float(np.mean(visual_baselines)), 4),
            "clean_detection_acc": detection_acc(visual_baselines),
            "mean_evasion_rate":   visual_mean_evasion,
            "worst_attack":        max(visual_evasion, key=visual_evasion.get),
            "worst_evasion_rate":  max(visual_evasion.values()),
            "per_attack_evasion":  visual_evasion,
            "per_attack_drop":     visual_drops,
        },
        "fusion": {
            "combined_attack_evasion_rate": fusion_evasion,
            "combined_attack_score_drop":   fusion_drop,
            "robustness_gain_vs_worst_agent":
                round(max(url_mean_evasion, text_mean_evasion, visual_mean_evasion)
                      - fusion_evasion, 4),
        },
        "outputs": [
            "url_evasion_bar.png", "text_evasion_bar.png", "visual_evasion_bar.png",
            "url_delta_heatmap.png", "text_delta_heatmap.png", "visual_delta_heatmap.png",
            "robustness_radar.png",
            "url_evasion_table.md", "text_evasion_table.md", "visual_evasion_table.md",
        ],
    }
    with open(EVAL_DIR / "robustness_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Robustness benchmark complete.")
    print(f"   All outputs → evaluation/robustness/")
    print(f"\nKey results:")
    print(f"  URL   agent — mean evasion: {url_mean_evasion:.3f}  "
          f"worst: {summary['url_agent']['worst_attack']} "
          f"({summary['url_agent']['worst_evasion_rate']:.3f})")
    print(f"  Text  agent — mean evasion: {text_mean_evasion:.3f}  "
          f"worst: {summary['text_agent']['worst_attack']}")
    print(f"  Image agent — mean evasion: {visual_mean_evasion:.3f}  "
          f"worst: {summary['image_agent']['worst_attack']}")
    print(f"  Fused model — combined evasion: {fusion_evasion:.3f}  "
          f"(robustness gain: {summary['fusion']['robustness_gain_vs_worst_agent']:+.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick run: 10 URLs")
    parser.add_argument("--n",    type=int, default=20,
                        help="Number of phishing URLs to test (default: 20)")
    args = parser.parse_args()
    if args.fast:
        args.n = 10
    asyncio.run(run(args))
