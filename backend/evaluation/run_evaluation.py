"""
evaluation/run_evaluation.py
==============================
PhishNet evaluation suite.

MODES
-----
Benchmark (fast, for demos and CI):
  python evaluation/run_evaluation.py --fast

  • 20 phishing + 20 legit URLs (40 total, no network data needed)
  • Image/screenshot branch disabled entirely
  • SHAP and LIME disabled (feature extraction only)
  • URL agent + Text agent + Fusion evaluated
  • Runs in ~30–60 seconds depending on CPU
  • Outputs: evaluation/latest_metrics.json, ablation_table.md,
             confusion_url.png, confusion_text.png, confusion_fused.png

Full evaluation (for paper/report):
  python evaluation/run_evaluation.py --n-samples 200
  python evaluation/run_evaluation.py \\
      --phishtank data/phishtank.csv \\
      --tranco    data/tranco.csv    \\
      --n-samples 1000
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.url_agent    import URLAgent, extract_features
from agents.text_agent   import TextAgent
from agents.fusion_agent import FusionAgent

EVAL_DIR = Path(__file__).parent.parent / "evaluation"

# ── Built-in benchmark URL lists (no network required) ───────────────────────
# These are used when --fast is set or no data files are provided.
# 20 clearly phishing URLs + 20 clearly legitimate domains.

BENCHMARK_PHISH_URLS = [
    "http://paypal-login-security-update.com/verify",
    "http://secure-paypal-login-update.net/account",
    "http://verify-amazon-account-login.xyz/confirm",
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
    "http://microsoft-office365-password-reset.ml/update",
    "http://amazon-order-problem-verify.tk/help",
    "http://login-paypal-secure.update-billing.com/auth",
    "http://account-verify.amazon-support-login.xyz",
    "http://192.168.1.1/login?redirect=banking&verify=account",
    "http://signin-microsoft.secure-update.ml/credentials",
    "http://apple-icloud-verify.account-hold.tk/confirm",
]

BENCHMARK_LEGIT_DOMAINS = [
    "google.com", "github.com", "microsoft.com", "amazon.com",
    "wikipedia.org", "stackoverflow.com", "python.org", "mozilla.org",
    "cloudflare.com", "anthropic.com", "openai.com", "linkedin.com",
    "apple.com", "bbc.co.uk", "reddit.com", "youtube.com",
    "twitter.com", "fastapi.tiangolo.com", "pytorch.org", "numpy.org",
]


# ── Data loaders ──────────────────────────────────────────────────────────────

def build_benchmark_dataset() -> tuple[list[str], list[int]]:
    """Return the built-in 40-URL benchmark set. No network needed."""
    urls   = list(BENCHMARK_PHISH_URLS)
    labels = [1] * len(urls)
    for domain in BENCHMARK_LEGIT_DOMAINS:
        urls.append(f"https://{domain}")
        labels.append(0)
    return urls, labels


def load_from_files(
    phishtank_path: str, tranco_path: str, n: int
) -> tuple[list[str], list[int]]:
    urls, labels = [], []
    half = n // 2
    if phishtank_path:
        with open(phishtank_path, encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or row.get("URL") or "").strip()
                if url and sum(1 for l in labels if l == 1) < half:
                    urls.append(url); labels.append(1)
    if tranco_path:
        with open(tranco_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts  = line.strip().split(",")
                domain = parts[-1].strip() if parts else ""
                if domain and sum(1 for l in labels if l == 0) < half:
                    urls.append(f"https://{domain}"); labels.append(0)
    return urls, labels


async def fetch_phishtank(n: int) -> tuple[list[str], list[int]]:
    """Download live PhishTank feed + use static legit list."""
    import requests
    urls, labels = [], []
    half = n // 2
    try:
        r = requests.get(
            "https://data.phishtank.com/data/online-valid.csv",
            timeout=20,
            headers={"User-Agent": "PhishNet-Eval/3.0"},
        )
        r.raise_for_status()
        for row in csv.DictReader(r.text.splitlines()):
            u = row.get("url", "").strip()
            if u and len(urls) < half:
                urls.append(u); labels.append(1)
        print(f"  PhishTank: {len(urls)} phishing URLs")
    except Exception as exc:
        print(f"  ⚠️  PhishTank download failed: {exc}")
        print("  Falling back to built-in phishing list")
        urls   = list(BENCHMARK_PHISH_URLS[:half])
        labels = [1] * len(urls)

    legit_domains = BENCHMARK_LEGIT_DOMAINS * 10   # cycle if needed
    for domain in legit_domains:
        if sum(1 for l in labels if l == 0) >= half:
            break
        urls.append(f"https://{domain}"); labels.append(0)
    return urls, labels


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list, y_pred: list, y_proba: list, label: str
) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = 0.0
    return {
        "system":    label,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(float(p), 4),
        "recall":    round(float(r), 4),
        "f1":        round(float(f), 4),
        "roc_auc":   round(float(auc), 4),
    }


def save_confusion(
    y_true: list, y_pred: list, title: str, filename: str
) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Phishing"],
        yticklabels=["Legit", "Phishing"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / filename, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    is_benchmark = args.fast

    # ── Build test set ───────────────────────────────────────────────────
    if is_benchmark:
        print("⚡ BENCHMARK MODE — 40 built-in URLs, no network required")
        urls, labels = build_benchmark_dataset()
    elif args.phishtank or args.tranco:
        print("Loading from local files…")
        urls, labels = load_from_files(
            args.phishtank, args.tranco, args.n_samples
        )
    else:
        print(f"Fetching {args.n_samples} URLs…")
        urls, labels = await fetch_phishtank(args.n_samples)

    if len(urls) < 4:
        print("❌ Not enough URLs to evaluate."); return

    n_phish = sum(labels)
    n_legit = len(labels) - n_phish
    print(f"Dataset: {len(urls)} URLs  ({n_phish} phishing, {n_legit} legit)\n")

    # ── Load agents ──────────────────────────────────────────────────────
    # In benchmark mode, disable SHAP and LIME by monkeypatching before load
    url_agent    = URLAgent()
    text_agent   = TextAgent()
    fusion_agent = FusionAgent()

    if is_benchmark:
        # Disable SHAP: prevent TreeExplainer from being built
        url_agent._get_explainer = lambda: None
        # Disable LIME: prevent LimeTextExplainer from being built
        import agents.text_agent as _ta_module
        _ta_module._lime_explainer = None
        _build_orig = _ta_module._build_lime_explainer
        _ta_module._build_lime_explainer = lambda: None   # no-op during load

    print("Loading agents…")
    t_load = time.perf_counter()
    await url_agent.load()
    await text_agent.load()
    await fusion_agent.load()
    print(f"  Agents loaded in {time.perf_counter() - t_load:.1f}s\n")

    # ── Prediction loop ──────────────────────────────────────────────────
    url_preds,   url_probas   = [], []
    text_preds,  text_probas  = [], []
    fused_preds, fused_probas = [], []
    url_lats,    text_lats    = [], []

    t_eval = time.perf_counter()

    for i, (url, label) in enumerate(zip(urls, labels)):
        if (i % 10 == 0) or (i == len(urls) - 1):
            elapsed = time.perf_counter() - t_eval
            print(f"  [{i+1:3d}/{len(urls)}]  {url[:52]:<52}  ({elapsed:.1f}s elapsed)")

        # URL agent (always fast — pure heuristic + RF)
        t0 = time.perf_counter()
        try:
            ur = await url_agent.analyze(url)
            us = ur.get("score")
            us = us if us is not None else 0.5
        except Exception as exc:
            print(f"    ⚠️  URL agent error: {exc}")
            us = 0.5
        url_lats.append((time.perf_counter() - t0) * 1000)
        url_preds.append(1 if us >= 0.5 else 0)
        url_probas.append(float(us))

        # Text agent (skips in benchmark: fetch disabled via 5s timeout,
        # LIME disabled; still evaluates DistilBERT content score)
        t0 = time.perf_counter()
        try:
            tr = await text_agent.analyze(url)
            ts = tr.get("score")
            ts = ts if ts is not None else 0.5
        except Exception as exc:
            print(f"    ⚠️  Text agent error: {exc}")
            ts = 0.5
        text_lats.append((time.perf_counter() - t0) * 1000)
        text_preds.append(1 if ts >= 0.5 else 0)
        text_probas.append(float(ts))

        # Fusion (no image in benchmark — image score = None)
        fres = fusion_agent.fuse(
            {"score": us, "features": ur.get("features", {}) if "ur" in dir() else {}},
            {"score": ts},
            {"score": None},
        )
        fp = fres["phishing_probability"] / 100.0
        fused_probas.append(fp)
        fused_preds.append(1 if fp >= 0.5 else 0)

    total_eval_s = time.perf_counter() - t_eval

    # ── Compute metrics ──────────────────────────────────────────────────
    url_m   = compute_metrics(labels, url_preds,   url_probas,   "URL only")
    text_m  = compute_metrics(labels, text_preds,  text_probas,  "Text only")
    fused_m = compute_metrics(labels, fused_preds, fused_probas, "Fused (URL+Text)")

    url_m["avg_latency_ms"]   = round(float(np.mean(url_lats)), 2)
    text_m["avg_latency_ms"]  = round(float(np.mean(text_lats)), 2)
    fused_m["avg_latency_ms"] = round(float(np.mean(url_lats) + np.mean(text_lats)), 2)

    all_metrics = [url_m, text_m, fused_m]

    # ── Print ablation table ─────────────────────────────────────────────
    W = 70
    print(f"\n{'='*W}")
    print(f"{'System':<24} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'AUC':>7} {'Lat(ms)':>9}")
    print(f"{'-'*W}")
    for m in all_metrics:
        print(
            f"{m['system']:<24} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>7.4f} "
            f"{m['avg_latency_ms']:>9.1f}"
        )
    print(f"{'='*W}")

    best_single_f1 = max(url_m["f1"], text_m["f1"])
    gain = fused_m["f1"] - best_single_f1
    print(f"\nFusion F1 gain over best single agent: {gain:+.4f}")
    print(f"Total evaluation time: {total_eval_s:.1f}s")

    # ── Save outputs ─────────────────────────────────────────────────────
    summary = {
        "n_test":          len(urls),
        "n_phishing":      n_phish,
        "n_legit":         n_legit,
        "benchmark_mode":  is_benchmark,
        "url_only":        url_m,
        "text_only":       text_m,
        "fused":           fused_m,
        "fusion_gain_f1":  round(gain, 4),
        "dominant_agent":  "url" if url_m["f1"] > text_m["f1"] else "text",
        "eval_time_s":     round(total_eval_s, 1),
    }
    (EVAL_DIR / "latest_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Ablation markdown
    md  = f"## PhishNet Evaluation {'(Benchmark Mode)' if is_benchmark else ''}\n\n"
    md += f"n={len(urls)} · {n_phish} phishing · {n_legit} legit\n\n"
    md += "| System | Accuracy | Precision | Recall | F1 | ROC-AUC | Latency |\n"
    md += "|--------|----------|-----------|--------|----|---------|----------|\n"
    for m in all_metrics:
        md += (f"| {m['system']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
               f"{m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | "
               f"{m['avg_latency_ms']:.1f}ms |\n")
    md += f"\n*Fusion F1 gain: {gain:+.4f}*\n"
    (EVAL_DIR / "ablation_table.md").write_text(md)
    (EVAL_DIR / "ablation_table.json").write_text(json.dumps(all_metrics, indent=2))

    # Confusion matrices
    save_confusion(labels, url_preds,   "URL Agent",    "confusion_url.png")
    save_confusion(labels, text_preds,  "Text Agent",   "confusion_text.png")
    save_confusion(labels, fused_preds, "Fused Model",  "confusion_fused.png")

    print(f"\n✅ Outputs saved to evaluation/")
    print(f"   latest_metrics.json  ablation_table.md  ablation_table.json")
    print(f"   confusion_url.png    confusion_text.png   confusion_fused.png")

    if not is_benchmark:
        # ROC curve (skip in benchmark — not worth the extra matplotlib time)
        from sklearn.metrics import RocCurveDisplay
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(
            labels, url_probas,   ax=ax, name=f"URL  (AUC={url_m['roc_auc']:.3f})")
        RocCurveDisplay.from_predictions(
            labels, text_probas,  ax=ax, name=f"Text (AUC={text_m['roc_auc']:.3f})")
        RocCurveDisplay.from_predictions(
            labels, fused_probas, ax=ax, name=f"Fused (AUC={fused_m['roc_auc']:.3f})")
        ax.set_title("PhishNet — ROC Curves")
        fig.tight_layout()
        fig.savefig(EVAL_DIR / "roc_comparison.png", dpi=150)
        plt.close()
        print("   roc_comparison.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhishNet Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast benchmark (40 built-in URLs, ~30-60s, no network needed):
  python evaluation/run_evaluation.py --fast

  # Standard run (downloads ~200 URLs from PhishTank):
  python evaluation/run_evaluation.py --n-samples 200

  # Full run with local data files:
  python evaluation/run_evaluation.py \\
      --phishtank data/phishtank.csv \\
      --tranco    data/tranco.csv    \\
      --n-samples 1000
""",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Benchmark mode: 40 built-in URLs, no SHAP/LIME, no screenshots (~30-60s)",
    )
    parser.add_argument(
        "--phishtank",
        default="",
        help="Path to PhishTank CSV file",
    )
    parser.add_argument(
        "--tranco",
        default="",
        help="Path to Tranco CSV file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of URLs to test in full mode (default: 200)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))
