"""
training/train_url_model.py
============================
Train the URL Random Forest on real public datasets.

Datasets (free, no login required):
  PhishTank verified feed:
    https://data.phishtank.com/data/online-valid.csv
  Tranco Top-1M (legitimate):
    https://tranco-list.eu/download/latest/1000000

Auto-download mode:
  python training/train_url_model.py --auto-download

Manual mode:
  python training/train_url_model.py \
    --phishtank data/phishtank.csv \
    --tranco    data/tranco.csv

Outputs:
  models/url_rf_model.joblib
  evaluation/url_rf_metrics.json
  evaluation/url_rf_confusion.png
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, RocCurveDisplay)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.url_agent import extract_features

MODELS_DIR = Path(__file__).parent.parent / "models"
EVAL_DIR   = Path(__file__).parent.parent / "evaluation"
DATA_DIR   = Path(__file__).parent.parent / "data"


def auto_download(limit: int = 5000) -> tuple[list, list]:
    """Download PhishTank feed + Tranco top-1M sample."""
    import requests
    phish_urls, legit_urls = [], []

    # PhishTank
    print("Downloading PhishTank feed…")
    try:
        r = requests.get(
            "https://data.phishtank.com/data/online-valid.csv",
            timeout=30, headers={"User-Agent":"PhishNet-Research/2.0"})
        r.raise_for_status()
        lines = r.text.splitlines()
        reader = csv.DictReader(lines)
        for row in reader:
            url = row.get("url","").strip()
            if url and len(phish_urls) < limit:
                phish_urls.append(url)
        print(f"  Got {len(phish_urls)} phishing URLs")
    except Exception as exc:
        print(f"  PhishTank download failed: {exc}")

    # Tranco
    print("Downloading Tranco sample…")
    try:
        r = requests.get("https://tranco-list.eu/download/latest/100000",
                         timeout=60, headers={"User-Agent":"PhishNet-Research/2.0"})
        r.raise_for_status()
        for i, line in enumerate(r.text.splitlines()):
            if i >= limit: break
            parts = line.split(",")
            if len(parts) >= 2:
                domain = parts[1].strip()
                if domain:
                    legit_urls.append(f"https://{domain}")
        print(f"  Got {len(legit_urls)} legitimate URLs")
    except Exception as exc:
        print(f"  Tranco download failed: {exc}")

    return phish_urls, legit_urls


def load_from_files(phishtank_path: str, tranco_path: str,
                    limit: int = 10000) -> tuple[list, list]:
    phish, legit = [], []

    with open(phishtank_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or row.get("URL") or "").strip()
            if url and len(phish) < limit:
                phish.append(url)

    with open(tranco_path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            parts = line.strip().split(",")
            domain = parts[-1].strip() if parts else ""
            if domain:
                legit.append(f"https://{domain}")

    return phish, legit


def featurize(urls: list[str], label: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for url in urls:
        try:
            feat, _ = extract_features(url)
            X.append(feat)
            y.append(label)
        except Exception:
            pass
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--phishtank", default="")
    parser.add_argument("--tranco",    default="")
    parser.add_argument("--limit",     type=int, default=5000)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    if args.auto_download:
        phish_urls, legit_urls = auto_download(args.limit)
    elif args.phishtank and args.tranco:
        phish_urls, legit_urls = load_from_files(
            args.phishtank, args.tranco, args.limit)
    else:
        print("No data source specified. Use --auto-download or --phishtank + --tranco")
        print("Falling back to synthetic data for testing…")
        from agents.url_agent import URLAgent
        agent = URLAgent()
        agent._train_synthetic_model()
        return

    if len(phish_urls) < 50 or len(legit_urls) < 50:
        print(f"⚠️  Too few samples (phish={len(phish_urls)}, legit={len(legit_urls)})")
        sys.exit(1)

    print(f"\nFeaturizing {len(phish_urls)} phishing + {len(legit_urls)} legit…")
    Xp, yp = featurize(phish_urls, 1)
    Xl, yl = featurize(legit_urls, 0)
    X = np.vstack([Xp, Xl])
    y = np.concatenate([yp, yl])
    print(f"Feature matrix: {X.shape}")

    # ── SMOTE to handle class imbalance ──────────────────────────────────
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    print(f"After SMOTE: {X_res.shape}  ({y_res.sum()} phishing)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # ── Train ────────────────────────────────────────────────────────────
    print("\nTraining Random Forest (200 trees)…")
    t0  = time.time()
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    train_time = round(time.time() - t0, 2)
    print(f"  Training time: {train_time}s")

    # ── 5-fold CV ────────────────────────────────────────────────────────
    cv_f1 = cross_val_score(clf, X_res, y_res, cv=5, scoring="f1", n_jobs=-1)
    print(f"  5-fold CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── Test set evaluation ───────────────────────────────────────────────
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    auc     = roc_auc_score(y_te, y_proba)

    print("\nTest Set Report:")
    report = classification_report(y_te, y_pred,
                                   target_names=["Legitimate","Phishing"],
                                   output_dict=True)
    print(classification_report(y_te, y_pred,
                                target_names=["Legitimate","Phishing"]))
    print(f"ROC-AUC: {auc:.4f}")

    # ── Per-sample latency ────────────────────────────────────────────────
    t_lat = time.perf_counter()
    for _ in range(100):
        clf.predict_proba(X_te[:1])
    lat_ms = round((time.perf_counter() - t_lat) / 100 * 1000, 3)
    print(f"Avg inference latency: {lat_ms}ms per sample")

    # ── Save metrics ─────────────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "RandomForest",
        "n_train": int(X_tr.shape[0]),
        "n_test":  int(X_te.shape[0]),
        "accuracy": round(report["accuracy"], 4),
        "precision_phishing": round(report["Phishing"]["precision"], 4),
        "recall_phishing":    round(report["Phishing"]["recall"], 4),
        "f1_phishing":        round(report["Phishing"]["f1-score"], 4),
        "roc_auc":            round(auc, 4),
        "cv_f1_mean":         round(cv_f1.mean(), 4),
        "cv_f1_std":          round(cv_f1.std(), 4),
        "avg_latency_ms":     lat_ms,
        "train_time_s":       train_time,
    }
    with open(EVAL_DIR / "url_rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion matrix plot ─────────────────────────────────────────────
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Phishing"],
                yticklabels=["Legit","Phishing"], ax=ax)
    ax.set_title("URL RF — Confusion Matrix")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "url_rf_confusion.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → evaluation/url_rf_confusion.png")

    # ── ROC curve plot ────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_te, y_proba, ax=ax2,
                                     name=f"RF (AUC={auc:.3f})")
    ax2.set_title("URL RF — ROC Curve")
    fig2.tight_layout()
    fig2.savefig(EVAL_DIR / "url_rf_roc.png", dpi=150)
    plt.close()
    print(f"  ROC curve saved → evaluation/url_rf_roc.png")

    # ── Save model ────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "url_rf_model.joblib")
    print(f"\n✅ Model saved → models/url_rf_model.joblib")
    print(f"✅ Metrics saved → evaluation/url_rf_metrics.json")


if __name__ == "__main__":
    main()
