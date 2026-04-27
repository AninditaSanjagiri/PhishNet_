## PhishNet Evaluation (Benchmark Mode)

n=40 · 20 phishing · 20 legit

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | Latency |
|--------|----------|-----------|--------|----|---------|----------|
| URL only | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 31.4ms |
| Text only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 551.2ms |
| Fused (URL+Text) | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 582.5ms |

*Fusion F1 gain: -1.0000*
