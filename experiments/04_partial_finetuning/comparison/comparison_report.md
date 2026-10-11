# Phase-4 Partial Fine-Tuning Comparison Report

## Executive Summary

This experiment compares **Phase-3 Fully Frozen Backbones** vs **Phase-4 Partial Fine-Tuning** (EfficientNet-B0 + Video Swin + Whisper + Wav2Vec2) on the identical 604-sample test split.

- **Dataset Manifest**: [`test_manifest.csv`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/same_test_comparison/test_manifest.csv) (604 samples)
- **Decision Threshold**: $t = 0.5$

---

## Metric Comparison Table

| Metric | Phase-3 (Fully Frozen) | Phase-4 (Partially Fine-Tuned) | Absolute Change |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **97.52%** | **97.52%** | **+0.00%** |
| **Precision** | 99.48% | 99.65% | +0.17% |
| **Recall** | 97.95% | 97.78% | -0.17% |
| **F1-Score** | **98.71%** | **98.71%** | **-0.00%** |
| **ROC-AUC** | **98.80%** | **98.25%** | **-0.54%** |
| **Balanced Accuracy** | **91.08%** | **93.63%** | **+2.55%** |
| **Real Class F1** | **68.09%** | **69.39%** | **+1.30%** |

---

## Confusion Matrix Comparison

### Phase-3 (Fully Frozen)
- True Real (TN): 16
- False Fake (FP): 3
- False Real (FN): 12
- True Fake (TP): 573

### Phase-4 (Partially Fine-Tuned)
- True Real (TN): 17
- False Fake (FP): 2
- False Real (FN): 13
- True Fake (TP): 572
