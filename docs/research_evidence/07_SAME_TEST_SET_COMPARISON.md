# Strict Same-Test-Set Comparison: Baseline vs. Proposed Models

## 1. Scientific Verification Protocol
To ensure rigorous, publication-grade comparative research, all models were evaluated on the **EXACT SAME test dataset population** without data alteration, post-hoc threshold tuning, or weight modification:
- **Test Set Manifest**: `experiments/03_vision_swin/same_test_comparison/test_manifest.csv`
- **Total Test Samples**: Exactly 604 videos (19 Real, 585 Fake)
- **Evaluation Mode**: Strict `torch.inference_mode()`, threshold $t = 0.50$.

---

## 2. Comparative Performance Matrix

```
========================================================================================================================
METRIC                   BASELINE (1V1)         PHASE 3 VIDEO SWIN (2V1A)     PHASE 5 FULL 3V3       ABSOLUTE GAIN (P5 vs Base)
========================================================================================================================
Test Samples             604 (Identical)        604 (Identical)               604 (Identical)        Identical Test Bed
Accuracy                 73.18%                 97.52%                        97.52%                 +24.34% pp
Precision                97.74%                 99.48%                        99.65%                 +1.91% pp
Recall (Sensitivity)     74.02%                 97.95%                        97.78%                 +23.76% pp
F1-Score                 84.24%                 98.71%                        98.71%                 +14.47% pp
ROC-AUC                  59.99%                 98.80%                        99.02%                 +39.03% pp
Balanced Accuracy        60.69%                 91.08%                        93.63%                 +32.94% pp
Specificity (Real Rec)   47.37%                 84.21%                        89.47%                 +42.10% pp
Real F1-Score            10.00%                 68.09%                        69.39%                 +59.39% pp
Fake F1-Score            84.24%                 98.71%                        98.71%                 +14.47% pp
------------------------------------------------------------------------------------------------------------------------
Confusion Matrix         [TN=9, FP=10,          [TN=16, FP=3,                 [TN=17, FP=2,          -139 False Negatives
(TN, FP, FN, TP)          FN=152, TP=433]        FN=12, TP=573]                FN=13, TP=572]         -8 False Positives
========================================================================================================================
```

---

## 3. Paired Sample-by-Sample Analysis (Baseline vs. Phase 3)
A paired sample-by-sample analysis across all 604 test videos (`paired_prediction_analysis.csv`) reveals:
- **Both Models Correct**: 436 samples (72.19%)
- **Only Baseline Correct**: 6 samples (0.99%)
- **Only Proposed Model Correct**: 153 samples (25.33%)
- **Both Models Incorrect**: 9 samples (1.49%)
- **Statistical Significance**: McNemar's test with continuity correction yields $\chi^2 = 134.5$, $p < 10^{-15}$, confirming that the improvement is statistically overwhelming.

---

## 4. Key Scientific Insights from Same-Test-Set Benchmarking
1. **Dramatically Reduced False Negatives**: The baseline missed 152 manipulated videos ($	ext{FN} = 152$). Incorporating 3D temporal modeling (Video Swin) and lip articulation (Face/Mouth ResNet) reduced false negatives to 13 ($	ext{FN} = 13$), capturing subtle temporal glitches and boundary seam warping.
2. **Substantial Specificity Gains**: Authentic video detection improved from 47.37% (9/19) to 89.47% (17/19), reducing false alarms on legitimate media by 80%.
3. **Imbalance Robustness**: ROC-AUC surged from 59.99% to 99.02% (+39.03 percentage points), demonstrating exceptional threshold-independent discrimination under 97.7% class skewness.
