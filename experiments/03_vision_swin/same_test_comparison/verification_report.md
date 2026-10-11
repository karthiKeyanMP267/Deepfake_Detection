# Research Verification Report: Strict Same-Test-Set Comparison

## Experiment Overview

This experiment evaluates the **Weighted Sampler Baseline** and **Phase-3 Video Swin** models on **EXACTLY THE SAME test dataset population** to establish a scientifically rigorous and publication-grade performance comparison.

> [!IMPORTANT]
> - **Evaluation-Only**: Neither model was retrained; model weights were not modified.
> - **Identical Test Set**: Both models were evaluated on the exact same 604 test samples.
> - **Fixed Decision Threshold**: Threshold $t = 0.5$ was applied to both models without threshold tuning.

---

## 1. Test Population Manifest

- **Total Test Samples**: 604
- **Real Video Samples (Class 0)**: 19 (3.15%)
- **Fake Video Samples (Class 1)**: 585 (96.85%)
- **Manifest Location**: [`test_manifest.csv`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/same_test_comparison/test_manifest.csv)

---

## 2. Evaluated Checkpoints

1. **Weighted Sampler Baseline**:
   - Path: [`experiments/weighted_sampler/models/best_model.pth`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/weighted_sampler/models/best_model.pth)
   - Vision Backbone: EfficientNet-B0
   - Audio Backbone: Whisper Encoder

2. **Phase-3 Video Swin**:
   - Path: [`experiments/03_vision_swin/models/best_model.pth`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/models/best_model.pth)
   - Vision Backbone: EfficientNet-B0 + Video Swin Transformer (Feature Aggregation)
   - Audio Backbone: Whisper Encoder

---

## 3. Comparative Performance Metrics

| Metric | Weighted Sampler Baseline | Phase-3 Video Swin | Absolute Difference (Phase-3 - Baseline) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **73.18%** | **97.52%** | **+24.34%** |
| **Precision** | 97.74% | 99.48% | +1.74% |
| **Recall** | 74.02% | 97.95% | +23.93% |
| **F1-Score** | **84.24%** | **98.71%** | **+14.47%** |
| **ROC-AUC** | **59.99%** | **98.80%** | **+38.81%** |
| **Balanced Accuracy** | **60.69%** | **91.08%** | **+30.39%** |
| **Specificity (Real Rec)** | 47.37% | 84.21% | +36.84% |
| **Sensitivity (Fake Rec)** | 74.02% | 97.95% | +23.93% |

---

## 4. Per-Class Detailed Breakdown

### Real Class (Label 0)
- **Baseline**: Precision = 5.59%, Recall = 47.37%, F1 = 10.00%
- **Phase-3**: Precision = 57.14%, Recall = 84.21%, F1 = 68.09%

### Fake Class (Label 1)
- **Baseline**: Precision = 97.74%, Recall = 74.02%, F1 = 84.24%
- **Phase-3**: Precision = 99.48%, Recall = 97.95%, F1 = 98.71%

---

## 5. Confusion Matrices

### Raw Confusion Matrix (TN, FP, FN, TP)

```
Weighted Sampler Baseline:
            Predicted Real (0)    Predicted Fake (1)
True Real (0)       9               10             
True Fake (1)       152             433            

Phase-3 Video Swin:
            Predicted Real (0)    Predicted Fake (1)
True Real (0)       16              3              
True Fake (1)       12              573            
```

---

## 6. Paired Sample Prediction Analysis

Paired sample-by-sample analysis across all 604 test items:

- **Both Models Correct**: 436 samples (72.19%)
- **Only Baseline Correct**: 6 samples (0.99%)
- **Only Phase-3 Correct**: 153 samples (25.33%)
- **Both Models Wrong**: 9 samples (1.49%)

Detailed paired prediction log saved to [`paired_prediction_analysis.csv`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/same_test_comparison/paired_prediction_analysis.csv).

---

## 7. Key Findings & Research Conclusion

1. **Substantial Performance Gain**: Phase-3 Video Swin outperforms the Weighted Sampler baseline significantly across all primary classification metrics on identical test data:
   - **Accuracy**: +24.34% (73.18% $\rightarrow$ 97.52%)
   - **F1-Score**: +14.47% (84.24% $\rightarrow$ 98.71%)
   - **ROC-AUC**: +38.81% (59.99% $\rightarrow$ 98.80%)
   - **Balanced Accuracy**: +30.39% (60.69% $\rightarrow$ 91.08%)

2. **Temporal Dynamics Feature Impact**: Incorporating 3D Video Swin Transformer features alongside 2D EfficientNet texture features enables the model to capture temporal inconsistencies across frames, dramatically reducing false negatives on fake videos.

---

## Generated Visualizations

- **ROC Comparison**: [`plots/roc_comparison.png`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/same_test_comparison/plots/roc_comparison.png)
- **Precision-Recall Comparison**: [`plots/precision_recall_comparison.png`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/same_test_comparison/plots/precision_recall_comparison.png)
- **Metrics Comparison Chart**: [`plots/metrics_comparison.png`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/same_test_comparison/plots/metrics_comparison.png)
- **Baseline Confusion Matrix**: [`plots/baseline_confusion_matrix.png`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/same_test_comparison/plots/baseline_confusion_matrix.png)
- **Phase-3 Confusion Matrix**: [`plots/phase3_confusion_matrix.png`](file:///D:/Deepfake/Agentic-Multimodal-Deepfake-Detection/experiments/03_vision_swin/same_test_comparison/plots/phase3_confusion_matrix.png)
