# 3V3 Champion Baseline Specification & Protection Guarantee
**Project:** Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion  
**Champion Checkpoint:** `experiments/05_full_3v3/models/best_model.pth`  
**Protection Level:** READ-ONLY / FROZEN / NEVER OVERWRITTEN  

---

## 1. Executive Summary

This document establishes the official reference record for our **Champion 3V3 Multimodal Deepfake Detector**. Under Phase 8, this model is strictly protected. It will not be retrained, fine-tuned, overwritten, moved, or deleted. All Phase 8 components (Robustness, Input Quality, OOD/Reliability, Confidence Calibration, Agentic Forensic Intelligence, and UI) wrap and support this model without modifying its learned weights or classification authority.

---

## 2. Champion Model Checkpoint Verification

| Property | Value |
| :--- | :--- |
| **Model Name** | Full 3-Vision + 3-Audio Multimodal Deepfake Detector (6 Encoders) |
| **Checkpoint Path** | `experiments/05_full_3v3/models/best_model.pth` |
| **File Size** | 763,326,482 bytes (727.97 MB) |
| **SHA-256 Checksum** | `A46C39996EF19F86D25E884209C37E50DB201A16E78B6CA0D8D1BD230934F92A` |
| **Modification Timestamp**| 2026-08-24 14:34:56 UTC |
| **Training Epoch** | Epoch 4 |
| **Validation F1** | 0.9722 |
| **Total Parameters** | 166,440,678 |
| **Frozen Parameters** | 157,682,658 (94.74%) |
| **Trainable Parameters** | 8,758,020 (5.26%) |

---

## 3. Verified Fixed-Test Baseline Metrics

Evaluated on the exact 604-sample test manifest (`experiments/03_vision_swin/same_test_comparison/test_manifest.csv`):

| Evaluation Metric | Baseline Value | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **97.52%** | 589 correct predictions out of 604 samples |
| **F1 Score (Overall)** | **98.71%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **99.02%** | Area under the receiver operating characteristic curve |
| **Balanced Accuracy** | **93.63%** | Macro-average of real and fake class accuracies |
| **Specificity (Real Recall)** | **89.47%** | 17 out of 19 Real videos correctly identified as REAL |
| **Sensitivity (Fake Recall)**| **97.78%** | 572 out of 585 Fake videos correctly identified as FAKE |
| **Real Precision** | **56.67%** | 17 True Reals out of 30 total Real predictions |
| **Real F1 Score** | **69.39%** | Harmonic mean of Real precision and recall |
| **Fake Precision** | **99.65%** | 572 True Fakes out of 574 total Fake predictions |
| **Fake F1 Score** | **98.71%** | Harmonic mean of Fake precision and recall |

### Confusion Matrix Breakdown:
```text
                    PREDICTED REAL        PREDICTED FAKE
ACTUAL REAL (19)          17 (TN)                 2 (FP)
ACTUAL FAKE (585)         13 (FN)               572 (TP)
```
- **Total Test Samples**: 604
- **True Negatives (TN)**: 17
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 13
- **True Positives (TP)**: 572

---

## 4. Architectural Invariance Rules

1. **Weight Immutability**: No optimizer or backward pass shall ever target the 3V3 checkpoint weights in Phase 8.
2. **Deterministic Reproducibility**: Running inference on the fixed 604 test samples must always return these exact numbers.
3. **Decoupled Classification**: Downstream layers (OOD warning, Calibration, LLM Agent) must never alter the classifier's verdict or confidence.
4. **Fallback Champion**: If any Phase 8 experimental extension decreases performance or introduces instability, the system reverts immediately to this untouched champion baseline.
