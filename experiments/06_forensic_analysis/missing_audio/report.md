# Missing Audio & Vision-Only Fallback Forensic Report

## Summary & Objectives
This experiment evaluates the multimodal deepfake detection system under real-world scenarios where audio is unavailable, silent, missing, or corrupted. The system features an automatic fallback mechanism: when audio is missing, it dynamically isolates the visual pathway (EfficientNet-B0, Video Swin, Face/Mouth ResNet-18) and sets vision_weight = 1.0, audio_weight = 0.0 to produce robust, non-crashing predictions.

---

## Quantitative Performance Comparison (604 Test Samples)

| Metric | Normal Multimodal | Vision-Only Fallback | Absolute Difference | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 97.52% | **97.68%** | +0.17% pp | +0.17% |
| **Precision** | 99.65% | **98.97%** | -0.68% pp | -0.68% |
| **Recall** | 97.78% | **98.63%** | +0.85% pp | +0.87% |
| **F1 Score** | 98.71% | **98.80%** | +0.10% pp | +0.10% |
| **Balanced Accuracy** | 93.63% | **83.53%** | -10.10% pp | -10.79% |
| **ROC-AUC** | 99.02% | **98.57%** | -0.45% pp | -0.45% |
| **Real F1** | 69.39% | **65.00%** | -4.39% pp | -6.32% |
| **Fake F1** | 98.71% | **98.80%** | +0.10% pp | +0.10% |

---

## Inference Efficiency & Latency Analysis

- **Normal Multimodal Latency**: 0.20 ms/sample
- **Vision-Only Fallback Latency**: 0.01 ms/sample
- **Inference Speedup**: +95.9% faster (since 3 audio encoders & audio aggregation are bypassed)

---

## Edge Case Stress Testing Matrix

| Scenario | Audio Condition | Assigned Mode | Vision Weight | Audio Weight | Test Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Case 1: Normal Audio** | Valid Synchronized | Adaptive Multimodal | 0.6666 | 0.3334 | **PASS** |
| **Case 2: Missing Audio** | None / Unsupplied | Vision-Only Fallback | 1.0000 | 0.0000 | **PASS** |
| **Case 3: Silent Audio** | Zero Amplitude Signal | Vision-Only Fallback | 1.0000 | 0.0000 | **PASS** |
| **Case 4: Corrupted Audio** | Severe Signal Noise | Vision-Only Fallback | 1.0000 | 0.0000 | **PASS** |
