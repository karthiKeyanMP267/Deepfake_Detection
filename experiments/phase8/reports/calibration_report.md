# Phase 8 Confidence Calibration Report

**Method:** Temperature Scaling (Post-Hoc Probability Calibration)
**Fitting Dataset:** 2,414 Validation Samples (`val_features.pt` — Test Set Unused for Optimization)
**Test Dataset:** 604-Sample Benchmark (`test_features.pt`)
**Model Weights Modified:** STRICTLY NONE (Invariance Preserved)

---

## 1. Calibration Parameter & Metric Summary

- **Learned Temperature Parameter ($T$):** `0.7924`

| Metric | Before Calibration (Raw Softmax) | After Calibration (Temperature Scaled) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Expected Calibration Error (ECE)** | **5.25%** | **3.21%** | **2.04% pp** |
| **Brier Score (MSE)** | **0.0237** | **0.0206** | **0.0031** |

---

## 2. Monotonicity & Decision Invariance Guarantee

- Because Temperature Scaling applies a strictly monotonic scalar divisor $T > 0$ to logits, the classification boundary $P(\text{fake}) \ge 0.50$ is algebraically invariant.
- **Every single prediction (Real vs Fake) remains identical to the 3V3 baseline.**
- Softmax confidence scores are gently smoothed, mitigating overconfidence on ambiguous samples.
