# Phase 8 OOD & Forensic Reliability Report

**Evaluation Type:** In-Distribution vs Out-of-Distribution Calibration (No Retraining)
**Reference Population:** 2,414 Validation Samples (`val_features.pt` - Test Set Unused for Fitting)
**Evaluated Model:** 3V3 Champion Model (`experiments/05_full_3v3/models/best_model.pth`)

---

## 1. Reference Distribution Calibration (Validation Set)

- **Total Validation Samples Fitted:** 2,414 (Real: 55, Fake: 2359)
- **Feature Space:** 512-D Fused Multimodal Embedding Space
- **Distance Metric:** Standardized Class-Conditional Euclidean Distance (Diagonal Mahalanobis)
- **In-Distribution Distance (95th Percentile Threshold):** `27.93`
- **In-Distribution Distance (99th Percentile Threshold):** `33.17`

---

## 2. Benchmark Test Set Evaluation (604 Samples)

| Reliability Tier | Sample Count | Percentage | Operational Meaning |
| :--- | :---: | :---: | :--- |
| **HIGH** | 425 | 70.4% | High confidence, aligned multimodal evidence, in-distribution geometry |
| **MEDIUM** | 89 | 14.7% | Minor stream divergence or moderate compression variance |
| **LOW** | 90 | 14.9% | Extreme distance from learned manifold, small face, or screen capture |

---

## 3. Strict Non-Overriding Guarantee (Rule 7)

- The OOD and reliability layers **NEVER** alter the classifier verdict from Real to Fake or Fake to Real.
- OOD flags serve purely to advise human reviewers and the LLM agent on how much certainty should be attached to the detector output.
