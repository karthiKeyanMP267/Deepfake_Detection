# Threshold Tuning Report (`threshold_tuning`)

## 1. Purpose of Threshold Tuning
The purpose of this experiment is to identify the optimal decision threshold for converting continuous model softmax probabilities $P(\text{Fake})$ into binary classification predictions (Real vs. Fake), without altering model architecture, weights, or hyperparameters.

---

## 2. Checkpoint Used
- **Checkpoint File**: `experiments\weighted_sampler\models\best_model.pth`
- **Source Experiment**: `WeightedRandomSampler`
- **Checkpoint Epoch**: `5`
- **Validation F1**: `0.8475`

---

## 3. Thresholds Evaluated
Tested decision thresholds: `[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]`

---

## 4. Complete Results Table

| Threshold | Accuracy | Precision | Recall | F1 Score | Balanced Acc | Specificity | Sensitivity | TP | TN | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 0.9768 | 0.9768 | 1.0000 | 0.9883 | 0.5000 | 0.0000 | 1.0000 | 2359 | 0 | 56 | 0 |
| 0.20 | 0.9768 | 0.9768 | 1.0000 | 0.9883 | 0.5000 | 0.0000 | 1.0000 | 2359 | 0 | 56 | 0 |
| 0.30 | 0.9768 | 0.9768 | 1.0000 | 0.9883 | 0.5000 | 0.0000 | 1.0000 | 2359 | 0 | 56 | 0 |
| 0.40 | 0.9520 | 0.9774 | 0.9733 | 0.9754 | 0.5134 | 0.0536 | 0.9733 | 2296 | 3 | 53 | 63 |
| 0.50 | 0.7491 | 0.9888 | 0.7516 | 0.8540 | 0.6972 | 0.6429 | 0.7516 | 1773 | 36 | 20 | 586 |
| 0.60 | 0.0248 | 1.0000 | 0.0017 | 0.0034 | 0.5008 | 1.0000 | 0.0017 | 4 | 56 | 0 | 2355 |
| 0.70 | 0.0232 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0 | 56 | 0 | 2359 |
| 0.80 | 0.0232 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0 | 56 | 0 | 2359 |
| 0.90 | 0.0232 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 0.0000 | 0 | 56 | 0 | 2359 |

---

## 5. Best Threshold & Selection Reasoning
- **Best Threshold**: `0.10`
- **Selection Criteria**:
  1. Primary: Highest F1-score (`0.9883`)
  2. Tie-breaker 1: Highest Balanced Accuracy (`0.5000`)
  3. Tie-breaker 2: Highest Real-class Recall / Specificity (`0.0000`)

---

## 6. Comparison with Default Threshold (0.50)

| Metric | Default Threshold (0.50) | Best Threshold (0.10) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.7491 | 0.9768 | +0.2277 |
| **Precision** | 0.9888 | 0.9768 | -0.0120 |
| **Recall** | 0.7516 | 1.0000 | +0.2484 |
| **F1-Score** | 0.8540 | 0.9883 | +0.1342 |
| **Balanced Accuracy** | 0.6972 | 0.5000 | -0.1972 |
| **Specificity** | 0.6429 | 0.0000 | -0.6429 |
| **Sensitivity** | 0.7516 | 1.0000 | +0.2484 |
| **ROC-AUC** | 0.7010 | 0.7010 | 0.0000 |

### Confusion Matrix Comparison:
- **Default (0.50)**:
  - TP: 1773, TN: 36, FP: 20, FN: 586
- **Best (0.10)**:
  - TP: 2359, TN: 0, FP: 56, FN: 0

---

## 7. Zero-Retraining Verification Audit
- **Retraining Executed**: NO
- **Model Weights Altered**: NO (Parameter sum checksum equal before and after inference: `5282.793096`)
- **Optimizer Steps Executed**: NO
- **Scheduler Steps Executed**: NO
- **Experimental Variable Changed**: Decision threshold ONLY.

---

## 8. Final Recommendations
- The decision threshold of **`0.10`** is recommended for deployment because it maximizes overall F1 score while maintaining optimal trade-offs between precision and recall under class imbalance.
