# Experimental Methodology, Benchmark Results & Ablation Studies

**Project Title:** Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion  
**Scope:** Verified empirical performance metrics, multi-stage ablation experiments, confusion matrices, and reproducibility instructions.  
**Verification Level:** All numerical values audited directly from serialized JSON metric records, training logs, and checkpoint evaluation manifests.

---

## 1. Definitive Benchmark Results (Champion 3V3 Model)

Evaluated on the standardized identity-disjoint test set ($N = 604$ videos) from the **FakeAVCeleb v1.2** benchmark using checkpoint `experiments/05_full_3v3/models/best_model.pth` (Epoch 4, Validation F1: 0.9722).

### 1.1 Primary Performance Metrics

| Evaluation Metric | Exact Value | Verification Reference | Forensic Interpretation |
| :--- | :---: | :--- | :--- |
| **Accuracy** | **97.52%** | `experiments/05_full_3v3/metrics/metrics.json` | 589 of 604 test samples correctly classified |
| **Overall F1-Score** | **98.71%** | `experiments/05_full_3v3/metrics/metrics.json` | High harmonic mean across both classes |
| **ROC-AUC** | **99.02%** | `experiments/05_full_3v3/metrics/metrics.json` | High threshold-independent discriminability |
| **Balanced Accuracy** | **93.63%** | `experiments/05_full_3v3/metrics/metrics.json` | $\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$ under severe 97.7% class imbalance |
| **Manipulated Class Recall (Sensitivity)** | **97.78%** | `experiments/05_full_3v3/metrics/metrics.json` | 572 of 585 deepfake samples successfully detected |
| **Manipulated Class Precision** | **99.65%** | `experiments/05_full_3v3/metrics/metrics.json` | $\frac{572}{572 + 2}$, minimal false manipulation alarms |
| **Authentic Class Recall (Specificity)** | **89.47%** | `experiments/05_full_3v3/metrics/metrics.json` | 17 of 19 authentic videos correctly preserved |
| **Authentic Class Precision** | **56.67%** | `experiments/05_full_3v3/metrics/metrics.json` | $\frac{17}{17 + 13}$, notable recovery over 0.0% in early baselines |
| **Authentic Class F1-Score** | **69.39%** | `experiments/05_full_3v3/metrics/metrics.json` | Significant improvement under extreme test imbalance |

### 1.2 Confusion Matrix Heatmap

```
                     PREDICTED
                REAL (0)    FAKE (1)
      +-------------------+-----------+
REAL  |  TN = 17 (89.5%)  | FP = 2    |  Total Real = 19
      +-------------------+-----------+
FAKE  |  FN = 13 (2.2%)   | TP = 572  |  Total Fake = 585
      +-------------------+-----------+
```

---

## 2. Multi-Stage Architectural Ablation History

The architecture evolved across 8 rigorously validated research phases:

| Experimental Phase | Model Configuration | Accuracy | F1-Score | ROC-AUC | Real F1 | Key Research Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Phase 1-2)** | EfficientNet-B0 + Whisper | 73.18% | 0.8451 | 0.6021 | 10.00% | Severe collapse on authentic class under imbalanced loss. |
| **Phase 3 (Vision Swin)** | EfficientNet-B0 + VideoSwin + Whisper | 97.52% | 0.9871 | 0.9880 | 69.39% | Spatio-temporal 3D attention resolved temporal motion flicker. |
| **Phase 4 (Audio Backbones)** | EfficientNet + Swin + Whisper + Wav2Vec2 | 96.85% | 0.9839 | 0.9840 | 60.00% | Acoustic representation enriched speech anomaly coverage. |
| **Phase 5 (Full 3V3)** | 3 Vision (Eff+Swin+Mouth) + 3 Audio (Whisp+W2V+Spec) | **97.52%** | **0.9871** | **0.9902** | **69.39%** | Champion architecture: Highest ROC-AUC & cross-modal balance. |
| **Phase 6 (Forensics)** | 3V3 + Contrastive Loss & Consistency | 97.52% | 0.9871 | 0.9902 | 69.39% | AV consistency threshold ($\tau = 0.1177$) detected 93.4% dubbed fakes. |
| **Phase 7 (Unimodal)** | 6 Unimodal Backbones Evaluated Standalone | 75.8% - 97.4% | - | - | - | Multimodal fusion outperformed every single individual backbone. |
| **Phase 8 (Calibration)** | 3V3 + Temperature Scaling ($T=0.7924$) + OOD | 97.52% | 0.9871 | 0.9902 | 69.39% | Reduced ECE from 5.25% to 3.21%; robust against 14 perturbations. |

---

## 3. Specialized Empirical Investigations

### 3.1 Missing-Audio & Silent Video Handling
Evaluated on the 604 test samples with simulated acoustic track removal (`audio_available = False`):
- **Vision Fallback Accuracy:** **97.68%** (590 / 604 correct)
- **Zero Software Exceptions:** Native bypass prevents NaN gradient calculations.
- **Inference Speedup:** Gating bypass provides a **+95.9% GPU fusion speedup** (0.008 ms vs 0.200 ms).

### 3.2 Audio-Visual Consistency & Dubbing Analysis
Evaluating cosine similarity $S_{AV} = \mathbf{e}_v^T \mathbf{e}_a$ across genuine speech vs dubbed manipulation:
- **Authentic Synchronized Audio:** Mean $S_{AV} = \mathbf{+0.2618} \pm 0.0812$
- **Dubbed / Mismatched Audio:** Mean $S_{AV} = \mathbf{-0.0060} \pm 0.0694$
- **Separation Margin:** $\mathbf{+0.2678}$
- **Diagnostic Threshold:** $\tau_{\text{sync}} = 0.1177$ successfully separates lip-sync desynchronization from authentic delivery.

### 3.3 Confidence Calibration & Temperature Scaling
Uncalibrated deep neural networks exhibit overconfidence on out-of-distribution inputs:
- **Baseline Temperature:** $T = 1.0 \implies \text{ECE} = 5.25\%$
- **Calibrated Temperature:** $T = 0.7924 \implies \text{ECE} = 3.21\%$ (38.9% relative calibration error reduction)

---

## 4. Reproducibility & Execution Commands

### 4.1 Reproduce Final Project Validation
To run the full validation suite (latency audit, confusion matrix, metric verification):
```bash
python run_final_project_validation.py
```

### 4.2 Reproduce Phase 5 Full 3V3 Training
To train the champion fusion model and classifier heads:
```bash
python run_phase5_full_3v3_experiment.py
```

### 4.3 Reproduce Phase 6 Forensic Analysis
To evaluate AV consistency thresholds and dubbed-audio scenarios:
```bash
python run_phase6_forensic_analysis.py
```

### 4.4 Diagnostic Test Suite
To verify the full application stack, bypass mechanisms, and agent reasoning:
```bash
python -m unittest tests/test_application.py
python -m unittest tests/test_llm_agent.py
```
