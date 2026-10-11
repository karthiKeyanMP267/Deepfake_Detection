# Known Limitations, Edge Cases & Scientific Disclosure

**Project Title:** Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion  
**Scope:** Transparent disclosure of empirical limitations, edge-case failure modes, domain shift behavior, and ethical guidelines.

---

## 1. Summary of Verification Status Across Project Claims

| Item / Claim Category | Claim Description | Status | Evidence & Resolution Notes |
| :--- | :--- | :---: | :--- |
| **Model Architecture** | 6 Encoders (3 Vision + 3 Audio) | **VERIFIED** | Audited in `models/multimodal_model.py` and `experiments/05_full_3v3/config.json`. |
| **Parameter Breakdown** | 166.4M total, 157.7M frozen, 8.76M trainable | **VERIFIED** | Audited directly from PyTorch state dictionary and config files. |
| **Final Test Performance** | 97.52% Acc, 98.71% F1, 99.02% AUC, 93.63% Bal Acc | **VERIFIED** | Audited in `experiments/05_full_3v3/metrics/metrics.json`. |
| **Confusion Matrix** | TN=17, FP=2, FN=13, TP=572 on 604 test set | **VERIFIED** | Matches test manifest evaluation and serialized metric JSONs. |
| **Missing Audio Robustness** | 97.68% Acc, +95.9% Speedup, 0 crashes | **VERIFIED** | Audited in `experiments/06_forensic_analysis/reports/missing_audio_report.md`. |
| **Dubbing Separation** | Mean $S_{AV} = +0.2618$ (normal) vs $-0.0060$ (mismatched) | **VERIFIED** | Audited in `experiments/06_forensic_analysis/reports/dubbed_audio_report.md`. |
| **Temperature Calibration** | $T = 0.7924$, ECE reduced from 5.25% to 3.21% | **VERIFIED** | Audited in `experiments/phase8/calibration/calibration_model.json`. |
| **OOD Validation Set Fit** | Centroids fitted on 2,414 validation samples | **VERIFIED** | Audited in `experiments/phase8/ood/reliability_detector.py`. |
| **Robustness Stress Test** | 14 degradation conditions, all $>95.5\%$ accuracy | **VERIFIED** | Audited in `experiments/phase8/reports/robustness_report.md`. |
| **Screen Capture Classification** | External screen-recorded deepfakes detected by model | **KNOWN FAILURE MODE** | Raw classifier predicted `REAL` (96.5%–98.0%); Phase 8 **did not retrain the classifier**, but successfully added **OOD / LOW RELIABILITY warnings**. |
| **Spectrogram Standalone Failure** | Spectrogram CNN 3.15% accuracy proves spectrograms fail | **INCORRECT INTERPRETATION** | Spectrogram CNN conv layers were **randomly initialized and frozen** in that comparison; poor performance was due to random projection, not spectrogram invalidity. |
| **Multi-Dataset Training** | Trained across FaceForensics++, DFDC, and Celeb-DF | **NOT IMPLEMENTED** | System was trained and evaluated strictly on **FakeAVCeleb**; cross-dataset generalization remains an open limitation. |
| **Phoneme-Viseme Token Sync** | Fine-grained token-level cross-attention (e.g. SyncNet) | **NOT IMPLEMENTED** | Implemented as global cosine similarity $S_{AV}$ between projected 256-D feature vectors. |

---

## 2. Analysis of Known Failures and Edge Cases

### 2.1 The External Screen-Recording Failure Case
- **Incident Description:** During evaluation with external screen-captured videos (`video1_222131.mp4`, `video2_222513.mp4`), the deepfake classifier predicted `REAL` with high raw confidence (96.47% and 98.00%).
- **Root Cause Investigation:**
  1. **Label Mapping Integrity:** Verified that Class 0 = REAL and Class 1 = FAKE across all modules. No label inversion occurred.
  2. **In-Distribution Integrity:** Evaluating 16 in-distribution test videos produced 100.0% accuracy, confirming weights were intact.
  3. **Compression & Resolution Factor:** The screen recordings possessed non-standard resolutions ($1336 \times 688$), atypical aspect ratios ($1.94$ vs standard $1.33$ or $1.77$), very small facial ROI coverage ($3.7\%$ and $5.9\%$ of total frame area vs $>25\%$ in FakeAVCeleb), and lossy H.264 desktop compression that smoothed spatial manipulation artifacts.
  4. **Softmax Overconfidence:** When presented with out-of-distribution vectors far from the training manifold, uncalibrated linear layers produced large logit separation ($z_0 - z_1 \approx 5.8$), causing saturated confidence for Class 0.
- **Architectural Safeguards Implemented:**
  - Added **Video Quality Pre-Check** (`app/input_quality.py`) flagging low face coverage and atypical aspect ratios.
  - Added **Mahalanobis Centroid Reliability Detector** (`experiments/phase8/ood/reliability_detector.py`) flagging such inputs as `LOW RELIABILITY — POSSIBLE OUT-OF-DISTRIBUTION INPUT`.

### 2.2 Clarification on Spectrogram CNN Standalone Performance
In the unimodal ablation study (`Phase 7`), the Spectrogram CNN exhibited low standalone classification accuracy (3.15%). This was caused by freezing randomly initialized convolutional layers without pretraining on AudioSet or speech data. In the complete 3V3 architecture, the Spectrogram CNN features are integrated into the multimodal fusion pipeline, contributing to overall performance.

---

## 3. Dataset Constraints & Generalization Boundaries

- **Single Benchmark Focus:** The detector was trained and tested on FakeAVCeleb v1.2. Performance on external benchmarks (FaceForensics++, Celeb-DF v2, DFDC) has not been evaluated in this setup.
- **Domain Adaptation:** Downstream applications deploying on heavily compressed social media streams (e.g., TikTok, WhatsApp, X) should incorporate domain-adaptive fine-tuning.

---

## 4. Responsible AI & Legal Evidentiary Disclaimer

This software is developed strictly as an academic research prototype and decision-support instrument for forensic investigators, researchers, and journalists. 

1. **Human-in-the-Loop Required:** Under no circumstances should binary predictions or automated agent narratives from this tool be used as sole evidence in legal, judicial, or disciplinary determinations without certified expert forensic validation.
2. **Defensive Research Intent:** This software is designed exclusively to identify and analyze synthetic media manipulations to enhance digital media authenticity and public trust.
