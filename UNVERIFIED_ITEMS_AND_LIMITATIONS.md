# Unverified Items, Conflicting Metrics, and Academic Limitations

**Project Title:** Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion  
**Scope:** Disclosure of scientific limitations, unverified assumptions, edge-case failure modes, and technical constraints for academic report preparation.

---

## 1. Summary of Verification Status Across Project Claims

| Item / Claim Category | Claim Description | Status | Evidence & Resolution Notes |
| :--- | :--- | :---: | :--- |
| **Model Architecture** | 6 Encoders (3 Vision + 3 Audio) | **VERIFIED** | Confirmed in `models/multimodal_model.py` and `experiments/05_full_3v3/config.json`. |
| **Parameter Breakdown** | 166.4M total, 157.7M frozen, 8.76M trainable | **VERIFIED** | Audited directly from model dictionary and configuration metadata. |
| **Final Test Performance** | 97.52% Acc, 98.71% F1, 99.02% AUC, 93.63% Bal Acc | **VERIFIED** | Confirmed in `experiments/05_full_3v3/metrics/metrics.json`. |
| **Confusion Matrix** | TN=17, FP=2, FN=13, TP=572 on 604 test set | **VERIFIED** | Matches test manifest evaluation and serialized metric JSONs. |
| **Missing Audio Robustness** | 97.68% Acc, +95.9% Speedup, 0 crashes | **VERIFIED** | Confirmed in `experiments/06_forensic_analysis/reports/missing_audio_report.md`. |
| **Dubbing Separation** | Mean $S_{AV} = +0.2618$ (normal) vs $-0.0060$ (mismatched) | **VERIFIED** | Confirmed in `experiments/06_forensic_analysis/reports/dubbed_audio_report.md`. |
| **Temperature Calibration** | $T = 0.7924$, ECE reduced from 5.25% to 3.21% | **VERIFIED** | Confirmed in `experiments/phase8/calibration/calibration_model.json`. |
| **OOD Validation Set Fit** | Centroids fitted on 2,414 validation samples | **VERIFIED** | Confirmed in `experiments/phase8/ood/reliability_detector.py`. |
| **Robustness Stress Test** | 14 degradation conditions, all $>95.5\%$ accuracy | **VERIFIED** | Confirmed in `experiments/phase8/reports/robustness_report.md`. |
| **Screen Capture Classification** | External screen-recorded deepfakes detected by model | **NOT IMPLEMENTED / KNOWN FAILURE** | Raw classifier predicted `REAL` (96.5%–98.0%); Phase 8 **did not retrain or correct the classifier**, but successfully added **OOD / LOW RELIABILITY warnings**. |
| **Spectrogram Standalone Failure** | Spectrogram CNN 3.15% accuracy proves spectrograms fail | **INCORRECT INTERPRETATION** | Spectrogram CNN conv layers were **randomly initialized and frozen** in that comparison; poor performance was due to random projection, not spectrogram invalidity. |
| **Multi-Dataset Training** | Trained across FaceForensics++, DFDC, and Celeb-DF | **NOT IMPLEMENTED** | System was trained and evaluated strictly on **FakeAVCeleb**; cross-dataset generalization remains an open limitation. |
| **Phoneme-Viseme Token Sync** | Fine-grained token-level cross-attention (e.g. SyncNet) | **NOT IMPLEMENTED** | Implemented as global cosine similarity $S_{AV}$ between projected 256-D feature vectors. |
| **Full Test Set Evaluation (2,415)**| Phase 5 3V3 evaluated on all 2,415 test samples | **UNVERIFIED / REPORTED ON 604** | Comparative phases (Phase 3–8) standardized on the 604-sample test manifest; full 2,415 test evaluation was not preserved in final logs. |

---

## 2. Analysis of Known Failures and Edge Cases

### A. The External Screen-Recording Failure Case
- **Incident Description**: During user demonstration testing, two external fake videos created via screen capture (`video1_222131.mp4`, `video2_222513.mp4`) were submitted to the deployed detector.
- **Observed Behavior**: The classifier outputted a prediction of `REAL` with high raw confidence ($96.5\%$ and $98.0\%$).
- **Investigation & Root Cause Analysis (`experiments/06_forensic_analysis/reports/external_generalization_diagnosis.md`)**:
  1. **Label Mapping Verification**: Code audits confirmed that label index 0 is REAL and index 1 is FAKE across all modules (`models/classifier.py`, `app/inference_service.py`, `app/static/js/app.js`). There was no label inversion bug.
  2. **In-Distribution Integrity**: Evaluating 16 in-distribution test videos (8 Real, 8 Fake) produced $100.0\%$ accuracy with zero errors, confirming the model weights were intact.
  3. **Compression & Resolution Factor**: The external screen recordings exhibited non-standard resolutions ($1336 \times 688$), highly atypical aspect ratios ($1.94$ vs standard $1.33$ or $1.77$), very small facial ROI coverage ($3.7\%$ and $5.9\%$ of total frame area vs $>25\%$ in FakeAVCeleb), and lossy H.264 desktop screen-recorder compression that smoothed high-frequency synthesis seams.
  4. **Softmax Overconfidence**: When presented with an out-of-distribution sample far from the training manifold, uncalibrated linear layers produced large logit separation ($z_0 - z_1 \approx 5.8$), resulting in saturated confidence for Class 0.
- **Phase 8 Resolution & Status**:
  - Phase 8 **did NOT modify the classifier weights** and did **NOT alter the decision boundary**.
  - Instead, Phase 8 introduced the **OOD & Forensic Reliability Layer**: both videos produced representation distances of $29.2$ and $29.4$ (exceeding the in-distribution 95th-percentile threshold of $27.93$) and small face area flags ($<6\%$), triggering an explicit **`LOW RELIABILITY / POSSIBLY OOD`** warning.
  - **Academic Integrity Disclosure**: The final report must not claim that the deepfake detector "successfully classified" these screen recordings; it must accurately state that the system **flagged them as untrusted out-of-distribution inputs**, thereby preventing uncritical acceptance of false negatives.

---

### B. The Spectrogram CNN Standalone Evaluation Caveat
- **Reported Metric**: In Phase 7 (`run_individual_model_comparison.py`), the standalone Spectrogram CNN recorded:
  - Accuracy: **3.15%**
  - Precision: **0.00%**, Recall: **0.00%**, F1-Score: **0.00%**
  - Confusion Matrix: $TN = 19, FP = 0, FN = 585, TP = 0$
- **Technical Context**:
  - In Phase 7, the experimental protocol required 100% frozen backbones (`requires_grad = False`).
  - Unlike the five other encoders (EfficientNet, Video Swin, ResNet-18, Whisper, Wav2Vec2) which loaded official pretrained foundation weights, the Spectrogram 2D CNN was a custom-designed architecture trained from scratch during multimodal training.
  - When instantiated with `freeze = True` for standalone benchmarking, its convolutional layers were **randomly initialized**.
  - Training a linear head on 256-D random projections under extreme 97.7% class imbalance caused complete optimization collapse to the majority negative class.
- **Academic Recommendation**: The final report must explicitly clarify that this 3.15% result reflects **un-adapted random feature projection**, not an inherent defect of log-Mel spectrogram acoustic features.

---

## 3. Inherent Dataset & Benchmark Limitations

### A. Extreme Benchmark Class Imbalance (42.5 : 1 Ratio)
- **Manifest Distribution**: `processed/metadata.csv` contains 15,725 Fake samples (97.70%) and only 370 Real samples (2.30%).
- **Impact on Training**: Standard unweighted Cross-Entropy loss collapsed to majority-class prediction (Phase 1 Baseline achieved 7.04% accuracy; Phase 2A achieved 0% Real Recall).
- **Mitigation Efficacy**: While `WeightedRandomSampler` and `FocalLoss` successfully recovered minority-class detection (Balanced Accuracy: 93.63%, Real F1: 69.39%), the scarcity of real data remains an unavoidable structural constraint of the underlying FakeAVCeleb corpus.

### B. Discrete Specificity Granularity on Test Manifest
- **Test Set Population**: The standardized test manifest contains exactly 604 videos, of which **19 are Real** and **585 are Fake**.
- **Metric Sensitivity**: In a cohort of 19 authentic videos, each individual sample accounts for:
  $$\frac{1}{19} \approx 5.263\% \text{ of the Specificity metric}$$
- Consequently, correctly classifying 16 samples yields $84.21\%$, 17 samples yields $89.47\%$, and 18 samples yields $94.74\%$. The discrete nature of this metric must be acknowledged when discussing statistical confidence intervals.

---

## 4. Methodological & Algorithmic Limitations

### A. Unimodal Video Dependence for Missing Audio
- **Operational Reality**: When audio is missing or muted, the system automatically falls back to visual-only mode ($\alpha_v = 1.0, \alpha_a = 0.0$), maintaining $97.68\%$ accuracy.
- **Trade-off**: In fallback mode, Balanced Accuracy drops from **93.63% to 83.53%** and Real F1 drops from **69.39% to 65.00%**. This demonstrates that while the visual stream is robust, acoustic cues contribute meaningfully to authentic class discrimination.
- **Asymmetric Fallback**: The reverse condition (video missing, audio only) is not currently supported by the preprocessing pipeline.

### B. Global vs Token-Level Audio-Visual Synchronization
- **Current Formulation**: Audio-visual consistency $S_{AV} = \mathbf{e}_v^T \mathbf{e}_a$ computes the cosine similarity between globally pooled 256-D visual and acoustic embeddings.
- **Capability**: It effectively detects global semantic misalignment, foreign-language dubbing, and voice replacement ($\Delta = +0.2678$ separation margin).
- **Constraint**: It does not compute fine-grained temporal alignment between sub-syllable phonemes and localized visemes (such as frame-level lip movement matching). Sub-second temporal desynchronization cannot be localized to specific words.

### C. Pretrained Encoder Weight Freezing
- **Design Decision**: In Phase 5, all six foundation backbones were frozen (`requires_grad = False`), and only the aggregation layers, projection heads, and fusion module (8.76M parameters) were trained.
- **Advantage**: Prevents catastrophic forgetting, reduces training time to 8.2 seconds per epoch on cached features, and prevents overfitting on the small authentic class.
- **Limitation**: The backbones cannot adapt their early receptive fields to deepfake-specific artifact manifolds, placing the entire burden of discrimination on high-level representation aggregation.

---

## 5. Agent & Explainability Limitations

### A. Non-Vision LLM Reasoning
- **Architecture**: The LLM agent (`agent/llm_forensic_agent.py`) receives a structured JSON evidence payload containing numerical confidences, modality attention weights, consistency scores, input quality statistics, and reliability badges.
- **Constraint**: The LLM does not perform native image or audio processing; it does not visually inspect video frames or listen to audio waveforms. Its reasoning is strictly an interpretation of the deep learning detector's quantitative telemetry.
- **Rule 7 Invariance Guarantee**: The LLM is strictly prohibited from altering or reversing the detector's classification verdict ($P(\text{Real}), P(\text{Fake})$).

### B. Threshold Dependency in OOD Detection
- **Calibration Source**: OOD distance thresholds ($27.93$ for 95th-percentile, $33.17$ for 99th-percentile) and consistency thresholds ($S_{AV} < 0.1177$) were fitted empirically on the validation split of FakeAVCeleb (`val_features.pt`).
- **Limitation**: These thresholds reflect the feature distribution of VoxCeleb2-based recordings. Videos captured in drastically different environments (e.g., thermal imaging, extreme low-light, CCTV footage) may trigger OOD warnings even if authentic.

---

## 6. Items Requiring User Confirmation Prior to Final Report Writing

1. **Target Academic Department & Degree Specifications**: Confirm the official department name, degree title, supervisor name, and institutional affiliation to be placed on the title page.
2. **Specific Formatting Template**: The final report must follow the institution's specific chapter numbering, margins, font sizes, and citation formatting (e.g., IEEE vs. Harvard).
3. **Preferred Final Report Scope**: Clarify whether the report should emphasize the **deep learning multimodal architecture (3V3)** or the **agentic explanation & forensic interface** as its primary academic contribution.
