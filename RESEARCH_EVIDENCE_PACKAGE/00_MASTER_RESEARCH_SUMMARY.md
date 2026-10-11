# Master Research Summary: Agentic Multimodal Deepfake Detection

**Project Title**: Agentic Multimodal Deepfake Detection using Audio-Visual Forensic Analysis  
**Target Paper Venue**: IEEE Transactions on Information Forensics and Security / IEEE Access  
**Final Research Model**: Phase 5 Full 3V3 Multimodal Architecture (6 Foundation Encoders + Autonomous Forensic Agent)  
**Selected Checkpoint**: `experiments/05_full_3v3/models/best_model.pth` (Epoch 4, Validation F1: 0.9722)  

---

## 1. Executive Summary & Research Gap
Contemporary generative deepfakes manipulate both visual spatial-temporal facial cues and acoustic speech representations. Existing detection frameworks suffer from three severe limitations:
1. **Unimodal & Narrow Encoders**: Fragility against cross-modal synthesis attacks due to reliance on a single 2D CNN or unimodal speech model.
2. **Dataset Class Imbalance (97.7% Fake)**: Unweighted training causes complete collapse to majority-class prediction.
3. **Operational Fragility**: Inability to handle missing audio tracks, conflation of foreign-language dubbing with visual deepfakes, and high-confidence softmax saturation under out-of-distribution (OOD) screen capture compression.

To overcome these gaps, this work proposes a **Full 3-Vision + 3-Audio (3V3)** foundation architecture unified via adaptive multimodal attention fusion, cross-modal InfoNCE contrastive alignment, dynamic missing-modality fallback, and an autonomous forensic reasoning agent.

---

## 2. Core System Architecture
```
========================================================================================================================
STREAM              BACKBONE MODEL               INPUT TENSOR SHAPE             OUTPUT DIM    FROZEN / TRAINABLE
========================================================================================================================
Vision Branch 1     EfficientNet-B0              (B, 16, 3, 224, 224)           1,280-D       Frozen (4.01M params)
Vision Branch 2     Video Swin-Tiny (3D)         (B, 16, 3, 224, 224)             768-D       Frozen (27.85M params)
Vision Branch 3     Face/Mouth ResNet-18         (B, 16, 3, 224, 224)             512-D       Frozen (11.18M params)
Vision Aggregator   Linear Projection Layer      (B, 2560)                      1,280-D       Trainable (5.90M params)
Vision Projector    MLP + BatchNorm + L2Norm     (B, 1280)                        256-D       Trainable (0.79M params)
------------------------------------------------------------------------------------------------------------------------
Audio Branch 4      Whisper Base.en              (B, 160000) -> Log-Mel           512-D       Frozen (19.82M params)
Audio Branch 5      Wav2Vec2 Base                (B, 160000) @ 16kHz              768-D       Frozen (94.37M params)
Audio Branch 6      Spectrogram 2D CNN           (B, 160000) -> STFT Mel          256-D       Trainable (0.45M params)
Audio Aggregator    Linear Projection Layer      (B, 1536)                        512-D       Trainable (1.44M params)
Audio Projector     MLP + BatchNorm + L2Norm     (B, 512)                         256-D       Trainable (0.40M params)
------------------------------------------------------------------------------------------------------------------------
Cross-Modal Metric  InfoNCE Contrastive Module   (B, 256) x 2                   Scalar Loss   Non-Parametric (tau=0.07)
Adaptive Fusion     Modality Attention Network   (B, 256) x 2                     512-D       Trainable (65.9K params)
Deepfake Classifier 2-Layer MLP + Dropout        (B, 512)                           2-D       Trainable (165.3K params)
Autonomous Agent    Forensic Reasoning Engine    Structured Evidence Dict       Report / JSON Deterministic Rule-Based
========================================================================================================================
TOTAL PARAMETER COUNT: 166,440,678  |  TRAINABLE: 9,212,612 (5.54%)  |  FROZEN: 157,228,066 (94.46%)
========================================================================================================================
```

---

## 3. Definitive Benchmark Results (604 Test Samples)

```
========================================================================================================================
METRIC                   BASELINE (1V1)         PHASE 3 VIDEO SWIN (2V1A)     FINAL PHASE 5 FULL 3V3 (3V3)
========================================================================================================================
Accuracy                 73.18%                 97.52%                        97.52% (+24.34% pp vs Baseline)
Precision (Fake)         97.74%                 99.48%                        99.65% (+1.91% pp vs Baseline)
Recall (Sensitivity)     74.02%                 97.95%                        97.78% (+23.76% pp vs Baseline)
F1-Score                 84.24%                 98.71%                        98.71% (+14.47% pp vs Baseline)
ROC-AUC                  59.99%                 98.80%                        99.02% (+39.03% pp vs Baseline)
Balanced Accuracy        60.69%                 91.08%                        93.63% (+32.94% pp vs Baseline)
Specificity (Real Rec)   47.37% (9/19)          84.21% (16/19)                89.47% (17/19) (+42.10% pp vs Baseline)
Authentic F1-Score       10.00%                 68.09%                        69.39% (+59.39% pp vs Baseline)
Manipulated F1-Score     84.24%                 98.71%                        98.71% (+14.47% pp vs Baseline)
------------------------------------------------------------------------------------------------------------------------
Confusion Matrix         TN=9, FP=10,           TN=16, FP=3,                  TN=17, FP=2,
(TN, FP, FN, TP)         FN=152, TP=433         FN=12, TP=573                 FN=13, TP=572
========================================================================================================================
```

---

## 4. Key Experimental Discoveries & Forensic Capabilities

### A. Non-Crashing Missing-Audio Robustness
- **Dynamic Fallback**: When audio is missing or silent ($\max(|x|) < 10^{-7}$), audio encoders are bypassed, and attention weights dynamically assign $lpha_v = 1.0, lpha_a = 0.0$.
- **Performance**: Retains **97.68% accuracy** and **98.80% F1-score**.
- **Inference Speedup**: +95.9% faster GPU fusion pass (0.008 ms vs 0.200 ms).

### B. Audio-Visual Consistency & Dubbing Disentanglement
- **Metric**: Embedding-level cosine similarity $S_{AV} = \mathbf{e}_v^T \mathbf{e}_a \in [-1.0, +1.0]$.
- **Separation Margin**: Matched audio yields mean $S_{AV} = \mathbf{+0.2618}$, while controlled dubbed audio drops to $S_{AV} = \mathbf{-0.0060}$ ($\Delta = \mathbf{+0.2678}$).
- **Forensic Principle**: Dubbed audio is treated as a desynchronization signal rather than automatically misclassified as a visual deepfake.

### C. Out-of-Distribution (OOD) Screen Capture Disentanglement
- **Technical Audit**: Verified 100% accuracy on in-distribution benchmark subset (16/16 correct).
- **OOD Mechanism**: Unseen external video misclassifications were caused by lossy desktop screen-recording compression smoothing high-frequency artifacts and sub-optimal facial ROI ($< 3.7\%$).
- **Agentic Solution**: Autonomous 12-criterion domain reliability layer flags `LOW RELIABILITY — POSSIBLE OUT-OF-DISTRIBUTION INPUT` without corrupting model predictions.

---

## 5. Master Evidence Index & Artifact Pointers
All detailed analyses, raw data tables, mathematical derivations, and limitations are organized in the following companion documents:
- `01_PROJECT_OVERVIEW.md`: System motivation, objectives, and high-level workflow.
- `02_FINAL_ARCHITECTURE.md`: Layer-by-layer specifications of all 6 encoders, aggregators, and heads.
- `03_PREPROCESSING_PIPELINE.md`: Step-by-step video, face, mouth ROI, and audio extraction parameters.
- `04_TRAINING_PIPELINE.md`: Optimization, Focal loss, WeightedRandomSampler, and caching protocol.
- `05_EXPERIMENT_HISTORY.md`: Chronological progression across Baseline, Class Weight, Sampler, and Phases 3–6.
- `06_FINAL_RESULTS.md`: Definitive numerical tables, confusion matrices, and per-class metrics.
- `07_SAME_TEST_SET_COMPARISON.md`: Controlled same-test-set head-to-head comparison (604 samples).
- `08_MISSING_AUDIO_ANALYSIS.md`: Quantitative fallback metrics, latency benchmarks, and stress tests.
- `09_DUBBED_AUDIO_ANALYSIS.md`: Cross-modal consistency formulation and dubbed audio experiment.
- `10_OOD_ANALYSIS.md`: Cross-dataset domain shift, screen capture diagnosis, and reliability layer.
- `11_AGENTIC_FORENSIC_SYSTEM.md`: Rule-based reasoning architecture, evidence synthesis, and report generator.
- `12_EQUATIONS.md`: Formal mathematical definitions and corresponding code references.
- `13_DATASET.md`: Benchmark source, manifest distributions, splitting, and imbalance statistics.
- `14_LIMITATIONS.md`: Academic disclosure of benchmark constraints, sample sizes, and OOD bounds.
- `15_RESEARCH_CONTRIBUTIONS.md`: Summary of core scientific and engineering contributions.
- `tables/`: 5 publication-ready CSV tables (`experiment_comparison.csv`, `architecture.csv`, `missing_audio.csv`, `consistency.csv`, `same_test_set.csv`).
- `SOURCE_INDEX.md`: Traceability index mapping every claim to source files and line numbers.
