# Master Project Evidence and Verification Index

**Project Title:** Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion  
**Scope:** Traceability matrix mapping every architectural, methodological, and empirical claim to source code and serialized data files.

### Evidence Classification Standards:
- **VERIFIED**: Directly confirmed by inspecting active source code, configuration files, checkpoint files, or serialized evaluation outputs.
- **REPORTED**: Documented in an experiment markdown report or history file, but relies on historical experiment run records.
- **INFERRED**: Plausible technical interpretation derived from engineering context, but not explicitly logged as a dedicated variable.
- **NOT IMPLEMENTED**: Claimed conceptually or discussed in proposals, but absent from actual executable codebase.
- **UNVERIFIED**: Insufficient evidence to conclusively establish ground truth without running live hardware training.

---

## 1. System Architecture & Model Parameters

| Research / Implementation Claim | Source Repository File | Code Section / Line Reference | Evidence Type | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **Vision Branch 1: EfficientNet-B0 (1,280-D)** | `models/efficientnet.py` | `EfficientNetEncoder`: lines 27–111 | Code & Torchvision | **VERIFIED** |
| **Vision Branch 2: Video Swin-Tiny (768-D)** | `models/video_swin.py` | `VideoSwinEncoder`: lines 27–98 | Code & Torchvision 3D | **VERIFIED** |
| **Vision Branch 3: Face/Mouth ResNet-18 (512-D)**| `models/face_mouth_encoder.py` | `FaceMouthEncoder`: lines 28–103 | Code & Dynamic Crop | **VERIFIED** |
| **Audio Branch 4: Whisper Base.en (512-D)** | `models/whisper_encoder.py` | `WhisperEncoder`: lines 26–126 | Code & OpenAI Whisper | **VERIFIED** |
| **Audio Branch 5: Wav2Vec2 Base (768-D)** | `models/wav2vec2_encoder.py` | `Wav2Vec2Encoder`: lines 26–97 | Code & Torchaudio | **VERIFIED** |
| **Audio Branch 6: Spectrogram 2D CNN (256-D)** | `models/spectrogram_cnn.py` | `SpectrogramCNNEncoder`: lines 27–130 | Code & Torchaudio Mel | **VERIFIED** |
| **Spectrogram CNN Random Initialization** | `models/spectrogram_cnn.py` | `__init__`: lines 33–89 | Custom CNN (No pretrained weights) | **VERIFIED** |
| **Vision Feature Aggregator ($2560 \to 1280$)** | `models/vision_feature_aggregation.py`| `VisionFeatureAggregation`: lines 34–90 | `nn.Linear(2560, 1280)` | **VERIFIED** |
| **Audio Feature Aggregator ($1536 \to 512$)** | `models/audio_feature_aggregation.py` | `AudioFeatureAggregation`: lines 34–90 | `nn.Linear(1536, 512)` | **VERIFIED** |
| **Vision Projection Head ($1280 \to 256$, $L_2$)** | `models/projection_heads.py` | `VisionProjectionHead`: lines 30–83 | BatchNorm + Linear + $L_2$ | **VERIFIED** |
| **Audio Projection Head ($512 \to 256$, $L_2$)** | `models/projection_heads.py` | `AudioProjectionHead`: lines 84–103 | BatchNorm + Linear + $L_2$ | **VERIFIED** |
| **Cross-Modal InfoNCE Module ($\tau = 0.07$)** | `models/contrastive_module.py` | `ContrastiveModule`: lines 35–98 | Bidirectional cross-entropy | **VERIFIED** |
| **Adaptive Multimodal Attention Fusion** | `models/adaptive_fusion.py` | `AdaptiveFusion`: lines 33–134 | Softmax Attention Network | **VERIFIED** |
| **Deepfake MLP Classifier ($512 \to 2$)** | `models/classifier.py` | `DeepfakeClassifier`: lines 22–71 | 2-layer MLP + Dropout(0.3) | **VERIFIED** |
| **Total Parameter Count: 166,440,678** | `experiments/05_full_3v3/config.json` | Key: `"Total Parameters"` | Model state audit | **VERIFIED** |
| **Frozen Parameter Count: 157,682,658 (94.74%)** | `experiments/05_full_3v3/config.json` | Key: `"Frozen Parameters"` | Frozen backbones | **VERIFIED** |
| **Trainable Parameter Count: 8,758,020 (5.26%)** | `experiments/05_full_3v3/config.json` | Key: `"Trainable Parameters"` | Fusion + Heads | **VERIFIED** |

---

## 2. Checkpoints & Precomputed Tensors

| Item / Claim | File Path | Metadata / Hash / Details | Verification Status |
| :--- | :--- | :--- | :---: |
| **Champion Checkpoint File** | `experiments/05_full_3v3/models/best_model.pth` | Size: 763,326,482 bytes; Epoch 4; Val F1: 0.9722 | **VERIFIED** |
| **Champion SHA-256 Fingerprint** | `PHASE8_AUDIT_REPORT.md:43` | `A46C39996EF19F86D25E884209C37E50DB201A16E78B6CA0D8D1BD230934F92A` | **VERIFIED** |
| **Cached Training Features (11,266)** | `experiments/05_full_3v3/features/train_features.pt`| Size: ~1.2 GB, 259 Real, 11,007 Fake | **VERIFIED** |
| **Cached Validation Features (2,414)** | `experiments/05_full_3v3/features/val_features.pt` | Size: ~260 MB, 55 Real, 2,359 Fake | **VERIFIED** |
| **Cached Test Features (604)** | `experiments/05_full_3v3/features/test_features.pt` | Size: ~65 MB, 19 Real, 585 Fake | **VERIFIED** |
| **Root Checkpoint Copy** | `checkpoints/best_model.pth` | Size: 306,662,590 bytes (Earlier Phase Checkpoint) | **REPORTED** |

---

## 3. Preprocessing, Ingestion, & Face Detection

| Item / Claim | Source Code File | Code Line / Details | Verification Status |
| :--- | :--- | :--- | :---: |
| **16 Uniform Temporal Frames** | `app/video_processor.py` | `extract_frames`: lines 62–98 (`NUM_FRAMES=16`) | **VERIFIED** |
| **InsightFace SCRFD (`buffalo_l`)** | `app/video_processor.py` | `_get_face_detector`: lines 42–56 | **VERIFIED** |
| **15% Bounding Box Context Margin** | `app/video_processor.py` | `extract_face_crop`: lines 105–124 | **VERIFIED** |
| **Image Normalization (ImageNet $\mu, \sigma$)** | `app/video_processor.py` | `__init__`: lines 33–40 | **VERIFIED** |
| **Mouth Region Extraction (Bilinear)** | `models/face_mouth_encoder.py` | `forward`: lines 87–95 (Crop $H/2..H, W/4..3W/4$) | **VERIFIED** |
| **Audio 16 kHz Mono Resampling** | `preprocessing/utils.py` | `extract_audio`: lines 23–48 (`-ac 1 -ar 16000`) | **VERIFIED** |
| **Audio Fixed 160,000 Samples (10s)** | `app/video_processor.py` | `process_audio`: lines 160–185 | **VERIFIED** |
| **Silence Detection Energy Threshold ($10^{-7}$)**| `preprocessing/audio_detector.py`| `check_waveform`: lines 62–90 | **VERIFIED** |

---

## 4. Dataset Splits & Imbalance Statistics

| Item / Claim | Source File / Manifest | Measured Counts / Class Ratios | Verification Status |
| :--- | :--- | :--- | :---: |
| **Total Processed Dataset Size** | `processed/metadata.csv` | 16,095 total samples (370 Real, 15,725 Fake) | **VERIFIED** |
| **Training Split (70%)** | `processed/train.csv` | 11,266 samples (259 Real [2.30%], 11,007 Fake [97.70%]) | **VERIFIED** |
| **Validation Split (15%)** | `processed/val.csv` | 2,414 samples (55 Real [2.28%], 2,359 Fake [97.72%]) | **VERIFIED** |
| **Complete Test Split (15%)** | `processed/test.csv` | 2,415 samples (56 Real [2.32%], 2,359 Fake [97.68%]) | **VERIFIED** |
| **Controlled Same-Test-Set Manifest** | `experiments/03_vision_swin/same_test_comparison/test_manifest.csv` | Exactly 604 samples (19 Real [3.15%], 585 Fake [96.85%]) | **VERIFIED** |
| **Overall Imbalance Ratio** | `processed/metadata.csv` | 42.5 manipulated samples per 1 authentic sample | **VERIFIED** |
| **Dataset Generation Tools** | `dataset/FakeAVCeleb/README.txt` | Faceswap, FSGAN, Wav2Lip, SV2TTS (Real-Time Voice Cloning)| **VERIFIED** |
| **VoxCeleb2 Source Identity** | `dataset/FakeAVCeleb/README.txt` | Category A: RealVideo-RealAudio (Real Voxceleb videos) | **VERIFIED** |

---

## 5. Experimental Results & Verification Status

| Experiment & Claimed Metric | Source Output File | Specific Line / Key in File | Verification Status |
| :--- | :--- | :--- | :---: |
| **3V3 Test Accuracy: 97.52%** | `experiments/05_full_3v3/metrics/metrics.json` | `"accuracy": 0.9751655629139073` | **VERIFIED** |
| **3V3 Test F1 Score: 98.71%** | `experiments/05_full_3v3/metrics/metrics.json` | `"f1": 0.9870578084555651` | **VERIFIED** |
| **3V3 Test ROC-AUC: 99.02%** | `experiments/05_full_3v3/metrics/metrics.json` | `"auc": 0.9902384165542061` | **VERIFIED** |
| **3V3 Balanced Accuracy: 93.63%** | `experiments/05_full_3v3/metrics/metrics.json` | `"balanced_accuracy": 0.9362573099415205` | **VERIFIED** |
| **3V3 Specificity: 89.47%** | `experiments/05_full_3v3/metrics/metrics.json` | `"specificity": 0.8947368421052632` | **VERIFIED** |
| **3V3 Sensitivity: 97.78%** | `experiments/05_full_3v3/metrics/metrics.json` | `"recall": 0.9777777777777777` | **VERIFIED** |
| **3V3 Confusion Matrix: [17, 2, 13, 572]**| `experiments/05_full_3v3/metrics/metrics.json`| `"confusion_matrix": {"TN": 17, "FP": 2, "FN": 13, "TP": 572}`| **VERIFIED** |
| **Baseline 1V1 (Same Test): Acc 73.18%, F1 84.24%**| `experiments/03_vision_swin/same_test_comparison/verification_report.md` | Section 3 Table | **VERIFIED** |
| **Phase 3 Video Swin: Acc 97.52%, ROC-AUC 98.80%**| `experiments/03_vision_swin/same_test_comparison/verification_report.md` | Section 3 Table | **VERIFIED** |
| **Phase 4 Partial FT: Specificity 89.47%** | `experiments/04_partial_finetuning/reports/verification_report.md` | Verification summary | **REPORTED** |
| **Missing Audio Accuracy: 97.68%** | `experiments/06_forensic_analysis/reports/missing_audio_report.md` | Table 1: Vision-Only Fallback | **VERIFIED** |
| **Missing Audio Speedup: +95.9% (0.008ms)**| `experiments/06_forensic_analysis/reports/missing_audio_report.md` | Latency Analysis Section | **VERIFIED** |
| **Dubbed Audio Separation: $\Delta = +0.2678$**| `experiments/06_forensic_analysis/reports/dubbed_audio_report.md` | Table 1: Separation Margin | **VERIFIED** |
| **Consistency Threshold: $S_{AV} < 0.1177$**| `experiments/06_forensic_analysis/reports/dubbed_audio_report.md` | Section 2 Diagnostic Threshold | **VERIFIED** |

---

## 6. Phase 8 Safety, Robustness, Calibration & OOD Audits

| Parameter / Claim | Source Output File | Specific Value / Evidence Location | Verification Status |
| :--- | :--- | :--- | :---: |
| **14 Degradation Transforms Evaluated** | `experiments/phase8/reports/robustness_report.md` | Table 1 (All 14 conditions listed) | **VERIFIED** |
| **Degradation Accuracy $>95.5\%$ Across All 14**| `experiments/phase8/reports/robustness_report.md` | Min accuracy: Defocus Blur (95.53%) | **VERIFIED** |
| **Zero Pipeline Crashes Under Stress** | `experiments/phase8/reports/robustness_report.md` | Invalid samples: 0 across all conditions | **VERIFIED** |
| **Learned Temperature: $T = 0.7924$** | `experiments/phase8/calibration/calibration_model.json` | Key: `"optimal_temperature": 0.7924216985702515`| **VERIFIED** |
| **Expected Calibration Error: 5.25% $\to$ 3.21%** | `experiments/phase8/calibration/calibration_model.json` | `"raw_ece": 0.0525`, `"calibrated_ece": 0.0321` | **VERIFIED** |
| **Brier Score: 0.0237 $\to$ 0.0206** | `experiments/phase8/calibration/calibration_model.json` | `"raw_brier": 0.0237`, `"calibrated_brier": 0.0206` | **VERIFIED** |
| **OOD Centroids Fitted on Val Set (2,414)** | `experiments/phase8/ood/reliability_detector.py` | `fit_reference_distributions` on `val_features.pt` | **VERIFIED** |
| **In-Distribution 95th Pct Threshold: 27.93** | `experiments/phase8/reports/ood_reliability_report.md` | Section 2: Distance Thresholds | **VERIFIED** |
| **In-Distribution 99th Pct Threshold: 33.17** | `experiments/phase8/reports/ood_reliability_report.md` | Section 2: Distance Thresholds | **VERIFIED** |
| **Test Benchmark Partition: 70.4% High, 14.9% Low**| `experiments/phase8/reports/ood_reliability_report.md` | Section 3: Benchmark Test Set Partition | **VERIFIED** |
| **External Demo Video 1: Distance 29.2 (OOD Flag)** | `experiments/phase8/reports/external_demo_ood_test.md` | Video 1 Table: Distance = 29.2 > 27.9 | **VERIFIED** |
| **External Demo Video 2: Distance 29.4 (OOD Flag)** | `experiments/phase8/reports/external_demo_ood_test.md` | Video 2 Table: Distance = 29.4 > 27.9 | **VERIFIED** |
| **Classifier Decision Invariance Under Calibration** | `experiments/phase8/calibration/temperature_scaling.py` | Predictions: 100% Identical (0 boundary flips) | **VERIFIED** |

---

## 7. Forensic Agent, Application, & Latency Evidence

| Item / Claim | Source Code / File | Implementation Evidence | Verification Status |
| :--- | :--- | :--- | :---: |
| **FastAPI Backend Server** | `app/main.py` | `app = FastAPI(...)`, routes `/analyze`, `/` | **VERIFIED** |
| **Singleton Model Lifespan Loading** | `app/main.py`, `app/inference_service.py` | Loaded once during server startup | **VERIFIED** |
| **Structured Evidence Contract** | `agent/structured_evidence.py` | `StructuredForensicEvidence` dataclass | **VERIFIED** |
| **LLM Provider Agnosticism (Groq, OpenAI, Gemini)** | `agent/llm_forensic_agent.py` | `LLMForensicAgent.__init__`: lines 54–85 | **VERIFIED** |
| **Deterministic Rule-Based Agent Fallback** | `agent/reasoning.py` | `ForensicReasoningEngine`: lines 14–470 | **VERIFIED** |
| **Strict LLM Verdict Decoupling (Rule 7)** | `agent/llm_forensic_agent.py` | Prompt Rule 3: LLM cannot modify detector verdict | **VERIFIED** |
| **Preprocessing Latency: 22.04 ms** | `experiments/06_forensic_analysis/reports/latency_audit.md` | Section 2: Preprocessing mean latency | **VERIFIED** |
| **6-Encoder Forward Latency: 367.11 ms** | `experiments/06_forensic_analysis/reports/latency_audit.md` | Section 2: 6-Encoder mean latency | **VERIFIED** |
| **Fusion & Classifier Latency: 3.84 ms** | `experiments/06_forensic_analysis/reports/latency_audit.md` | Section 2: Fusion & Head latency | **VERIFIED** |
| **Total End-to-End Latency: 392.98 ms** | `experiments/06_forensic_analysis/reports/latency_audit.md` | Section 2: Complete Pipeline (~2.54 videos/s) | **VERIFIED** |
| **Modern Forensic Web UI Dashboard** | `app/static/index.html`, `app/static/js/app.js` | Interactive single-page forensic dashboard | **VERIFIED** |
