# Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.1-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SDG 9](https://img.shields.io/badge/UN%20SDG-9%3A%20Industry%2C%20Innovation%20%26%20Infrastructure-orange.svg)](https://sdgs.un.org/goals/goal9)

---

## 1. Executive Summary & Objective

Modern deepfakes employ generative adversarial networks (GANs), diffusion models, and neural vocoders to synthesize hyper-realistic facial reenactments, face swaps, and cloned voices. Conventional unimodal detectors frequently falter when single-modality cues are subtly manipulated or degraded.

This project implements an end-to-end **Agent-Assisted Multimodal Deepfake Detection System** combining:
1. **6 Specialized Neural Encoders**: A $3\times3$ architectural design leveraging 3 visual backbones and 3 acoustic backbones.
2. **Cross-Modal Consistency Learning**: Contrastive projection alignment capturing cross-modal dissonance (e.g., viseme-phoneme divergence and acoustic-visual desynchronization).
3. **Adaptive Modality Fusion**: Gated attention that dynamically adjusts modality weights ($w_{\text{vision}}, w_{\text{audio}}$) based on feature quality, with robust missing/silent audio handling.
4. **Agentic Forensic Reasoning**: An autonomous reasoning engine providing evidence-grounded, explainable forensic reports (local deterministic reasoning + optional LLM integration).
5. **Interactive Web Application**: A full-stack FastAPI and Vanilla CSS/JS forensic dashboard for real-time video investigation.

---

## 2. Problem Statement & Motivation

- **Multimodal Deception**: Attackers rarely limit attacks to a single modality. Videos with authentic visuals and cloned audio (or synthesized faces with real audio) bypass unimodal forensic systems.
- **Cross-Modal Desynchronization**: Generating temporally and phonetically coherent audio-visual deepfakes is difficult. Sub-perceptual temporal mismatches between mouth movements and phonetic cadence serve as key forensic markers.
- **Explainability Deficit**: Black-box classifiers outputting binary scalar probabilities are insufficient for legal, journalistic, and security workflows. Forensic analysts require interpretable evidence breakdowns.
- **Robustness Under Missing Modalities**: In real-world scenarios, videos often lack audio tracks or suffer from acoustic silence. Systems must gracefully handle missing modalities without catastrophic performance degradation.

---

## 3. UN Sustainable Development Goal (SDG 9) Relevance

This project aligns directly with **United Nations Sustainable Development Goal 9: Industry, Innovation, and Infrastructure**:
- **Target 9.c**: *Significantly increase access to information and communications technology and strive to provide universal and affordable access to the Internet.* By securing digital media integrity, this system strengthens trust in public communications infrastructure.
- **Target 9.5**: *Enhance scientific research, upgrade technological capabilities of industrial sectors.* Advances trustworthy machine learning and automated forensics against synthetic media disinformation threats.

---

## 4. System Architecture & Methodology

```
+---------------------------------------------------------------------------------------+
|                                    INPUT VIDEO (.mp4, .avi, etc.)                     |
+---------------------------------------------------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
            [Visual Pipeline]                             [Audio Pipeline]
       16 Sampled Frames (224x224)                  Extracted 16kHz Mono Audio
       Face Crop Extraction (RetinaFace)             Silence / Presence Detection
                    |                                             |
         +----------+----------+                       +----------+----------+
         |          |          |                       |          |          |
    EfficientNet VideoSwin  FaceMouth               Whisper    Wav2Vec2  Spectrogram
        -B0       -Tiny      ResNet18                           Base        CNN
      (1280-D)   (768-D)     (512-D)                (512-D)    (768-D)    (256-D)
         |          |          |                       |          |          |
         +----------+----------+                       +----------+----------+
                    |                                             |
        [Vision Aggregation: 1280-D]                   [Audio Aggregation: 768-D]
                    |                                             |
          +---------+---------+                         +---------+---------+
          |                   |                         |                   |
    Visual Projection   Vision Tokens              Audio Projection   Audio Tokens
       Head (256-D)                                  Head (256-D)           |
          |                                             |                   |
          +-------------> [Contrastive Module] <--------+                   |
                                  |                                         |
                         AV Consistency Score                               |
                            (tau = 0.1177)                                  |
                                                                            |
                                  +-----------------------------------------+
                                  |
                      [Adaptive Modality Fusion]
                 Gated Modality Weights (w_v, w_a)
                 Missing Audio Bypass (w_a = 0.0)
                                  |
                     [Fused Representation: 1024-D]
                                  |
                       [Deepfake Classifier]
                     P(Real), P(Fake), Verdict
                                  |
              +-------------------+-------------------+
              |                                       |
    [OOD Reliability Detector]             [Agentic Forensic Reasoner]
    Input Quality & Distribution Check     Grounded Evidence Synthesis
    (High / Medium / Low Trust)            Deterministic Engine + LLM
              |                                       |
              +-------------------+-------------------+
                                  |
                   [FastAPI REST API & Web Dashboard]
```

---

## 5. Model Components & Encoders

| Modality | Encoder | Architecture | Dimension | Forensic Role |
| :--- | :--- | :--- | :--- | :--- |
| **Vision** | EfficientNet-B0 | Convolutional | 1280-D | High-frequency spatial boundary & blending artifacts |
| **Vision** | VideoSwin-Tiny | Spatio-Temporal 3D Transformer | 768-D | Temporal flickering, unnatural motion, eye-blink dynamics |
| **Vision** | FaceMouth-ResNet18 | Residual CNN | 512-D | Localized lip-sync, mouth opening, facial landmark dynamics |
| **Audio** | OpenAI Whisper | Transformer Encoder | 512-D | Phonetic cadence, lexical alignment, speech representation |
| **Audio** | Wav2Vec2-Base | Self-Supervised Acoustic CNN | 768-D | Vocoder artifacts, phase irregularities, neural voice synthesis |
| **Audio** | Spectrogram-CNN | 2D Mel-Spectrogram CNN | 256-D | Spectral power distribution, acoustic frequency anomalies |
| **Fusion** | Contrastive Module | Projection MLP + Cosine | 256-D | Audio-Visual cross-modal synchronization score |
| **Fusion** | Adaptive Fusion | Softmax Gating Network | 1024-D | Dynamic modal reliability weighting & silent audio bypass |
| **Head** | Classifier | 2-Layer MLP Head | 2-Class | Binary classification (Real vs Fake) with calibrated confidence |

---

## 6. Preprocessing & Missing-Audio Handling

### 6.1 Visual Preprocessing
- **Frame Extraction**: 16 uniformly spaced frames are sampled across video duration.
- **Face Localization**: RetinaFace detects facial regions; crops are aligned and normalized to $224 \times 224$ pixels.
- **Normalization**: Pixel values are standardized using ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 6.2 Acoustic Preprocessing
- **Audio Extraction**: Audio tracks are extracted via `ffmpeg` as 16 kHz mono waveforms capped at 10 seconds.
- **Silence & Energy Detection**: `preprocessing/audio_detector.py` computes Root Mean Square (RMS) energy. Signals with mean energy below $-50\text{ dB}$ are flagged as silent.

### 6.3 Missing Audio Handling
In standard multimodal architectures, missing audio causes dimension mismatch or NaN gradients. Our system handles missing or silent audio natively:
- When no audio stream exists or audio is flagged as silent, the audio feature vector is zeroed.
- The **Adaptive Fusion Gating Layer** detects the missing modality flag and overrides weights:
  $$w_{\text{vision}} = 1.0, \quad w_{\text{audio}} = 0.0$$
- The classifier smoothly operates in a **vision-fallback mode** without throwing exceptions or generating hallucinations.

---

## 7. Dataset Specifications

- **Benchmark Dataset**: [FakeAVCeleb v1.2](https://github.com/DMS-Lab/FakeAVCeleb)
- **Dataset Composition**: Real celebrity interviews and multimodal deepfakes (Faceswap, Wav2Lip, SV2TTS vocoding).
- **Split Strategy**: Identity-disjoint splits across Train ($70\%$), Validation ($15\%$), and Test ($15\%$).

> **Important Notice on Data Files**:
> Raw video and audio files ($\sim 6.3\text{ GB}$) are **not included** in this GitHub repository due to storage and licensing constraints. To reproduce training or evaluation from scratch, download FakeAVCeleb v1.2 and place it in the `dataset/FakeAVCeleb/` folder as documented below.

---

## 8. Repository Structure

```
Agentic-Multimodal-Deepfake-Detection/
├── .env.example                               # Environment template for LLM configuration
├── .gitignore                                 # Exclusions for datasets, weights, caches, and raw logs
├── LICENSE                                    # MIT License
├── README.md                                  # Primary project overview and instructions
├── requirements.txt                           # Pinned Python package dependencies
│
├── agent/                                     # Agentic Forensic Reasoning Engine
│   ├── forensic_agent.py                      # Master Forensic Agent coordinator
│   ├── llm_forensic_agent.py                  # LLM reasoning client (Groq / OpenAI / Gemini / Fallback)
│   ├── reasoning.py                           # Local deterministic evidence reasoning engine
│   ├── report_generator.py                    # Markdown and structured report generator
│   └── structured_evidence.py                 # Forensic evidence schema definitions
│
├── app/                                       # FastAPI Web Application & Dashboard
│   ├── config.py                              # Application configuration & thresholds
│   ├── demo_service.py                        # Pre-indexed benchmark and OOD demo catalog
│   ├── inference_service.py                   # Singleton 6-encoder inference runner
│   ├── input_quality.py                       # Video quality analyzer (resolution, face size, aspect)
│   ├── main.py                                # FastAPI app and REST route definitions
│   ├── video_processor.py                     # Video frame extraction and audio ingestion
│   ├── temp_uploads/                          # Temporary upload staging (excluded from git)
│   └── static/                                # Frontend UI assets (HTML, CSS, JS)
│
├── models/                                    # Neural Network Architecture Definitions
│   ├── adaptive_fusion.py                     # Dynamic modality gating & missing-audio bypass
│   ├── audio_feature_aggregation.py           # Multi-audio encoder feature pooling
│   ├── base_vision_encoder.py                 # Abstract base class for vision encoders
│   ├── classifier.py                          # Deepfake binary classification head
│   ├── contrastive_module.py                  # Cross-modal AV consistency learning module
│   ├── efficientnet.py                        # EfficientNet-B0 vision backbone
│   ├── face_mouth_encoder.py                  # ResNet-18 lip and facial region encoder
│   ├── multimodal_model.py                    # Unified 6-encoder detector model
│   ├── projection_heads.py                    # MLP projection heads for contrastive space
│   ├── spectrogram_cnn.py                     # Mel-Spectrogram 2D CNN encoder
│   ├── video_swin.py                          # VideoSwin-Tiny 3D temporal transformer
│   ├── vision_encoder_factory.py              # Vision encoder instantiation factory
│   ├── vision_feature_aggregation.py          # Vision feature projection and pooling
│   ├── wav2vec2_encoder.py                    # Wav2Vec2 acoustic encoder
│   └── whisper_encoder.py                     # OpenAI Whisper speech encoder
│
├── preprocessing/                             # Dataset Preprocessing Pipeline
│   ├── 01_extract_frames.py                   # Uniform video frame extraction
│   ├── 02_extract_audio.py                    # 16kHz audio extraction
│   ├── 03_detect_faces.py                     # RetinaFace face detection and cropping
│   ├── 04_resize_faces.py                     # Face crop normalization to 224x224
│   ├── 05_create_metadata.py                  # Metadata CSV generation and dataset split
│   ├── audio_detector.py                      # Silence and missing audio detector
│   ├── config.py                              # Preprocessing configurations
│   └── utils.py                               # Image and audio preprocessing utilities
│
├── training/                                  # Model Training and Evaluation Modules
│   ├── dataloader.py                          # Multimodal PyTorch DataLoaders
│   ├── dataset.py                             # Audio-Visual dataset implementation
│   ├── evaluate.py                            # Evaluation metrics computation
│   ├── losses.py                              # Composite loss: Focal Loss + Contrastive Loss
│   ├── metrics.py                             # Accuracy, F1, ROC-AUC, EER calculations
│   └── train.py                               # Model training loop
│
├── tests/                                     # Automated Test Suite
│   ├── test_application.py                    # End-to-end API, pipeline & fallback tests
│   ├── test_llm_agent.py                      # Forensic agent reasoning unit tests
│   ├── smoke_test_model.py                    # Forward pass architecture smoke test
│   └── smoke_test_loss.py                     # Composite loss calculation smoke test
│
├── tools/                                     # Developer & Forensic Diagnostic Utilities
│   ├── audit_inference_investigation.py       # Deep-dive 6-encoder and label diagnostic audit
│   ├── audit_test_set.py                      # Test set feature evaluation with Fast3V3FusionModel
│   ├── compare_offline_vs_web.py              # Live vs offline precomputed feature comparator
│   ├── debug_inference_pipeline.py            # CLI video pipeline inspection diagnostic
│   ├── discover_experiments.py                # Multi-experiment metric discovery aggregator
│   ├── repair_spectrogram_cache.py            # Numerical stability spectrogram repair utility
│   ├── test_fusion_pipeline_diagnostics.py    # Gated fusion backward/gradient checker
│   └── verify_3v3_cached_features.py          # Feature tensor NaN/Inf audit utility
│
├── docs/                                      # Consolidated Project Documentation
│   ├── architecture.md                        # Detailed architecture, tensor shapes & equations
│   ├── experiments.md                         # Benchmark metrics, ablations & run guide
│   ├── limitations.md                         # Edge cases, OOD diagnostics & ethical scope
│   ├── references.md                          # Academic bibliography & module citations
│   ├── api_documentation.md                   # REST API endpoint reference
│   ├── demo_walkthrough.md                    # User guide and demonstration walkthrough
│   ├── final_report_summary.md                # Comprehensive final-year project summary
│   └── research_evidence/                     # Complete empirical research notes and tables
│
├── experiments/                               # Experiment Runner Scripts & Results
│   └── phase8/                                # Phase 8 Calibration & OOD Reliability
│       ├── calibration/calibration_model.json # Temperature scaling calibration parameters
│       ├── ood/reference_stats.pt             # In-distribution validation feature centroids
│       └── ood/reliability_detector.py        # OOD & Reliability classifier module
│
├── run_phase5_full_3v3_experiment.py          # Phase 5 full 3v3 training orchestration
├── run_phase6_forensic_analysis.py            # Phase 6 AV consistency & dubbed evaluation
└── run_final_project_validation.py            # Final end-to-end project validation script
```

---

## 9. Installation & Environment Setup

### 9.1 Prerequisites
- Python **3.10**, **3.11**, or **3.12**
- NVIDIA GPU with CUDA 11.8+ / 12.1+ recommended for real-time inference (CPU fallback supported)
- `ffmpeg` installed on your system PATH (required for audio extraction)

### 9.2 Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/karthiKeyanMP267/Deepfake_Detection.git
cd Deepfake_Detection

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 9.3 Environment Configuration (`.env`)

To enable optional LLM-assisted natural language explanations, copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your preferred provider:

```ini
# Optional: Provide API key for LLM explanations (Groq, OpenAI, or Gemini)
LLM_API_KEY=your_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile
```

> **Note**: An API key is **not required** to use the application. If no key is set, the system automatically uses its built-in local deterministic reasoning engine.

---

## 10. Model Checkpoints & Pre-Trained Weights

The full multimodal detector expects trained weights located at:
```
experiments/05_full_3v3/models/best_model.pth
```
- **Size**: $\sim 728\text{ MB}$
- **Architecture**: 6-encoder multimodal fusion model ($3\times3$) with contrastive projection heads and adaptive gating.

### Checkpoint Storage & Acquisition
Because this checkpoint exceeds GitHub's 100 MB file limit, model weights are excluded from Git tracking. You can obtain the weights via:
1. **Direct Download**: Download the trained checkpoint from the project's [GitHub Releases / Model Hub Link](https://github.com/karthiKeyanMP267/Projectwork1/releases) and place it in `experiments/05_full_3v3/models/best_model.pth`.
2. **Train from Scratch**: Run the Phase 5 training pipeline as described in Section 11.
3. **Instant Demo Mode**: The web dashboard includes **pre-indexed benchmark cases** that run instantly without downloading model weights.

---

## 11. Pipeline Execution Guide

### 11.1 Dataset Preparation (FakeAVCeleb)
Place the raw dataset in `dataset/FakeAVCeleb/` and run the preprocessing pipeline:

```bash
# 1. Extract frames
python preprocessing/01_extract_frames.py

# 2. Extract audio waveforms
python preprocessing/02_extract_audio.py

# 3. Detect and crop faces
python preprocessing/03_detect_faces.py

# 4. Resize and normalize face crops
python preprocessing/04_resize_faces.py

# 5. Generate metadata manifests and train/val/test splits
python preprocessing/05_create_metadata.py
```

### 11.2 Training
Train the full 6-encoder multimodal model:

```bash
python run_phase5_full_3v3_experiment.py
```
Checkpoints will be saved automatically to `experiments/05_full_3v3/models/`.

### 11.3 Evaluation & Forensic Analysis
Evaluate AV consistency and dubbed-audio scenarios:

```bash
python run_phase6_forensic_analysis.py
```

---

## 12. Launching the Web Application

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
```
http://localhost:8000
```

### Key API Endpoints
- `GET /api/health` — Backend status, device (CUDA/CPU), checkpoint status, and encoder details.
- `GET /api/demo/samples` — Retrieve pre-indexed demonstration cases.
- `POST /api/demo/analyze/{sample_id}` — Analyze a pre-indexed benchmark case.
- `POST /api/analyze` — Upload a video file (`.mp4`, `.avi`, `.mov`, `.webm`) for full inference.

---

## 13. Running Automated Tests

Run the test suite to verify pipeline integrity:

```bash
# Test forensic reasoning agent and LLM fallback
python -m unittest tests/test_llm_agent.py

# Test application endpoints, audio bypass, and quality checks
python -m unittest tests/test_application.py
```

---

## 14. Key Benchmark Results

Evaluated on the standardized identity-disjoint test set ($N = 604$ videos) on FakeAVCeleb v1.2:

| Metric | Baseline (2-Encoder) | Proposed Champion (6-Encoder Full 3V3) | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 73.18% | **97.52%** | **+24.34%** |
| **F1 Score** | 0.8451 | **0.9871** | **+0.1420** |
| **ROC-AUC** | 0.6021 | **0.9902** | **+0.3881** |
| **Balanced Accuracy** | 50.00% | **93.63%** | **+43.63%** |
| **Authentic Class F1** | 10.00% | **69.39%** | **+59.39%** |
| **Manipulated Class Recall** | 98.63% | **97.78%** | High sensitivity |
| **AV Consistency Separation ($\tau = 0.1177$)** | N/A | **93.4% Dubbed Detection** | Novel Capability |
| **Missing Audio Fallback Accuracy** | Fails (NaN) | **97.68% (Vision Fallback)** | Non-crashing bypass |

*For complete experimental breakdowns, confusion matrices, ablation histories, and mathematical formulations, see [docs/experiments.md](docs/experiments.md) and [docs/architecture.md](docs/architecture.md).*

---

## 15. Limitations & Responsible Use

### Generalization & Out-of-Distribution (OOD) Data
The model was trained on FakeAVCeleb v1.2. Videos originating from heavily compressed social media feeds, screen recordings, non-standard aspect ratios, or unconventional lighting conditions may exhibit distribution shift.
- The included **OOD Reliability Detector** (`experiments/phase8/ood/reliability_detector.py`) checks input quality and flags low-reliability assessments with appropriate cautionary warnings.
- For detailed technical analysis of known failure modes and domain shift, see [docs/limitations.md](docs/limitations.md).

### Forensic Evidentiary Notice
This tool is intended as a **decision-support instrument** for researchers, journalists, and forensic analysts. It should not be used as the sole determinant in legal or punitive decisions without human-in-the-loop verification.

---

## 16. Citation & License

This project is licensed under the [MIT License](LICENSE).

```bibtex
@article{multimodal_deepfake_detection_2026,
  title={Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion},
  author={Karthikeyan, M. P.},
  year={2026},
  journal={GitHub Repository},
  url={https://github.com/karthiKeyanMP267/Deepfake_Detection}
}
```
