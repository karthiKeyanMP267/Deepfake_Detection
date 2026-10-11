# Project Overview: Agentic Multimodal Deepfake Detection

## 1. Project Identification
- **Title**: Agentic Multimodal Deepfake Detection using Audio-Visual Forensic Analysis
- **Domain**: Artificial Intelligence / Computer Vision / Speech Processing / Forensic Multimedia Analysis
- **Target Publication Venue**: IEEE Transactions on Information Forensics and Security / IEEE Access / IEEE Conference Style
- **Repository Location**: `D:\Deepfake\Agentic-Multimodal-Deepfake-Detection`
- **Primary Objective**: Formulate, implement, and benchmark an end-to-end multimodal deep learning framework incorporating 6 foundation backbones (3 visual + 3 acoustic) and an autonomous evidence-grounded forensic reasoning agent to detect complex audio-visual manipulations under severe class imbalance and cross-dataset distribution shifts.

---

## 2. Research Problem & Motivation
The rapid advancement of generative artificial intelligence models—such as diffusion-based image generators, neural radiance fields (NeRFs), Generative Adversarial Networks (GANs), neural vocoders (e.g., HiFi-GAN, WaveGlow), and autoregressive voice cloning systems (e.g., XTTS, ElevenLabs, VALL-E)—has democratized the creation of hyper-realistic digital forgeries. 

Contemporary deepfakes present multifaceted forensic challenges:
1. **Multimodal Manipulation**: State-of-the-art attacks alter both spatial facial textures (face-swapping, reenactment) and speech acoustics (voice cloning, audio synthesis), rendering unimodal detectors ineffective.
2. **Extreme Class Imbalance**: Real-world forensic datasets and operational pipelines exhibit severe class skewness (in FakeAVCeleb, 97.7% of samples are manipulated and only 2.3% are authentic), causing conventional unweighted classifiers to collapse entirely to the majority class.
3. **Stream Degradation & Missing Modality**: In real-world transmission channels, audio streams may be missing, muted, or heavily corrupted. Standard multimodal fusion architectures crash due to tensor shape mismatches or make erroneous predictions.
4. **Cross-Modal Inconsistency vs. Deepfake Confusion**: Dubbed videos, foreign film translations, or asynchronous streams exhibit acoustic-visual mismatch without containing any synthetic facial forgeries. Traditional detectors conflate desynchronization with generative deepfakes.
5. **Cross-Dataset Generalization Failure (Out-of-Distribution Shift)**: High benchmark performance fails to transfer to in-the-wild video recordings due to unseen compression artifacts (e.g., desktop screen capture), resolution disparities, and uncalibrated softmax confidence saturation.

---

## 3. Proposed Research Framework
To address these challenges, this research implements a unified 6-encoder multimodal architecture coupled with an autonomous forensic reasoning layer:

```
+----------------------------------------------------------------------------------------------------+
|                                    RAW INPUT MEDIA (MP4 VIDEO)                                     |
+----------------------------------------------------------------------------------------------------+
                                      |                             |
                       [Visual Stream: 16 Frames]       [Acoustic Stream: 10s @ 16kHz]
                                      |                             |
                                      v                             v
           +------------------------------------+   +------------------------------------+
           |          3 VISION ENCODERS         |   |          3 AUDIO ENCODERS          |
           | 1. EfficientNet-B0    (1280-D)     |   | 4. Whisper Base        (512-D)     |
           | 2. Video Swin (3D-T)  (768-D)      |   | 5. Wav2Vec2 Base       (768-D)     |
           | 3. Face/Mouth ResNet  (512-D)      |   | 6. Spectrogram CNN     (256-D)     |
           +------------------------------------+   +------------------------------------+
                                      |                             |
                                      v                             v
           +------------------------------------+   +------------------------------------+
           |      Vision Aggregator (2560->1280)|   |      Audio Aggregator (1536->512)  |
           |      Vision Projector (1280->256)  |   |      Audio Projector (512->256)    |
           +------------------------------------+   +------------------------------------+
                                      |                             |
                                      +--------------+--------------+
                                                     |
                                                     v
                                 +---------------------------------------+
                                 |       InfoNCE Contrastive Loss        |
                                 |       Audio-Visual Cosine S_AV        |
                                 +---------------------------------------+
                                                     |
                                                     v
                                 +---------------------------------------+
                                 |  Adaptive Multimodal Attention Fusion |
                                 |   alpha_v * e_v  ||  alpha_a * e_a    |
                                 |   (Fallback: alpha_v=1.0, alpha_a=0.0)|
                                 +---------------------------------------+
                                                     |
                                                     v
                                 +---------------------------------------+
                                 |      Deepfake MLP Classifier          |
                                 |    Softmax Output -> P(Real), P(Fake) |
                                 +---------------------------------------+
                                                     |
                                                     v
                                 +---------------------------------------+
                                 |      AUTONOMOUS FORENSIC AGENT        |
                                 |  - Grounded Visual & Audio Evidence   |
                                 |  - Cross-Modal Dubbing Disentanglement|
                                 |  - 12-Criterion OOD Reliability Layer |
                                 |  - Structured Forensic Report (MD/PDF)|
                                 +---------------------------------------+
```

---

## 4. Key Quantitative Achievements
- **Test Accuracy**: **97.52%** on the verified 604-sample test manifest.
- **Test F1-Score**: **98.71%** (Fake F1: 98.71%, Real F1: 69.39%).
- **Test ROC-AUC**: **99.02%** (Phase 5 Full 3V3).
- **Balanced Accuracy**: **93.63%** under 97.7% class imbalance.
- **Same-Test-Set Gain**: **+24.34% Accuracy** and **+38.81% ROC-AUC** over baseline.
- **Missing Audio Fallback**: Zero crashes, **97.68% accuracy**, +95.9% GPU fusion speedup.
- **Cross-Modal Consistency**: Clear separation margin ($\Delta = +0.2678$) between matched ($S_{AV} = +0.2618$) and dubbed speech ($S_{AV} = -0.0060$).
- **OOD Disentanglement**: Transparent forensic reliability scoring that flags screen-recording compression shifts while preserving in-distribution accuracy ($100.0\%$ on benchmark test subset).

---

## 5. Document Structure of the Evidence Package
- `00_MASTER_RESEARCH_SUMMARY.md`: High-level executive synthesis for rapid paper writing.
- `01_PROJECT_OVERVIEW.md`: Detailed introduction, motivation, scope, and system diagram.
- `02_FINAL_ARCHITECTURE.md`: Exhaustive layer-by-layer specification of all 6 encoders, aggregators, projectors, fusion, and classifier.
- `03_PREPROCESSING_PIPELINE.md`: Step-by-step video, facial ROI, and audio extraction pipeline with parameters.
- `04_TRAINING_PIPELINE.md`: Optimization parameters, loss formulations, class balancing, mixed precision, and parameter groups.
- `05_EXPERIMENT_HISTORY.md`: Chronological progression across Baseline, Class Weight, Sampler, Phase 3, Phase 4, Phase 5, and Phase 6.
- `06_FINAL_RESULTS.md`: Definitive numerical tables, confusion matrices, and per-class metrics.
- `07_SAME_TEST_SET_COMPARISON.md`: Strict head-to-head comparison on the identical 604-sample test set.
- `08_MISSING_AUDIO_ANALYSIS.md`: Quantitative fallback metrics, latency benchmarks, and edge-case stress tests.
- `09_DUBBED_AUDIO_ANALYSIS.md`: Cross-modal consistency formulation, dubbed audio experiment, and forensic separation.
- `10_OOD_ANALYSIS.md`: Cross-dataset domain shift investigation, screen capture compression diagnosis, and reliability layer.
- `11_AGENTIC_FORENSIC_SYSTEM.md`: Rule-based reasoning architecture, evidence synthesis, reliability scoring, and report generation.
- `12_EQUATIONS.md`: Formal mathematical definitions and corresponding code references for IEEE manuscript inclusion.
- `13_DATASET.md`: Benchmark source, manifest distributions, splitting methodology, and imbalance statistics.
- `14_LIMITATIONS.md`: Academic disclosure of benchmark constraints, minority sample sizes, and OOD bounds.
- `15_RESEARCH_CONTRIBUTIONS.md`: Summary of core scientific and engineering contributions.
- `SOURCE_INDEX.md`: Traceability index mapping every claim to source code files and line numbers.
- `tables/`: 5 publication-ready CSV tables.
