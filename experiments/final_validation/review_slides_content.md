# Project Review Presentation Slides (15 Slides)

## Slide 1: Title Slide
- **Title**: Agentic Multimodal Deepfake Detection: A Robust 3-Vision + 3-Audio Framework
- **Subtitle**: Cross-Modal Spatio-Temporal Attention, Missing Audio Fallback, and Forensic Dubbing Analysis
- **Key Highlight**: Complete evaluation across 6 distinct pretrained backbones with 99.02% ROC-AUC.

---

## Slide 2: Problem & Motivation
- **Context**: Deepfakes have escalated from crude face-swaps to hyper-realistic multimodal manipulations.
- **Vulnerability**: Unimodal detectors fail against cross-modal synthesis or degraded real-world inputs.
- **Challenge**: Extreme class imbalance and real-world audio corruption (missing, silent, or out-of-sync audio).

---

## Slide 3: Existing Limitations & Baseline Collapse
- **Baseline Failure**: Standard 2V1A model without balance strategy suffered minority-class collapse (F1: 9.22%, Acc: 7.04%).
- **Key Insight**: Severe class imbalance (97.7% Fake vs 2.3% Real) requires class-aware sampling and robust temporal backbones.

---

## Slide 4: Proposed Architecture Overview (Full 3V3)
- **Vision Stream (3 Encoders)**: Spatial Texture + Spatio-temporal Coherence + Lip Articulation.
- **Audio Stream (3 Encoders)**: Speech Semantics + Acoustic Synthesis + Log-Mel Spectrogram.
- **Trainable Multimodal Core**: Aggregation, Projection Heads, InfoNCE Contrastive, and Adaptive Attention Fusion.

---

## Slide 5: The Three Vision Encoders
1. **EfficientNet-B0 (1280-D)**: High-resolution spatial artifacts, blending boundaries, edge distortion.
2. **Video Swin Transformer (768-D)**: 3D spatio-temporal self-attention across 16 uniform frames.
3. **Face/Mouth ResNet-18 (512-D)**: Dedicated 2D+1D spatio-temporal convolution on lip/mouth ROIs.

---

## Slide 6: The Three Audio Encoders
1. **Whisper-Base.en (512-D)**: Robust speech phonetic and semantic sequence modeling.
2. **Wav2Vec2-Base (768-D)**: Raw waveform acoustic synthesis artifacts and neural vocoder glitches.
3. **Spectrogram CNN (256-D)**: 128-band log-Mel time-frequency spectral anomalies.

---

## Slide 7: Adaptive Multimodal Fusion & InfoNCE Loss
- **Dual Attention**: Dynamically predicts modality trust weights (w_vision, w_audio) via Softmax attention.
- **Loss Function**: Focal Loss (gamma=2.0, alpha=0.25) + Cross-Modal InfoNCE Contrastive alignment (lambda=0.10).

---

## Slide 8: Training Strategy & Optimization
- **Freezing Policy**: 157.7M encoder parameters strictly frozen; 8.76M fusion parameters trained with AdamW.
- **Feature Precomputation**: 155-minute one-time caching enabled fast 8-second training epochs.
- **Numerical Stability**: FP32 Spectrogram log-Mel conversion and gradient clipping (`max_norm=1.0`).

---

## Slide 9: Experimental Milestones Evolution
- **Baseline**: F1 = 9.22%, AUC = 68.65%
- **Weighted Sampler**: F1 = 85.40%, AUC = 70.10%
- **Video Swin (Phase 3)**: F1 = 98.71%, AUC = 98.80% (+28.7% jump)
- **Full 3V3 (Phase 5)**: **F1 = 98.71%, AUC = 99.02%, Balanced Acc = 93.63%**

---

## Slide 10: Final 3V3 Test Results (604 Samples)
- **Accuracy**: **97.52%**
- **Precision**: **99.65%**
- **Recall**: **97.78%**
- **ROC-AUC**: **99.02%**
- **Sensitivity / Specificity**: **97.78% / 89.47%**

---

## Slide 11: Missing Audio Robustness (Experiment A)
- **Mechanism**: Automatically skips 3 audio encoders when audio is missing/silent/corrupted.
- **Performance**: Vision-only fallback retains **97.68% Accuracy** and **98.80% F1**.
- **Latency**: Vision-only mode is **95.9% faster** without software exceptions.

---

## Slide 12: Dubbing & Audio-Visual Inconsistency (Experiment B)
- **Cross-Modal Metric**: S_AV = cos(e_vision, e_audio).
- **Synchronized Speech**: Mean S_AV = +0.2618 +/- 0.1441.
- **Mismatched / Dubbed Speech**: Mean S_AV = -0.0060 +/- 0.1925 (Separation: **+0.2678**).
- **Core Principle**: Inconsistency is reported as separate forensic evidence, avoiding false deepfake labels on dubbed movies.

---

## Slide 13: Empirical Latency & Profiling Audit
- **Preprocessing (Decode/Crop)**: `22.04 ms`
- **6-Encoder Neural Extraction**: `367.11 ms`
- **GPU In-Memory Fusion**: `3.841 ms`
- **Total End-to-End Latency**: `392.98 ms / video` (`2.5 FPS`).

---

## Slide 14: Limitations & Future Directions
- **Class Imbalance**: Test split contains 19 Real samples out of 604.
- **Lip-Sync Granularity**: Embedding alignment is not phoneme-viseme level verification.
- **Future Work**: Scaling to multi-lingual dubbed datasets and real-time streaming pipelines.

---

## Slide 15: Conclusion & Key Takeaways
1. **Full 3V3 State of the Art**: 6 encoders provide complementary defense achieving 99.02% ROC-AUC.
2. **True Multimodal Resilience**: Vision-only fallback gracefully handles missing audio.
3. **Dual-Stream Forensics**: Disentangles deepfake classification from dubbing inconsistency.
