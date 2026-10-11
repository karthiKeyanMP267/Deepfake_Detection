# Phase 6 Forensic Audio-Visual Analysis Research Report

## 1. Objective & Scope
This phase provides forensic validation of the Phase 5 Full 3V3 Multimodal Deepfake Detection system (EfficientNet-B0, Video Swin, Face/Mouth ResNet-18, Whisper, Wav2Vec2, Spectrogram CNN). This evaluation is **strictly inference-only** with all models and weights completely frozen.

The dual objectives of this investigation are:
1. **Missing / Degraded Audio Robustness**: Validate automatic vision-only fallback when audio is unavailable, silent, missing, or corrupted without software exceptions.
2. **Dubbing & Audio-Visual Inconsistency Forensics**: Evaluate cross-modal synchronization and separate visual manipulation detection from audio-visual mismatch evidence.

---

## 2. Methodology & Architecture
- **Checkpoint Evaluated**: `D:\Deepfake\Agentic-Multimodal-Deepfake-Detection\experiments\05_full_3v3\models\best_model.pth` (Phase 5 Full 3V3 best model, Epoch 4)
- **Test Benchmark**: Exact 604 test samples from `test_manifest.csv` (Real: 19, Fake: 585)
- **Training Mode**: `torch.inference_mode()` (No gradients, no fine-tuning, no optimizer steps)

---

## 3. Experiment A: Missing Audio Robustness & Vision-Only Fallback

When audio is unavailable (`audio_available = False`), the three audio encoders (Whisper, Wav2Vec2, Spectrogram CNN) and audio projection heads are bypassed. The adaptive fusion module automatically sets `vision_weight = 1.0, audio_weight = 0.0`.

### Quantitative Metrics Comparison:

| Metric | Normal Multimodal (3V3) | Vision-Only Fallback | Absolute Impact |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 97.52% | **97.68%** | +0.17% pp |
| **Precision** | 99.65% | **98.97%** | -0.68% pp |
| **Recall** | 97.78% | **98.63%** | +0.85% pp |
| **F1 Score** | 98.71% | **98.80%** | +0.10% pp |
| **ROC-AUC** | 99.02% | **98.57%** | -0.45% pp |
| **Balanced Accuracy** | 93.63% | **83.53%** | -10.10% pp |
| **Real F1** | 69.39% | **65.00%** | -4.39% pp |
| **Fake F1** | 98.71% | **98.80%** | +0.10% pp |

### Latency Comparison:
- **Normal Multimodal Latency**: 0.20 ms/sample
- **Vision-Only Fallback Latency**: 0.01 ms/sample (+95.9% faster)

---

## 4. Experiment B: Dubbing & Audio-Visual Inconsistency Forensics

To detect translated dubbing, speaker replacement, and cross-modal desynchronization, the system measures embedding-level alignment S_AV = cos(e_vision, e_audio).

### Audio-Visual Consistency Statistics:
- **Normal Matched Audio**: Mean S_AV = 0.2618 +/- 0.1441
- **Controlled Mismatched / Dubbed Audio**: Mean S_AV = -0.0060 +/- 0.1925
- **Consistency Separation Margin**: +0.2678
- **Exploratory Threshold**: S_AV < 0.1177 flags 71.69% of mismatched videos while only flagging 17.05% of naturally synchronized videos.

---

## 5. Important Scientific Distinctions & Limitations
1. **Inconsistency != Deepfake**: Translated voice-overs or movies are frequently authentic videos with mismatched audio. The forensic system reports **Deepfake Prediction** and **Audio-Visual Inconsistency** as separate outputs.
2. **Embedding Alignment vs. Phoneme Lip-Sync**: The consistency score measures cross-modal feature correlation; it is documented as *embedding-level audio-visual consistency* rather than fine-grained phoneme-viseme lip synchronization.
3. **Exploratory Thresholding**: The threshold S_AV < 0.1177 is documented as an exploratory research threshold.
