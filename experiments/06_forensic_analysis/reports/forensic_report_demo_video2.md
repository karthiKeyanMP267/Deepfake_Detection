# Forensic Deepfake Analysis Report

**Target Asset:** `video2_222513.mp4`  
**Analysis Timestamp:** `2026-08-28 22:50:11`  
**Architecture:** Full 3-Vision + 3-Audio Multimodal (6 Encoders)  
**Evaluation Mode:** Inference-Only (`torch.inference_mode()`)  

---

## 1. Executive Summary

| Attribute | Forensic Value |
| :--- | :--- |
| **AI Classifier Verdict** | **REAL** |
| **Classifier Confidence Score** | **98.00%** |
| **Probability (Real)** | 0.9800 |
| **Probability (Fake / Synthetic)** | 0.0200 |
| **Forensic Reliability Status** | **LOW RELIABILITY — POSSIBLE OUT-OF-DISTRIBUTION INPUT** |
| **Forensic Reliability Tier** | **LOW** |
| **Assessed Risk Level** | **MODERATE-UNCERTAIN** |
| **Audio Availability** | YES (Multi-stream Active) |
| **Cross-Modal Consistency ($S_{AV}$)** | 0.1625 (Synchronized / Coherent) |

---

## 2. Forensic Reliability & Observable Domain Characteristics

> **Forensic Assessment Distinction:**  
> The system strictly distinguishes between *Classifier Verdict* ("REAL") and *Forensic Reliability* ("LOW").

> [!WARNING]
> **OUT-OF-DISTRIBUTION INPUT NOTICE:**  
> The classifier predicts **REAL**, however the forensic reliability layer identifies characteristics outside the verified benchmark distribution (`FakeAVCeleb`). The classification should therefore be interpreted cautiously.

### Out-of-Distribution Indicators:
- Screen-captured desktop video format detected (multi-layer compression artifact smoothing).
- Small facial ROI (5.9% of frame area) limits high-frequency spatial artifact resolution.
- Non-standard aspect ratio (1.95) diverges from standard portrait training distribution.

### Observable Evidence Matrix:
| Parameter | Observed Characteristic | Benchmark Standard |
| :--- | :--- | :--- |
| **Input Spatial Resolution** | `1324x680 (Widescreen Desktop)` | Portrait / Square ($224 \times 224$) |
| **Aspect Ratio** | `1.95` | $1.00$ ($0.80 - 1.33$) |
| **Compression & Codec** | `H264 (Camera Video)` | Camera Direct Encode |
| **Frame Sharpness / Noise** | `203.2 (Sharp)` | High Uncompressed Detail |
| **Face / ROI Coverage** | `5.9% (Sub-optimal (<15%))` | > 25.0% of frame area |
| **Audio Channel Status** | `Active` | Synchronized Audio Track |
| **Audio Sample Rate** | `16,000 Hz (Standard Mono)` | $16,000$ Hz Mono |
| **Audio-Visual Consistency ($S_{AV}$)** | `0.1625` | >= 0.1177 (Aligned) |
| **Vision Modality Weight ($\alpha_v$)** | `57.0%` | Dynamic (50% - 75%) |
| **Audio Modality Weight ($\alpha_a$)** | `43.0%` | Dynamic (25% - 50%) |
| **Classifier Confidence** | `98.0%` | Calibrated Softmax |
| **Benchmark Alignment** | `Out-of-Distribution Shift` | `FakeAVCeleb` Verified |

---

## 3. Modality Contribution & Adaptive Fusion

The adaptive fusion mechanism weighted the complementary sensory branches as follows:

- **Visual Modality Weight ($\alpha_{vision}$):** `0.5699` (57.0%)
  - *Encoders:* EfficientNet-B0 (Spatial), Video Swin-Tiny (Temporal), Face/Mouth ResNet-18 (Articulation)
- **Audio Modality Weight ($\alpha_{audio}$):** `0.4301` (43.0%)
  - *Encoders:* Whisper (Semantic), Wav2Vec2 (Acoustic), Spectrogram CNN (Time-Frequency)
- **Fusion Mode:** `Adaptive Multimodal (3V3)`

---

## 4. Forensic Evidence Breakdown

### Visual Branch Evidence:
- Visual branch indicates natural facial texture fidelity with no high-frequency blending artifacts (EfficientNet-B0).
- Temporal frame-to-frame motion continuity is coherent and biologically consistent (Video Swin Transformer).
- Facial muscular movements and mouth ROI contours demonstrate natural articulation (Face/Mouth ResNet-18).

### Audio Branch Evidence:
- Acoustic spectrogram displays natural harmonic distribution and environmental reverberation (Spectrogram CNN).
- Phonetic timbre and speech cadence are consistent with natural human vocalization (Whisper + Wav2Vec2).
- Acoustic modality actively contributed to decision with 43.0% fusion attention weight.

### Cross-Modal Consistency & Synchronization:
- **Consistency Score ($S_{AV}$):** `0.1625`
- **Threshold:** `0.1177` (Phase-6 Forensic Baseline)
- **Diagnostic:** Audio-visual embedding consistency score (0.1625) meets the synchronization threshold (>= 0.1177), indicating alignment between visual speech articulation and acoustic representations.

---

## 5. Agentic Reasoning & Synthesis

> The multimodal detection system classified this media as REAL with 98.0% confidence (authenticity probability: 0.9800). Multimodal feature fusion combined visual (57.0%) and acoustic (43.0%) cues, confirming natural biological motion and genuine acoustic resonance. OOD Notice: The classifier predicts REAL, however the forensic reliability layer identifies characteristics outside the verified benchmark distribution. The classification should therefore be interpreted cautiously.

---

## 6. Risk Assessment & Operational Recommendation

- **Assessed Risk Tier:** **MODERATE-UNCERTAIN**
- **Actionable Recommendation:** AI Classifier indicates REAL based on standard features, but input is OUT-OF-DISTRIBUTION (Screen Recording / Unseen Domain). Conduct secondary manual forensic inspection before concluding authenticity.

---
*Report automatically generated by Agentic Multimodal Deepfake Forensic Analyzer.*
