# Forensic Deepfake Analysis Report

**Target Asset:** `video1_222131.mp4`  
**Analysis Timestamp:** `2026-08-28 22:50:03`  
**Architecture:** Full 3-Vision + 3-Audio Multimodal (6 Encoders)  
**Evaluation Mode:** Inference-Only (`torch.inference_mode()`)  

---

## 1. Executive Summary

| Attribute | Forensic Value |
| :--- | :--- |
| **AI Classifier Verdict** | **REAL** |
| **Classifier Confidence Score** | **96.47%** |
| **Probability (Real)** | 0.9647 |
| **Probability (Fake / Synthetic)** | 0.0353 |
| **Forensic Reliability Status** | **LOW RELIABILITY — POSSIBLE OUT-OF-DISTRIBUTION INPUT** |
| **Forensic Reliability Tier** | **LOW** |
| **Assessed Risk Level** | **MODERATE-UNCERTAIN** |
| **Audio Availability** | YES (Multi-stream Active) |
| **Cross-Modal Consistency ($S_{AV}$)** | 0.2419 (Synchronized / Coherent) |

---

## 2. Forensic Reliability & Observable Domain Characteristics

> **Forensic Assessment Distinction:**  
> The system strictly distinguishes between *Classifier Verdict* ("REAL") and *Forensic Reliability* ("LOW").

> [!WARNING]
> **OUT-OF-DISTRIBUTION INPUT NOTICE:**  
> The classifier predicts **REAL**, however the forensic reliability layer identifies characteristics outside the verified benchmark distribution (`FakeAVCeleb`). The classification should therefore be interpreted cautiously.

### Out-of-Distribution Indicators:
- Screen-captured desktop video format detected (multi-layer compression artifact smoothing).
- Small facial ROI (3.7% of frame area) limits high-frequency spatial artifact resolution.
- Non-standard aspect ratio (1.94) diverges from standard portrait training distribution.

### Observable Evidence Matrix:
| Parameter | Observed Characteristic | Benchmark Standard |
| :--- | :--- | :--- |
| **Input Spatial Resolution** | `1336x688 (Widescreen Desktop)` | Portrait / Square ($224 \times 224$) |
| **Aspect Ratio** | `1.94` | $1.00$ ($0.80 - 1.33$) |
| **Compression & Codec** | `H264 (Camera Video)` | Camera Direct Encode |
| **Frame Sharpness / Noise** | `302.0 (Sharp)` | High Uncompressed Detail |
| **Face / ROI Coverage** | `3.7% (Sub-optimal (<15%))` | > 25.0% of frame area |
| **Audio Channel Status** | `Active` | Synchronized Audio Track |
| **Audio Sample Rate** | `16,000 Hz (Standard Mono)` | $16,000$ Hz Mono |
| **Audio-Visual Consistency ($S_{AV}$)** | `0.2419` | >= 0.1177 (Aligned) |
| **Vision Modality Weight ($\alpha_v$)** | `61.3%` | Dynamic (50% - 75%) |
| **Audio Modality Weight ($\alpha_a$)** | `38.7%` | Dynamic (25% - 50%) |
| **Classifier Confidence** | `96.5%` | Calibrated Softmax |
| **Benchmark Alignment** | `Out-of-Distribution Shift` | `FakeAVCeleb` Verified |

---

## 3. Modality Contribution & Adaptive Fusion

The adaptive fusion mechanism weighted the complementary sensory branches as follows:

- **Visual Modality Weight ($\alpha_{vision}$):** `0.6133` (61.3%)
  - *Encoders:* EfficientNet-B0 (Spatial), Video Swin-Tiny (Temporal), Face/Mouth ResNet-18 (Articulation)
- **Audio Modality Weight ($\alpha_{audio}$):** `0.3867` (38.7%)
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

### Cross-Modal Consistency & Synchronization:
- **Consistency Score ($S_{AV}$):** `0.2419`
- **Threshold:** `0.1177` (Phase-6 Forensic Baseline)
- **Diagnostic:** Audio-visual embedding consistency score (0.2419) meets the synchronization threshold (>= 0.1177), indicating alignment between visual speech articulation and acoustic representations.

---

## 5. Agentic Reasoning & Synthesis

> The multimodal detection system classified this media as REAL with 96.5% confidence (authenticity probability: 0.9647). Multimodal feature fusion combined visual (61.3%) and acoustic (38.7%) cues, confirming natural biological motion and genuine acoustic resonance. OOD Notice: The classifier predicts REAL, however the forensic reliability layer identifies characteristics outside the verified benchmark distribution. The classification should therefore be interpreted cautiously.

---

## 6. Risk Assessment & Operational Recommendation

- **Assessed Risk Tier:** **MODERATE-UNCERTAIN**
- **Actionable Recommendation:** AI Classifier indicates REAL based on standard features, but input is OUT-OF-DISTRIBUTION (Screen Recording / Unseen Domain). Conduct secondary manual forensic inspection before concluding authenticity.

---
*Report automatically generated by Agentic Multimodal Deepfake Forensic Analyzer.*
