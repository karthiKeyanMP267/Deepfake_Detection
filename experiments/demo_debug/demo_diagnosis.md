# Complete Demo Troubleshooting & Pipeline Forensic Diagnosis Report

**Audit Target:** Full 3V3 Multimodal Deepfake Detector & Demonstration Ingestion Pipeline  
**Audited Checkpoint:** `experiments/05_full_3v3/models/best_model.pth`  
**Checkpoint Size:** 763,326,482 bytes (Epoch 4, Validation F1: 0.9722)  
**Investigation Timestamp:** 2026-08-28 22:42  
**Investigation Scope:** Autonomous investigation of 2 uploaded demonstration videos (`Screen Recording 2026-08-28 222131.mp4` and `Screen Recording 2026-08-28 222513.mp4`).

---

## 1. Demonstration Video Properties & Forensic Inspection

Both demonstration assets are Windows desktop screen recordings capturing external video playback:

| Property | Video 1 (`video1_222131.mp4`) | Video 2 (`video2_222513.mp4`) |
| :--- | :--- | :--- |
| **Original Filename** | `Screen Recording 2026-08-28 222131.mp4` | `Screen Recording 2026-08-28 222513.mp4` |
| **File Size** | 14,022,064 bytes (13.37 MB) | 13,344,564 bytes (12.73 MB) |
| **Spatial Resolution** | $1336 \times 688$ px (Aspect Ratio: 1.94) | $1324 \times 680$ px (Aspect Ratio: 1.95) |
| **Frame Rate / Total Frames** | 30.00 FPS / 818 frames | 30.00 FPS / 801 frames |
| **Duration** | 27.27 seconds | 26.70 seconds |
| **Video Codec** | H.264 / AVC (`h264`) | H.264 / AVC (`h264`) |
| **Audio Stream** | Stereo 2-channel, 48,000 Hz | Stereo 2-channel, 48,000 Hz |
| **Audio Duration** | 27.37 seconds (max amp: 0.1084) | 26.84 seconds (max amp: 0.1129) |
| **Face Area vs Screen Area** | **2.59% - 3.70% of screen area** | **5.90% - 10.34% of screen area** |
| **Scene Layout** | Small video window in desktop recording | Multi-face shot with scene cuts & titles |

---

## 2. Frame Extraction & Visual Contact Sheet Analysis

High-resolution $4 \times 4$ contact sheets were generated for both raw uniform frame extraction and face-cropped model inputs:

- `experiments/demo_debug/frames/video1_frames.png` (Raw sampled desktop frames)
- `experiments/demo_debug/frames/video1_face_frames.png` (16 face-cropped tensors)
- `experiments/demo_debug/frames/video2_frames.png` (Raw sampled desktop frames)
- `experiments/demo_debug/frames/video2_face_frames.png` (16 face-cropped tensors)

### Observations:
1. **Video 1**: The face is small (~137 px wide) and embedded in an operating system desktop window. In several frames, secondary background faces and screen UI elements are visible.
2. **Video 2**: Scene cuts occur across the 26-second recording. In frames 6 and 7, up to 6 faces are visible (thumbnails/group shot); in frames 8 and 12, the face turns or is occluded.

---

## 3. Preprocessing Verification against Training Pipeline

A programmatic comparison of the live application pipeline against the training/feature-cache pipeline confirmed full alignment:

```
Preprocessing Stage        Training Pipeline                     Live Application Pipeline             Status
-------------------------------------------------------------------------------------------------------------
Temporal Sampling          16 uniform frames over duration       16 uniform frames over duration       MATCH
Frame Count                16 frames (B, 16, 3, 224, 224)        16 frames (B, 16, 3, 224, 224)        MATCH
Frame Ordering             Chronological (0..T-1)                Chronological (0..T-1)                MATCH
OpenCV Decoding            cv2.VideoCapture -> RGB               cv2.VideoCapture -> RGB               MATCH
RGB / BGR Conversion       cv2.COLOR_BGR2RGB                     cv2.COLOR_BGR2RGB                     MATCH
Resize Resolution          (224, 224)                            (224, 224)                            MATCH
Contextual BBox Padding    15% (padx=0.15*bw, pady=0.15*bh)      15% (padx=0.15*bw, pady=0.15*bh)      MATCH
ImageNet Normalization     mean=[.485,.456,.406], std=[.229,.224,.225]  mean=[.485,.456,.406], std=[.229,.224,.225] MATCH
Face Detector Model        InsightFace SCRFD buffalo_l           InsightFace SCRFD buffalo_l           MATCH
Audio Extraction           16,000 Hz Mono                        16,000 Hz Mono (imageio-ffmpeg)       MATCH
Audio Waveform Length      160,000 samples (10 seconds)          160,000 samples (10 seconds)          MATCH
Spectrogram Mel Bins       80 Mel Bins, FFT 1024, Hop 512        80 Mel Bins, FFT 1024, Hop 512        MATCH
Whisper Preprocessing      whisper.log_mel_spectrogram(80, 3000) whisper.log_mel_spectrogram(80, 3000) MATCH
Wav2Vec2 Preprocessing     Raw 16kHz waveform -> extract_feat    Raw 16kHz waveform -> extract_feat    MATCH
```

---

## 4. Checkpoint & State Verification

- **Checkpoint File:** `experiments/05_full_3v3/models/best_model.pth`
- **Total Parameters:** 166,440,678 (Trainable: 9,212,612; Frozen: 157,228,066)
- **State Dict Load Status:** **STRICT MATCH** (0 missing keys, 0 unexpected keys)
- **Validation Metric:** Epoch 4, Validation F1: 0.9722

---

## 5. Label Mapping Independent Proof

- `preprocessing/05_create_metadata.py`: `get_label(folder_name)` sets `label = 1` for `"Fake"`, `label = 0` for `"Real"`.
- `models/classifier.py`: `num_classes = 2`, `Linear(128, 2)` produces logits $[z_0, z_1]$.
  - $z_0$ represents REAL, $z_1$ represents FAKE.
- `app/inference_service.py`: `probs = softmax([z_0, z_1])`, `prediction = 'FAKE' if probs[1] >= 0.50 else 'REAL'`.
- **Verdict:** **PASS**. Verified mathematically and logically; no label inversion exists.

---

## 6. Six-Encoder Output Integrity Diagnostics

Saved in: `experiments/demo_debug/encoder_diagnostics.csv`

| Video | Encoder | Expected Dim | NaNs | Infs | Zero Vector? | $L_2$ Norm | Mean | Std | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Video 1** | EfficientNet-B0 | 1280 | 0 | 0 | False | 11.1613 | 0.1791 | 0.2554 | **PASS** |
| **Video 1** | Video Swin-Tiny | 768 | 0 | 0 | False | 6.9435 | 0.0023 | 0.2505 | **PASS** |
| **Video 1** | Face/Mouth ResNet | 512 | 0 | 0 | False | 18.3709 | 0.6351 | 0.5057 | **PASS** |
| **Video 1** | Whisper Base | 512 | 0 | 0 | False | 18.9915 | -0.0165 | 0.8392 | **PASS** |
| **Video 1** | Wav2Vec2 Base | 768 | 0 | 0 | False | 5.8749 | -0.0021 | 0.2120 | **PASS** |
| **Video 1** | Spectrogram CNN | 256 | 0 | 0 | False | 2.3544 | -0.0026 | 0.1471 | **PASS** |
| **Video 2** | EfficientNet-B0 | 1280 | 0 | 0 | False | 7.7192 | 0.0912 | 0.1956 | **PASS** |
| **Video 2** | Video Swin-Tiny | 768 | 0 | 0 | False | 9.3958 | 0.0011 | 0.3390 | **PASS** |
| **Video 2** | Face/Mouth ResNet | 512 | 0 | 0 | False | 16.7981 | 0.5923 | 0.4476 | **PASS** |
| **Video 2** | Whisper Base | 512 | 0 | 0 | False | 18.9453 | -0.0171 | 0.8371 | **PASS** |
| **Video 2** | Wav2Vec2 Base | 768 | 0 | 0 | False | 6.0271 | -0.0021 | 0.2175 | **PASS** |
| **Video 2** | Spectrogram CNN | 256 | 0 | 0 | False | 2.3297 | -0.0027 | 0.1456 | **PASS** |

---

## 7. Direct Model Inference & Modality Breakdown

```
Video 1 (222131):
  Logit[0] (REAL)       : +1.8575
  Logit[1] (FAKE)       : -1.4495
  Probability (REAL)    : 96.47%
  Probability (FAKE)    : 3.53%
  Model Prediction      : REAL (Confidence: 96.47%)
  Vision Weight (α_v)   : 61.33%
  Audio Weight  (α_a)   : 38.67%
  Consistency S_AV      : 0.1879
  Vision-Only P(Fake)   : 3.91%

Video 2 (222513):
  Logit[0] (REAL)       : +1.9588
  Logit[1] (FAKE)       : -1.9321
  Probability (REAL)    : 98.00%
  Probability (FAKE)    : 2.00%
  Model Prediction      : REAL (Confidence: 98.00%)
  Vision Weight (α_v)   : 56.99%
  Audio Weight  (α_a)   : 43.01%
  Consistency S_AV      : 0.2292
  Vision-Only P(Fake)   : 1.95%
```

---

## 8. Known-Good In-Distribution Test Set Verification

Evaluated 5 known REAL and 5 known FAKE samples from the benchmark test set:

```
Sample                                            Ground Truth   Predicted   P(Real)   P(Fake)   Confidence   Match
-------------------------------------------------------------------------------------------------------------------
00056.mp4                                         REAL           REAL        0.9021    0.0979    90.21%       PASS
00072.mp4                                         REAL           REAL        0.9988    0.0012    99.88%       PASS
00074.mp4                                         REAL           REAL        0.9396    0.0604    93.96%       PASS
00174.mp4                                         REAL           REAL        0.6131    0.3869    61.31%       PASS
00417.mp4                                         REAL           REAL        0.9412    0.0588    94.12%       PASS
00080_id00568_gakhcJXnJZ8.mp4                     FAKE           FAKE        0.0593    0.9407    94.07%       PASS
00109_id01171_XSINMRoTb0E.mp4                     FAKE           FAKE        0.0775    0.9225    92.25%       PASS
00017_id01972_hBYoeQTm8ZQ_id01610_wavtolip.mp4     FAKE           FAKE        0.0256    0.9744    97.44%       PASS
00160_2_id01182_wavtolip.mp4                     FAKE           FAKE        0.0143    0.9857    98.57%       PASS
00073_id06467_bN8jVda0Byw_id09053_wavtolip.mp4     FAKE           FAKE        0.0331    0.9669    96.69%       PASS
```
- **Known-Good Accuracy:** **100.00%** (10/10 correct)

---

## 9. Root Cause: Out-of-Distribution Screen-Recording Domain Shift

The false negative on the demonstration videos is caused by **Cross-Dataset & Compression Domain Shift**:

1. **Secondary Desktop Screen Capture Compression**:
   - Both demo videos were recorded using Windows Screen Recording ($1336 \times 688$ and $1324 \times 680$, 30 FPS H.264).
   - Screen-recording encoders apply lossy block-based quantization over the entire desktop surface. This spatial re-encoding effectively smooths out the fine-grained high-frequency boundary warping and lip-sync blending artifacts that the 3 vision encoders (EfficientNet-B0, Video Swin, Face/Mouth ResNet) rely upon.
2. **Sub-Optimal Facial Resolution**:
   - In Video 1, the face occupies only $2.59\% - 3.70\%$ of the screen area (~$137 \times 174$ pixels). When resized to $224 \times 224$, upsampling blur further washes out generator noise artifacts.
3. **Acoustic Vocoder Domain Mismatch**:
   - The audio in both recordings is stereo system audio at 48,000 Hz, lacking the synthetic SV2TTS low-pass filter boundaries learned from FakeAVCeleb. Both audio encoders (Whisper, Wav2Vec2) and the Spectrogram CNN find no unnatural phase jumps or spectral ringing $\to$ registering natural speech.
4. **Softmax Logit Saturation on OOD Samples**:
   - Neural network classifiers trained on in-distribution features output high logit margins when features project into clean, unperturbed regions of the embedding space, producing $96.47\%$ and $98.00\%$ confidence.

---

## 10. Implemented Safe Fixes & Demo-Safe Architecture

To resolve pipeline vulnerabilities without retraining or modifying weights:

1. **Dynamic Per-Frame Face Tracking with Fallback**:
   - Enhanced `VideoProcessor.process_video_frames` to dynamically track the largest detected face on each individual frame, with fallback to previous valid bounding boxes for occlusions and scene cuts.
2. **Autonomous Forensic Warning & OOD Reliability Layer**:
   - Enhanced `ForensicReasoningEngine` and `ForensicAgent` with automatic domain reliability assessment (`evaluate_domain_reliability`).
   - When screen recordings, small face ratios ($< 15\%$), or non-standard aspect ratios are detected:
     - The AI Classifier Verdict remains 100% genuine and unaltered: `REAL (96.47%)`.
     - The Forensic Status clearly displays: `LOW RELIABILITY — POSSIBLE OUT-OF-DISTRIBUTION INPUT`.
     - Actionable forensic evidence explains why secondary manual inspection is recommended.
3. **UI Integration**:
   - Updated `index.html` and `app.js` to render the `Forensic Reliability Status` badge prominently alongside the AI Classifier verdict.
