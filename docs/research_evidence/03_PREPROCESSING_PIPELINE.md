# Preprocessing Pipeline: Data Extraction, Normalization, & Routing

## 1. End-to-End Multimodal Data Flow
The multimodal deepfake detection pipeline transforms raw MP4 video containers into aligned spatio-temporal video tensors, lower-facial ROI tensors, and acoustic waveforms:

```
+-----------------------------------------------------------------------------------------+
|                                    RAW INPUT MP4 VIDEO                                  |
+-----------------------------------------------------------------------------------------+
                                 |                                   |
                  (Visual Processing Branch)          (Acoustic Processing Branch)
                                 |                                   |
                                 v                                   v
             +---------------------------------------+   +---------------------------------------+
             | cv2.VideoCapture Video Loading        |   | ffmpeg Audio Stream Extraction        |
             | Source: preprocessing/01_extract_...  |   | Source: preprocessing/02_extract_...  |
             +---------------------------------------+   +---------------------------------------+
                                 |                                   |
                                 v                                   v
             +---------------------------------------+   +---------------------------------------+
             | Uniform Temporal Frame Sampling       |   | Audio Channel & Sample Rate Handling  |
             | T = 16 frames: np.linspace(...)       |   | Mono Conversion; 16,000 Hz Resampling |
             +---------------------------------------+   +---------------------------------------+
                                 |                                   |
                                 v                                   v
             +---------------------------------------+   +---------------------------------------+
             | SCRFD Face Detection & Bounding Box   |   | Robust Audio Availability Verification|
             | InsightFace buffalo_l (det_size=640)  |   | Source: preprocessing/audio_detector  |
             | +15% Bounding Box Expansion           |   | Silence / NaN / File Size Checks      |
             +---------------------------------------+   +---------------------------------------+
                                 |                                   |
                                 v                                   v
             +---------------------------------------+   +---------------------------------------+
             | Frame Resizing & RGB Color Conversion |   | Waveform Temporal Normalization       |
             | cv2.resize((224, 224), INTER_AREA)    |   | 10-Second Pad/Trim (160,000 floats)   |
             +---------------------------------------+   +---------------------------------------+
                                 |                                   |
                                 v                                   v
             +---------------------------------------+   +---------------------------------------+
             | Facial ROI & Mouth Extraction         |   | Acoustic Modality Routing             |
             | Lower Face: rows [112..224], [56..168]|   | Pass to Whisper, Wav2Vec2, SpecCNN    |
             | Bilinear Interpolation to (224, 224)  |   | (Or trigger Vision-Only Fallback)     |
             +---------------------------------------+   +---------------------------------------+
                                 |                                   |
                                 v                                   v
             +---------------------------------------+   +---------------------------------------+
             | ImageNet Color Normalization          |   | Final Audio Tensor                    |
             | Mean: [0.485, 0.456, 0.406]           |   | Shape: (B, 160000), dtype=float32     |
             | Std:  [0.229, 0.224, 0.225]           |   +---------------------------------------+
             +---------------------------------------+
                                 |
                                 v
             +---------------------------------------+
             | Final Vision Tensors                  |
             | Full Face: (B, 16, 3, 224, 224)       |
             | Mouth ROI: (B, 16, 3, 224, 224)       |
             +---------------------------------------+
```

---

## 2. Visual Preprocessing Stages

### A. Video Ingestion & Frame Sampling
- **Source File**: `preprocessing/01_extract_frames.py` (`extract_frames`) and `preprocessing/utils.py` (`sample_frames`)
- **Mechanism**: Reads total frame count $N_{	ext{total}}$ via `cv2.VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT)`.
- **Sampling Logic**: Computes 16 uniformly spaced frame indices:
  $$	ext{indices} = 	ext{round}\left( 	ext{linspace}(0, N_{	ext{total}} - 1, 16) ight)$$
- **FPS Invariance**: Temporal sampling spans the complete duration of the video, ensuring invariant representation across 24, 25, 30, and 60 FPS sources.
- **Padding**: In `training/dataloader.py`, if a video contains fewer than 16 frames, the last frame is repeated to pad the tensor to exactly 16 frames.

### B. Facial Landmark Detection & Tracking
- **Source File**: `preprocessing/03_detect_faces.py` (`initialize_detector`, `detect_bbox`)
- **Detector**: InsightFace SCRFD (Sample and Computation Redistribution for Efficient Face Detection) with `buffalo_l` weights and ONNX runtime provider (CUDA or CPU).
- **Detection Resolution**: `det_size = (640, 640)`
- **Strategy**: Detects the largest face across sampled frames ($A = (x_2 - x_1) 	imes (y_2 - y_1)$).
- **Bounding Box Expansion**: Expands bounding box by 15% padding on all sides to retain forehead boundaries, chin, and hair margins without truncation:
  $$x_1' = \max(0, x_1 - 0.15 \cdot w), \quad y_1' = \max(0, y_1 - 0.15 \cdot h)$$
  $$x_2' = \min(W, x_2 + 0.15 \cdot w), \quad y_2' = \min(H, y_2 + 0.15 \cdot h)$$
- **Live Application Extension**: In `app/services.py`, per-frame dynamic tracking with previous-box fallback is implemented to prevent tracking loss during head rotations.

### C. Mouth / Perioral ROI Extraction
- **Source File**: `models/face_mouth_encoder.py` (`FaceMouthEncoder.forward`)
- **ROI Selection**: From the 224x224 normalized face crop, slices the lower-central quadrant:
  $$	ext{ROI} = 	ext{Frame}[112:224, 56:168] \in \mathbb{R}^{3 	imes 112 	imes 112}$$
- **Receptive Field Matching**: Resizes $(3, 112, 112)$ via bilinear interpolation (`F.interpolate`) to standard $(3, 224, 224)$ before feeding to ResNet-18.

### D. Image Normalization & Tensor Formatting
- **Source File**: `training/dataloader.py` (`DeepfakeDataset.__init__`)
- **Color Space**: BGR to RGB conversion via `Image.open().convert("RGB")`.
- **Normalization Formula**:
  $$I_{	ext{norm}}(c, x, y) = rac{I(c, x, y) / 255.0 - \mu_c}{\sigma_c}$$
  where $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$ (ImageNet standard).
- **Tensor Output**: Shape `(B, 16, 3, 224, 224)`, dtype `torch.float32`.

---

## 3. Acoustic Preprocessing Stages

### A. Audio Extraction & Codec Normalization
- **Source File**: `preprocessing/02_extract_audio.py` (`extract_audio`)
- **FFmpeg Execution Command**:
  ```bash
  ffmpeg -i <input_video.mp4> -ac 1 -ar 16000 -vn <output_audio.wav> -y
  ```
- **Channels**: Monophonic (`-ac 1`). If multi-channel audio is encountered during runtime, stereo channels are averaged: `waveform.mean(dim=0, keepdim=True)`.
- **Sampling Rate**: Resampled to exactly 16,000 Hz (`-ar 16000`).

### B. Waveform Alignment & Duration Normalization
- **Source File**: `training/dataloader.py` (`DeepfakeDataset.load_audio`)
- **Target Duration**: 10.0 seconds $\implies 160,000$ discrete float samples ($16,000 	imes 10$).
- **Padding**: If duration $< 10.0$ s, right-padded with zeros: `torch.nn.functional.pad(waveform, (0, 160000 - L))`.
- **Truncation**: If duration $> 10.0$ s, trimmed: `waveform[:160000]`.

### C. Robust Audio Availability Detection
- **Source File**: `preprocessing/audio_detector.py` (`AudioDetector`)
- **Verification Routine**:
  1. Checks if audio path is `None`, empty string, or non-existent file.
  2. Verifies `os.path.getsize(path) > 0`.
  3. Checks `torchaudio.load` successful decoding.
  4. Inspects waveform tensor: asserts `waveform.numel() > 0` and `waveform.shape[-1] > 0`.
  5. Asserts no `NaN` or `Inf` floating point values.
  6. Computes maximum absolute amplitude:
     $$	ext{is\_silent} = \max(|x|) < 1.0 	imes 10^{-7}$$
     If maximum amplitude $< 10^{-7}$, audio is marked `audio_available = False`.
- **Modality Fallback Trigger**: When `audio_available == False`, the audio tensor is replaced with a zero tensor of shape $(160000,)$, and the model automatically executes in **Vision-Only Fallback Mode**.

---

## 4. Summary of Preprocessing Parameters

| Parameter | Implemented Value | Source File Reference |
| :--- | :--- | :--- |
| **Video Frame Count ($T$)** | 16 frames | `training/dataloader.py:171-191` |
| **Spatial Resolution ($H 	imes W$)** | $224 	imes 224$ pixels | `preprocessing/config.py:34` |
| **Color Space** | RGB (ImageNet Normalized) | `training/dataloader.py:47-51` |
| **Face Detector** | InsightFace SCRFD `buffalo_l` | `preprocessing/03_detect_faces.py:53-62` |
| **Detection Size** | $640 	imes 640$ pixels | `preprocessing/03_detect_faces.py:61` |
| **BBox Expansion Factor** | +15% padding | `preprocessing/03_detect_faces.py:131-147` |
| **Mouth ROI Bounding Box** | Rows $[112..224]$, Cols $[56..168]$ | `models/face_mouth_encoder.py:87-94` |
| **Audio Sample Rate ($f_s$)** | 16,000 Hz | `preprocessing/config.py:37` |
| **Audio Channels** | 1 (Monophonic) | `preprocessing/config.py:36` |
| **Audio Length ($L$)** | 10.0 seconds (160,000 samples) | `training/dataloader.py:25, 98` |
| **Spectrogram FFT Size ($N_{	ext{fft}}$)** | 1024 points | `models/spectrogram_cnn.py:36` |
| **Spectrogram Hop Length** | 512 samples | `models/spectrogram_cnn.py:37` |
| **Mel Frequency Bins ($N_{	ext{mels}}$)** | 80 filterbanks | `models/spectrogram_cnn.py:38` |
| **Silence Detection Threshold** | $1.0 	imes 10^{-7}$ amplitude | `preprocessing/audio_detector.py:26` |
