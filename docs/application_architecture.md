# Application Architecture Documentation

**System Title:** Agentic Multimodal Deepfake Detection using Audio-Visual Forensic Analysis  
**Framework:** FastAPI + PyTorch + Uvicorn + Vanilla JS/CSS Single-Page Web Interface  
**Checkpoint Evaluated:** `experiments/05_full_3v3/models/best_model.pth` (Phase 5 Best Model, 6 Encoders)  

---

## 1. System Topology & Architecture

The application is structured into four decoupled layers:

```
┌─────────────────────────────────────────────────────────────┐
│                 Frontend UI (Web Browser)                   │
│   [User Analysis View]  [Agent Pipeline]  [Architecture]    │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST API (JSON / Multipart)
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Server                   │
│   • /api/health    • /api/analyze    • /api/demo/*          │
│   • Request Validation & Temporary File Lifecycle           │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│           Video & Audio Preprocessing Service               │
│   • OpenCV Video Frame Sampling (16 Uniform Frames)         │
│   • InsightFace / SCRFD Face Detection & Bounding Box Crop  │
│   • ImageNet Normalization (224 x 224 x 3)                  │
│   • 16 kHz Mono Audio Waveform Extraction & AudioDetector   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│             Singleton Multimodal Inference Service          │
│   • Pre-loaded MultimodalDeepfakeDetector (Loaded Once)     │
│   • Vision Branch (EffNet + Swin-T + FaceMouth ResNet)      │
│   • Audio Branch (Whisper + Wav2Vec2 + Spectrogram CNN)     │
│   • Cross-Modal Consistency S_AV = cos(e_v, e_a)            │
│   • Adaptive Modality Attention Fusion                      │
│   • Classification Head (512 -> 2)                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│           Autonomous Forensic Reasoning Agent Layer         │
│   • Evidence Extraction & Feature Grounding                 │
│   • Deterministic Cross-Modal Narrative Synthesis           │
│   • Risk Tier Assessment & Actionable Recommendations       │
│   • Markdown Forensic Report Generator                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Ingestion & Preprocessing Pipeline

1. **Video Decoding & Sampling**:
   - The video is decoded using OpenCV (`cv2.VideoCapture`).
   - Exactly 16 frames are uniformly sampled across the video timeline.
   - If the video contains fewer than 16 frames, boundary padding is applied.
2. **Face ROI Localization**:
   - InsightFace SCRFD (`buffalo_l`) locates the primary facial region.
   - A 15% contextual margin is expanded around the bounding box to capture jawline and hairline blending boundaries.
   - Each frame is cropped, resized to $224 \times 224$, and normalized to ImageNet distribution: $\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$.
   - Resulting tensor shape: `(1, 16, 3, 224, 224)`.
3. **Audio Ingestion & Availability Detection**:
   - Audio is extracted via `imageio-ffmpeg` or `torchaudio` into a 16 kHz mono signal.
   - The signal is inspected by `AudioDetector`:
     - Checks for missing stream, 0-byte duration, unreadable codec, all-zero silence ($<10^{-7}$ amplitude), and NaN/Inf values.
   - If valid, the waveform is padded or trimmed to exactly 160,000 samples (10 seconds) with shape `(1, 160000)` and `audio_available = True`.
   - If invalid/silent/missing, `audio_available = False` is flagged and audio encoders are bypassed.

---

## 3. 6-Encoder Multimodal Architecture

### Visual Stream (3 Encoders):
- **EfficientNet-B0 (1280-D)**: Extracts spatial high-resolution texture representations, forensic noise patterns, and boundary blending artifacts.
- **Video Swin-Tiny (768-D)**: Shifted-window 3D video transformer capturing spatio-temporal dynamics and inter-frame temporal continuity.
- **Face/Mouth ResNet-18 (512-D)**: Localized perioral articulation encoder capturing mouth region landmark transitions.
- **Vision Feature Aggregator**: Linear projection layer mapping $(1280 + 768 + 512 = 2560\text{-D}) \to 1280\text{-D}$.
- **Vision Projection Head**: 2-layer MLP with LayerNorm mapping $1280\text{-D} \to 256\text{-D}$ ($L_2$-normalized).

### Acoustic Stream (3 Encoders):
- **Whisper Base (512-D)**: Pretrained speech encoder capturing semantic linguistic structures and phonetic phrasing.
- **Wav2Vec2 Base (768-D)**: Self-supervised acoustic speech model capturing vocal timbre, pitch consistency, and vocoder artifacts.
- **Spectrogram CNN (256-D)**: 4-layer 2D CNN operating on Mel-spectrograms capturing time-frequency harmonics and phase anomalies.
- **Audio Feature Aggregator**: Linear projection layer mapping $(512 + 768 + 256 = 1536\text{-D}) \to 512\text{-D}$.
- **Audio Projection Head**: 2-layer MLP with LayerNorm mapping $512\text{-D} \to 256\text{-D}$ ($L_2$-normalized).

---

## 4. Cross-Modal Consistency & Adaptive Fusion

### Cross-Modal Consistency ($S_{AV}$):
The system computes the cosine similarity between the $L_2$-normalized visual embedding $\mathbf{e}_v \in \mathbb{R}^{256}$ and acoustic embedding $\mathbf{e}_a \in \mathbb{R}^{256}$:
$$S_{AV} = \mathbf{e}_v \cdot \mathbf{e}_a = \sum_{i=1}^{256} e_{v,i} \cdot e_{a,i}$$
- **Synchronized Audio ($S_{AV} \ge 0.1177$)**: Visual facial motion aligns with acoustic speech representations.
- **Desynchronized / Dubbed Audio ($S_{AV} < 0.1177$)**: Indicates cross-modal mismatch (e.g., translated audio, voice replacement, out-of-sync soundtrack).

### Adaptive Modality Fusion:
The concatenated embeddings $[\mathbf{e}_v \,\|\, \mathbf{e}_a] \in \mathbb{R}^{512}$ are passed through an attention network to compute dynamic trust weights:
$$[\alpha_{vision}, \alpha_{audio}] = \text{Softmax}(\mathbf{W}_{att} [\mathbf{e}_v \,\|\, \mathbf{e}_a] + \mathbf{b}_{att})$$
$$\mathbf{e}_{fused} = [\alpha_{vision} \cdot \mathbf{e}_v \,\|\, \alpha_{audio} \cdot \mathbf{e}_a] \in \mathbb{R}^{512}$$

When `audio_available = False`, the fusion network dynamically overrides the weights:
$$\alpha_{vision} = 1.0, \quad \alpha_{audio} = 0.0$$
$$\mathbf{e}_{fused} = [\mathbf{e}_v \,\|\, \mathbf{0}] \in \mathbb{R}^{512}$$

---

## 5. Security & Lifecycle Management

- **Memory Optimization**: Model weights (`best_model.pth`) are loaded once into GPU/CPU memory during the FastAPI `lifespan` startup event. No encoders are re-initialized per request.
- **Inference Mode**: All forward passes execute strictly within `torch.inference_mode()` with `requires_grad=False`.
- **Upload Safety**: Unique UUID temporary directories are created per upload session and safely unlinked in a `finally` block upon completion.
