# System Architecture & Technical Specifications

**Project Title:** Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion  
**System Designation:** Hexa-Modal ($3\times3$) Forensic Deepfake Detection Framework  
**Document Scope:** Detailed mathematical specifications, tensor flows, architectural schematics, and operational pipelines.

---

## 1. High-Level System Architecture

```
+========================================================================================================+
|                                        INPUT MEDIA INGESTION                                           |
|                            Raw Digital Video (.mp4, .avi, .mov, .mkv, .webm)                           |
+========================================================================================================+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
       [VISUAL PREPROCESSING PIPELINE]                       [ACOUSTIC PREPROCESSING PIPELINE]
       - OpenCV Temporal Frame Sampling                       - FFmpeg Demuxing & 16 kHz Resampling
       - 16 Uniform Frames @ (3, 224, 224)                    - 1-Channel Mono Waveform (160,000 samples)
       - InsightFace SCRFD (buffalo_l)                        - AudioDetector (Missing/Silence/SNR check)
       - 15% Context Margin Expansion                         - Audio Flag: audio_available in {True, False}
       - Lower Facial Mouth ROI Dynamic Crop                                   |
                         |                                                     |
       +-----------------+-----------------+                 +-----------------+-----------------+
       |                 |                 |                 |                 |                 |
       v                 v                 v                 v                 v                 v
+--------------+  +--------------+  +--------------+  +--------------+  +--------------+  +--------------+
|   BRANCH 1   |  |   BRANCH 2   |  |   BRANCH 3   |  |   BRANCH 4   |  |   BRANCH 5   |  |   BRANCH 6   |
| EfficientNet |  |  Video Swin  |  |  Face/Mouth  |  | Whisper Base |  | Wav2Vec2 Base|  | Spectrogram  |
|      B0      |  |  Tiny (3D)   |  |  ResNet-18   |  |   Encoder    |  |   Encoder    |  |    2D CNN    |
| (1,280-D)    |  |   (768-D)    |  |   (512-D)    |  |   (512-D)    |  |   (768-D)    |  |   (256-D)    |
| Spatial Text.|  | Temporal Dyn.|  | Perioral Art.|  | Semantic Sp. |  | Acoustic Tok.|  | Harmonics    |
|   [FROZEN]   |  |   [FROZEN]   |  |   [FROZEN]   |  |   [FROZEN]   |  |   [FROZEN]   |  |   [FROZEN]   |
+--------------+  +--------------+  +--------------+  +--------------+  +--------------+  +--------------+
       |                 |                 |                 |                 |                 |
       +-----------------+-----------------+                 +-----------------+-----------------+
                         |                                                     |
                         v                                                     v
       [VISION FEATURE AGGREGATION]                          [AUDIO FEATURE AGGREGATION]
       Concat [1280 + 768 + 512] = 2560-D                    Concat [512 + 768 + 256] = 1536-D
       Linear(2560 -> 1280) + LayerNorm + GELU               Linear(1536 -> 512) + LayerNorm + GELU
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
            [VISION PROJECTION HEAD]                               [AUDIO PROJECTION HEAD]
            BatchNorm1d(1280)                                      BatchNorm1d(512)
            Linear(1280 -> 256)                                    Linear(512 -> 256)
            L2 Unit Normalization (||e_v||_2 = 1)                  L2 Unit Normalization (||e_a||_2 = 1)
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                                  [CROSS-MODAL CONTRASTIVE MODULE]
                                  Bidirectional InfoNCE Loss (tau = 0.07)
                                  Cosine Similarity Metric: S_AV = <e_v, e_a>
                                  Dubbing Diagnostic Margin: tau_sync = 0.1177
                                                    |
                         +--------------------------+
                         |
                         v
       [ADAPTIVE MULTIMODAL ATTENTION FUSION]
       Input: Fused Vision (1280-D) + Fused Audio (512-D)
       Dynamic Attention Gating Network -> Weights [w_v, w_a]
       Missing-Audio Bypass Override: If audio_available is False -> w_v = 1.0, w_a = 0.0
       Fused Multimodal Vector: h_fuse = [w_v * h_v; w_a * h_a] in R^1024
                         |
                         v
            [DEEPFAKE CLASSIFIER HEAD]
            Linear(1024 -> 256) -> LeakyReLU(0.1) -> Dropout(0.3)
            Linear(256 -> 128)  -> LeakyReLU(0.1) -> Dropout(0.3)
            Linear(128 -> 2)    -> Raw Classification Logits [z_0, z_1]
                         |
                         +--------------------------+
                         |                          |
                         v                          v
       [TEMPERATURE SCALING CALIBRATION]       [OOD RELIABILITY CENTROID DETECTOR]
       Calibrated Softmax: T = 0.7924          Validation Centroid Distance d_M
       P(Real), P(Fake) with Reduced ECE       Flag: HIGH / MEDIUM / LOW RELIABILITY
                         |                          |
                         +--------------------------+
                                     |
                                     v
                 [AGENTIC FORENSIC REASONING ENGINE]
                 Structured Evidence Data Contract (Schema v1.0)
                 Deterministic Rule-Based Evidence Grounding (Rules 1-7)
                 Decoupled Natural Language Synthesis (LLaMA-3.3-70B / Fallback)
                                     |
                                     v
                  [FASTAPI WEB APPLICATION & DASHBOARD]
                  REST Endpoints: /api/analyze, /api/demo, /api/health
                  Dynamic Dark-Mode UI with Modality Contribution Gauges
```

---

## 2. Modality Pipeline Breakdown

### 2.1 Visual Processing Pipeline
The visual stream processes 16 uniformly sampled frames ($224 \times 224$ pixels) through three specialized backbones targeting complementary manipulation artifacts:

1. **Branch 1: EfficientNet-B0 (Spatial Texture)**
   - **Architecture:** Pretrained CNN backbone (`torchvision.models.efficientnet_b0`).
   - **Input:** $B \times 16 \times 3 \times 224 \times 224 \to$ pooled over time to $B \times 1280$.
   - **Forensic Role:** Captures high-frequency spatial discrepancies, blending boundaries, resampling artifacts, and color-space aberrations around the face.
   - **Parameters:** 4,007,548 (Frozen during fusion training).

2. **Branch 2: Video Swin Transformer Tiny (Spatio-Temporal Dynamics)**
   - **Architecture:** 3D shifted-window video transformer (`torchvision.models.video_swin_t`).
   - **Input:** $B \times 3 \times 16 \times 224 \times 224 \to$ spatio-temporal attention pooling to $B \times 768$.
   - **Forensic Role:** Captures temporal flickering, inter-frame motion jitter, unnatural eye-blink dynamics, and expression discontinuities.
   - **Parameters:** 27,665,958 (Frozen during fusion training).

3. **Branch 3: Face/Mouth ResNet-18 (Localized Perioral Articulation)**
   - **Architecture:** Custom ResNet-18 with dynamic lower-face ROI cropping (`models/face_mouth_encoder.py`).
   - **Input:** Bilinear crop of lower facial region ($H/2 \dots H, W/4 \dots 3W/4$) resized to $112 \times 112 \to B \times 512$.
   - **Forensic Role:** Captures lip-sync anomalies, unnatural phoneme-viseme mouth shape deformation, and speech articulation inconsistencies.
   - **Parameters:** 11,176,512 (Frozen during fusion training).

### 2.2 Acoustic Processing Pipeline
The audio stream extracts 16 kHz mono waveforms (up to 10 seconds, 160,000 samples) through three complementary backbones:

4. **Branch 4: OpenAI Whisper Base (Speech Semantics)**
   - **Architecture:** Transformer audio encoder (`whisper.load_model('base.en')`).
   - **Input:** 80-channel log-Mel spectrogram $\to B \times 512$.
   - **Forensic Role:** Encodes phonetic cadence, lexical pronunciation patterns, and high-level linguistic structure.
   - **Parameters:** 87,888,896 (Frozen during fusion training).

5. **Branch 5: Torchaudio Wav2Vec2 Base (Acoustic Waveforms)**
   - **Architecture:** Self-supervised acoustic convolutional + transformer model (`torchaudio.pipelines.WAV2VEC2_BASE`).
   - **Input:** Raw waveform tensor $B \times 1 \times 160000 \to B \times 768$.
   - **Forensic Role:** Captures neural vocoder artifacts, phase irregularities, robotic timbre, and synthetic voice cloning markers.
   - **Parameters:** 94,371,520 (Frozen during fusion training).

6. **Branch 6: Spectrogram 2D CNN (Time-Frequency Spectral Power)**
   - **Architecture:** Custom 4-stage convolutional neural network (`models/spectrogram_cnn.py`).
   - **Input:** 128-channel Mel-spectrogram ($128 \times 313$) $\to B \times 256$.
   - **Forensic Role:** Captures spectral energy distribution, harmonic distortion, and frequency band cutoffs.
   - **Parameters:** 1,224,704 (Frozen during fusion training).

---

## 3. Feature Aggregation & Projection Heads

### 3.1 Vision Feature Aggregator
Concatenates all 3 visual representations and projects them into a unified visual space:
$$\mathbf{h}_v = \text{LayerNorm}\left(\text{Linear}_{2560 \to 1280}([\mathbf{x}_{\text{eff}}; \mathbf{x}_{\text{swin}}; \mathbf{x}_{\text{mouth}}])\right) \in \mathbb{R}^{1280}$$

### 3.2 Audio Feature Aggregator
Concatenates all 3 acoustic representations and projects them into a unified audio space:
$$\mathbf{h}_a = \text{LayerNorm}\left(\text{Linear}_{1536 \to 512}([\mathbf{x}_{\text{whisper}}; \mathbf{x}_{\text{w2v}}; \mathbf{x}_{\text{spec}}])\right) \in \mathbb{R}^{512}$$

### 3.3 Projection Heads
To enable contrastive cross-modal alignment without distorting classification representations, both aggregated vectors are mapped into a shared 256-D hyperspherical space:
$$\mathbf{e}_v = \frac{\text{Linear}_{1280 \to 256}(\text{BatchNorm}(\mathbf{h}_v))}{\|\text{Linear}_{1280 \to 256}(\text{BatchNorm}(\mathbf{h}_v))\|_2} \in \mathbb{S}^{255}$$
$$\mathbf{e}_a = \frac{\text{Linear}_{512 \to 256}(\text{BatchNorm}(\mathbf{h}_a))}{\|\text{Linear}_{512 \to 256}(\text{BatchNorm}(\mathbf{h}_a))\|_2} \in \mathbb{S}^{255}$$

---

## 4. Cross-Modal Consistency Learning

### 4.1 Contrastive Alignment Module
Cross-modal synchronization is evaluated using the cosine similarity between normalized embeddings:
$$S_{AV} = \mathbf{e}_v^T \mathbf{e}_a \in [-1, 1]$$

During training, cross-modal alignment is enforced using a bidirectional InfoNCE contrastive loss with temperature $\tau = 0.07$:
$$\mathcal{L}_{\text{contrast}} = -\frac{1}{2B} \sum_{i=1}^B \left[ \log \frac{\exp(\mathbf{e}_{v,i}^T \mathbf{e}_{a,i} / \tau)}{\sum_{j=1}^B \exp(\mathbf{e}_{v,i}^T \mathbf{e}_{a,j} / \tau)} + \log \frac{\exp(\mathbf{e}_{v,i}^T \mathbf{e}_{a,i} / \tau)}{\sum_{j=1}^B \exp(\mathbf{e}_{v,j}^T \mathbf{e}_{a,i} / \tau)} \right]$$

### 4.2 Dubbing & Desynchronization Diagnostic
Authentic speech demonstrates high cross-modal correlation ($\mu = +0.2618$), while desynchronized or dubbed speech drops sharply ($\mu = -0.0060$). A forensic threshold of $\tau_{\text{sync}} = 0.1177$ provides a $+0.2678$ separation margin to reliably detect audio-visual manipulation.

---

## 5. Adaptive Modality Attention Fusion & Missing-Audio Bypass

### 5.1 Gated Softmax Attention Network
The fusion layer computes dynamic scalar attention weights for vision and audio:
$$[\alpha_v, \alpha_a] = \text{Softmax}\left(\text{MLP}_{1792 \to 2}([\mathbf{h}_v; \mathbf{h}_a])\right)$$

### 5.2 Robust Missing-Audio Bypass Mechanism
In real-world scenarios where the audio track is missing or silent ($\text{audio\_available} = \text{False}$):
$$\alpha_v = 1.0, \quad \alpha_a = 0.0, \quad \mathbf{h}_a = \mathbf{0}$$
$$\mathbf{h}_{\text{fuse}} = [\alpha_v \mathbf{h}_v; \alpha_a \mathbf{h}_a] = [\mathbf{h}_v; \mathbf{0}] \in \mathbb{R}^{1792}$$

This mathematical override prevents NaN gradient propagation and ensures graceful **vision-only fallback** (achieving 97.68% accuracy on silent media).

---

## 6. Deepfake Classifier Head

The fused representation passes through a non-linear classifier:
$$\mathbf{z} = \text{Linear}_{128 \to 2}\left(\text{Drop}_{0.3}(\text{LeakyReLU}(\text{Linear}_{256 \to 128}(\dots)))\right) \in \mathbb{R}^2$$

- Logit $z_0$: Class 0 (REAL)
- Logit $z_1$: Class 1 (FAKE)

The model is trained using a composite objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}(\mathbf{z}, y) + \lambda \mathcal{L}_{\text{contrast}}(\mathbf{e}_v, \mathbf{e}_a), \quad \lambda = 0.1$$

---

## 7. Post-Hoc Reliability & Explainability Layers

### 7.1 Temperature Scaling Calibration
To mitigate neural network overconfidence on out-of-distribution inputs, probabilities are calibrated using validation-optimized temperature scaling ($T = 0.7924$):
$$\hat{p}_k = \frac{\exp(z_k / T)}{\sum_{j} \exp(z_j / T)}$$
This reduces Expected Calibration Error (ECE) from 5.25% to 3.21%.

### 7.2 Out-of-Distribution (OOD) Centroid Detector
A Mahalanobis distance metric computed against in-distribution feature centroids stored in `experiments/phase8/ood/reference_stats.pt` flags out-of-distribution inputs (e.g., lossy screen recordings, atypical aspect ratios) as `LOW RELIABILITY`, issuing cautionary warnings without altering classifier decision boundaries.

### 7.3 Decoupled Forensic Agent Layer
The deterministic reasoning engine (`agent/reasoning.py`) analyzes detector telemetry according to 7 strict rules:
1. Ground all findings strictly in quantitative metrics.
2. Disentangle spatial vs temporal visual artifacts.
3. Quantify audio-visual synchronization using $S_{AV}$.
4. Explain modality fusion weighting ($\alpha_v, \alpha_a$).
5. Audit input video quality (resolution, face ROI size).
6. Issue explicit OOD warnings on degraded captures.
7. Strictly preserve classifier verdict authority (zero hallucination).

Reports can be translated into natural language narratives via plug-and-play LLM providers (Groq LLaMA-3.3-70B, OpenAI GPT-4o-mini, Google Gemini, or local fallback).
