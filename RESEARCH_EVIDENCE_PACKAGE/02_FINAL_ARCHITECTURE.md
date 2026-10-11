# Final Proposed Model Architecture: Full 3V3 Multimodal System

## 1. Architectural Blueprint
The final proposed deepfake detection system implements a **Full 3-Vision + 3-Audio (6-Encoder)** architecture. It fuses spatial texture analysis, temporal 3D motion consistency, perioral articulation dynamics, semantic speech phonetics, acoustic self-supervised representations, and time-frequency log-Mel spectrograms.

```
========================================================================================================================
BRANCH              FOUNDATION MODEL         PRETRAINING           OUTPUT DIM    STATUS       PARAMETERS
========================================================================================================================
Vision Branch 1     EfficientNet-B0          ImageNet-1K           1,280-D       Frozen        4,007,548
Vision Branch 2     Video Swin-Tiny (3D)     Kinetics-400            768-D       Frozen       27,850,470
Vision Branch 3     Face/Mouth ResNet-18     ImageNet-1K             512-D       Frozen       11,176,512
Vision Aggregation  Linear Layer             Newly Initialized     1,280-D       Trainable     5,900,800
Vision Projection   MLP + L2-Norm            Newly Initialized       256-D       Trainable       788,224
------------------------------------------------------------------------------------------------------------------------
Audio Branch 4      Whisper Base.en          OpenAI Multilingual     512-D       Frozen       19,822,592
Audio Branch 5      Wav2Vec2 Base            LibriSpeech 960h        768-D       Frozen       94,370,944
Audio Branch 6      Spectrogram 2D CNN       Newly Initialized       256-D       Trainable       454,592
Audio Aggregation   Linear Layer             Newly Initialized       512-D       Trainable     1,442,816
Audio Projection    MLP + L2-Norm            Newly Initialized       256-D       Trainable       395,008
------------------------------------------------------------------------------------------------------------------------
Multimodal Fusion   Adaptive Attention       Newly Initialized       512-D       Trainable        65,922
Deepfake Classifier 2-Layer MLP + Dropout    Newly Initialized         2-D       Trainable       165,250
Contrastive Module  InfoNCE Metric (tau=0.07) Non-Parametric       Scalar Loss   Non-Parametric        0
========================================================================================================================
TOTAL PARAMETER COUNT:                                                           166,440,678
  - Total Frozen Pretrained Parameters:                                          157,228,066 (94.46%)
  - Total Trainable Fusion & Aggregation Parameters:                               9,212,612 (5.54%)
========================================================================================================================
```

---

## 2. Vision Encoders (3 Branches)

### A. Vision Branch 1: EfficientNet-B0 (Spatial Texture & Blending Artifacts)
- **Source File**: `models/efficientnet.py` (`EfficientNetEncoder`)
- **Backbone**: `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`
- **Input Shape**: $(B, T, C, H, W) = (B, 16, 3, 224, 224)$
- **Output Shape**: $(B, 1280)$
- **Role**: Extracts high-frequency spatial texture anomalies, boundary blending seams, color inconsistencies, and warping artifacts across individual video frames.
- **Operations**:
  1. Reshapes input tensor to $(B 	imes T, 3, 224, 224)$.
  2. Extracts feature maps via `backbone.features`.
  3. Applies 2D Adaptive Average Pooling `backbone.avgpool` $	o (B 	imes T, 1280, 1, 1)$.
  4. Flattens and reshapes back to $(B, T, 1280)$.
  5. Computes temporal mean pooling across the $T=16$ dimension $	o (B, 1280)$.

### B. Vision Branch 2: Video Swin Transformer (Spatio-Temporal Coherence)
- **Source File**: `models/video_swin.py` (`VideoSwinEncoder`)
- **Backbone**: `torchvision.models.video.swin3d_t(weights=Swin3D_T_Weights.DEFAULT)`
- **Input Shape**: $(B, T, C, H, W) = (B, 16, 3, 224, 224)$
- **Output Shape**: $(B, 768)$
- **Role**: Captures 3D spatio-temporal self-attention across shifted 3D video windows, detecting inter-frame flickering, motion jitter, temporal discontinuity, and unnatural facial expressions.
- **Operations**:
  1. Permutes input tensor from $(B, T, C, H, W)$ to $(B, C, T, H, W) = (B, 3, 16, 224, 224)$.
  2. Replaces classification head with `nn.Identity()`.
  3. Propagates 3D windowed attention through Swin stages $	o (B, 768)$.

### C. Vision Branch 3: Face/Mouth ResNet-18 (Perioral Articulation & Lip ROI)
- **Source File**: `models/face_mouth_encoder.py` (`FaceMouthEncoder`)
- **Backbone**: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`
- **Input Shape**: $(B, T, C, H, W) = (B, 16, 3, 224, 224)$
- **Output Shape**: $(B, 512)$
- **Role**: Dedicated visual branch focused exclusively on the lower facial / mouth ROI (lips, jawline, teeth, mouth contours) to detect speech-driven lip-sync synthesis artifacts.
- **Operations**:
  1. Dynamically crops the lower facial ROI: rows $H/2 \dots H$ ($112 \dots 224$), cols $W/4 \dots 3W/4$ ($56 \dots 168$) $	o (B, T, 3, 112, 112)$.
  2. Resizes cropped ROI bilaterally to $(224, 224)$ to match the receptive field of ResNet-18.
  3. Passes $(B 	imes T, 3, 224, 224)$ through ResNet-18 backbone with `resnet.fc = nn.Identity()`.
  4. Applies temporal mean pooling across $T=16 	o (B, 512)$.

---

## 3. Vision Feature Aggregation & Projection Head

### A. Vision Feature Aggregator
- **Source File**: `models/vision_feature_aggregation.py` (`VisionFeatureAggregation`)
- **Concatenation**: $\mathbf{f}_{v,	ext{concat}} = [\mathbf{f}_{	ext{eff}} \,\|\, \mathbf{f}_{	ext{swin}} \,\|\, \mathbf{f}_{	ext{mouth}}] \in \mathbb{R}^{1280 + 768 + 512} = \mathbb{R}^{2560}$
- **Projection**: $\mathbf{f}_v = \mathbf{W}_{	ext{agg},v} \mathbf{f}_{v,	ext{concat}} + \mathbf{b}_{	ext{agg},v} \in \mathbb{R}^{1280}$
- **Parameters**: `nn.Linear(2560, 1280)` ($2560 	imes 1280 + 1280 = 3,278,080$ weights + biases; includes backward compatibility layer for 2V mode).

### B. Vision Projection Head
- **Source File**: `models/projection_heads.py` (`VisionProjectionHead`)
- **Architecture**:
  $$\mathbf{h}_v = 	ext{ReLU}(	ext{BatchNorm1d}(\mathbf{W}_{v1} \mathbf{f}_v + \mathbf{b}_{v1})) \in \mathbb{R}^{512}$$
  $$	ilde{\mathbf{e}}_v = \mathbf{W}_{v2} 	ext{Dropout}_{0.2}(\mathbf{h}_v) + \mathbf{b}_{v2} \in \mathbb{R}^{256}$$
  $$\mathbf{e}_v = rac{	ilde{\mathbf{e}}_v}{\|	ilde{\mathbf{e}}_v\|_2} \in \mathbb{R}^{256} \quad (	ext{Unit } L_2 	ext{ Norm})$$

---

## 4. Audio Encoders (3 Branches)

### A. Audio Branch 4: Whisper Base.en (Semantic Speech & Phonetics)
- **Source File**: `models/whisper_encoder.py` (`WhisperEncoder`)
- **Backbone**: `whisper.load_model("base", device="cpu").encoder`
- **Input**: Raw waveform $(B, 160000)$ @ 16,000 Hz.
- **Output**: $(B, 512)$
- **Role**: Extracts high-level semantic speech patterns, phonetic transitions, and language cadence markers.
- **Operations**:
  1. Computes 80-channel log-Mel spectrogram with 30-second padding/trimming via `whisper.log_mel_spectrogram` $	o (B, 80, 3000)$.
  2. Passes through 2 Conv1D layers (stride 2) + 6 Transformer encoder blocks.
  3. Performs Global Average Pooling across time dimension $	o (B, 512)$.

### B. Audio Branch 5: Wav2Vec2 Base (Self-Supervised Acoustic Representations)
- **Source File**: `models/wav2vec2_encoder.py` (`Wav2Vec2Encoder`)
- **Backbone**: `torchaudio.pipelines.WAV2VEC2_BASE.get_model()`
- **Input**: Raw waveform $(B, 160000)$ @ 16,000 Hz.
- **Output**: $(B, 768)$
- **Role**: Captures fine-grained acoustic speech characteristics, vocoder synthesis artifacts, and unnatural pitch/formant trajectories.
- **Operations**:
  1. Temporal Convolutional Feature Extractor processes raw waveform into latent feature representations.
  2. 12 Transformer encoder blocks extract contextual representations.
  3. Extracts last transformer layer output $(B, T, 768)$ and performs Global Average Pooling across $T 	o (B, 768)$.

### C. Audio Branch 6: Spectrogram 2D CNN (Time-Frequency Spectral Harmonics)
- **Source File**: `models/spectrogram_cnn.py` (`SpectrogramCNNEncoder`)
- **Architecture**: Lightweight 4-stage 2D CNN operating on 80-bin log-Mel spectrograms.
- **Parameters**: 1024 FFT, 512 Hop Length, 80 Mel Bins.
- **Layers**:
  - Block 1: `Conv2d(1, 32, k=3, p=1)` $	o$ `BatchNorm2d(32)` $	o$ `ReLU` $	o$ `MaxPool2d(2, 2)`
  - Block 2: `Conv2d(32, 64, k=3, p=1)` $	o$ `BatchNorm2d(64)` $	o$ `ReLU` $	o$ `MaxPool2d(2, 2)`
  - Block 3: `Conv2d(64, 128, k=3, p=1)` $	o$ `BatchNorm2d(128)` $	o$ `ReLU` $	o$ `MaxPool2d(2, 2)`
  - Block 4: `Conv2d(128, 256, k=3, p=1)` $	o$ `BatchNorm2d(256)` $	o$ `ReLU` $	o$ `AdaptiveAvgPool2d((1, 1))`
  - Linear FC: `nn.Linear(256, 256)`
- **Output**: $(B, 256)$
- **Role**: Detects high-frequency spectral phase discontinuities, harmonic inconsistencies, and vocoder cut-off frequencies.

---

## 5. Audio Feature Aggregation & Projection Head

### A. Audio Feature Aggregator
- **Source File**: `models/audio_feature_aggregation.py` (`AudioFeatureAggregation`)
- **Concatenation**: $\mathbf{f}_{a,	ext{concat}} = [\mathbf{f}_{	ext{whisper}} \,\|\, \mathbf{f}_{	ext{wav2vec2}} \,\|\, \mathbf{f}_{	ext{spec}}] \in \mathbb{R}^{512 + 768 + 256} = \mathbb{R}^{1536}$
- **Projection**: $\mathbf{f}_a = \mathbf{W}_{	ext{agg},a} \mathbf{f}_{a,	ext{concat}} + \mathbf{b}_{	ext{agg},a} \in \mathbb{R}^{512}$
- **Parameters**: `nn.Linear(1536, 512)` ($1536 	imes 512 + 512 = 786,944$ parameters).

### B. Audio Projection Head
- **Source File**: `models/projection_heads.py` (`AudioProjectionHead`)
- **Architecture**:
  $$\mathbf{h}_a = 	ext{ReLU}(	ext{BatchNorm1d}(\mathbf{W}_{a1} \mathbf{f}_a + \mathbf{b}_{a1})) \in \mathbb{R}^{512}$$
  $$	ilde{\mathbf{e}}_a = \mathbf{W}_{a2} 	ext{Dropout}_{0.2}(\mathbf{h}_a) + \mathbf{b}_{a2} \in \mathbb{R}^{256}$$
  $$\mathbf{e}_a = rac{	ilde{\mathbf{e}}_a}{\|	ilde{\mathbf{e}}_a\|_2} \in \mathbb{R}^{256} \quad (	ext{Unit } L_2 	ext{ Norm})$$

---

## 6. Contrastive Learning & Cross-Modal Consistency

- **Source File**: `models/contrastive_module.py` (`ContrastiveModule`)
- **InfoNCE Loss Formulation**:
  $$\mathcal{L}_{	ext{contrastive}} = -rac{1}{2B} \sum_{i=1}^B \left[ \log rac{\exp(\mathbf{e}_{v,i}^T \mathbf{e}_{a,i} / 	au)}{\sum_{j=1}^B \exp(\mathbf{e}_{v,i}^T \mathbf{e}_{a,j} / 	au)} + \log rac{\exp(\mathbf{e}_{a,i}^T \mathbf{e}_{v,i} / 	au)}{\sum_{j=1}^B \exp(\mathbf{e}_{a,i}^T \mathbf{e}_{v,j} / 	au)} ight]$$
  where temperature $	au = 0.07$.
- **Audio-Visual Consistency Metric**:
  $$S_{AV} = \mathbf{e}_v^T \mathbf{e}_a = \cos(\mathbf{e}_v, \mathbf{e}_a) \in [-1.0, 1.0]$$

---

## 7. Adaptive Multimodal Attention Fusion & Deepfake Classifier

### A. Adaptive Fusion Module
- **Source File**: `models/adaptive_fusion.py` (`AdaptiveFusion`)
- **Mechanism**:
  $$\mathbf{u} = [\mathbf{e}_v \,\|\, \mathbf{e}_a] \in \mathbb{R}^{512}$$
  $$\mathbf{z} = \mathbf{W}_2 	ext{ReLU}(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^2$$
  $$[lpha_v, lpha_a] = 	ext{Softmax}(\mathbf{z}) \quad 	ext{such that } lpha_v + lpha_a = 1.0$$
  $$\mathbf{e}_{	ext{fused}} = [lpha_v \mathbf{e}_v \,\|\, lpha_a \mathbf{e}_a] \in \mathbb{R}^{512}$$
- **Fallback Protocol (Missing Audio)**:
  If `audio_available == False` or `audio_embedding is None`:
  $$lpha_v = 1.000, \quad lpha_a = 0.000 \implies \mathbf{e}_{	ext{fused}} = [\mathbf{e}_v \,\|\, \mathbf{0}] \in \mathbb{R}^{512}$$

### B. Deepfake Classifier Head
- **Source File**: `models/classifier.py` (`DeepfakeClassifier`)
- **Structure**:
  $$\mathbf{h}_1 = 	ext{Dropout}_{0.3}(	ext{ReLU}(	ext{BatchNorm1d}(\mathbf{W}_{c1} \mathbf{e}_{	ext{fused}} + \mathbf{b}_{c1}))) \in \mathbb{R}^{256}$$
  $$\mathbf{h}_2 = 	ext{Dropout}_{0.3}(	ext{ReLU}(	ext{BatchNorm1d}(\mathbf{W}_{c2} \mathbf{h}_1 + \mathbf{b}_{c2}))) \in \mathbb{R}^{128}$$
  $$\mathbf{z}_{	ext{out}} = \mathbf{W}_{c3} \mathbf{h}_2 + \mathbf{b}_{c3} \in \mathbb{R}^2 \quad (	ext{Logits: Class 0 = Real, Class 1 = Fake})$$
