# System Architecture, Workflow, and Technical Diagrams

**Project Title:** Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion  
**Scope:** Architectural blueprint, end-to-end operational workflow, pseudocode, module-by-module breakdown, and figure descriptions for academic report preparation.

---

## 1. High-Level Textual Architecture Block Diagram

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
       - 15% Context Margin Expansion                         - Audio Flag: audio_available ∈ {True, False}
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
       +------------------------------------+                +------------------------------------+
       |     VISION FEATURE AGGREGATOR      |                |      AUDIO FEATURE AGGREGATOR      |
       |  f_v,concat ∈ R^2560 -> Linear     |                |  f_a,concat ∈ R^1536 -> Linear     |
       |      Unified f_v ∈ R^1280          |                |      Unified f_a ∈ R^512           |
       |          [TRAINABLE]               |                |          [TRAINABLE]               |
       +------------------------------------+                +------------------------------------+
                         |                                                     |
                         v                                                     v
       +------------------------------------+                +------------------------------------+
       |       VISION PROJECTION HEAD       |                |       AUDIO PROJECTION HEAD        |
       | Linear -> BN -> ReLU -> Linear     |                | Linear -> BN -> ReLU -> Linear     |
       |       Normalized e_v ∈ R^256       |                |       Normalized e_a ∈ R^256       |
       |       (||e_v||_2 = 1.0)            |                |       (||e_a||_2 = 1.0)            |
       +------------------------------------+                +------------------------------------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                               +-----------------------------------------+
                               |     CROSS-MODAL CONSISTENCY MODULE      |
                               |  - InfoNCE Contrastive Loss (tau = 0.07)|
                               |  - Cosine Alignment: S_AV = e_v^T · e_a  |
                               |  - Diagnostic Threshold: S_AV < 0.1177   |
                               +-----------------------------------------+
                                                    |
                                                    v
                               +-----------------------------------------+
                               |    ADAPTIVE MULTIMODAL ATTENTION FUSION |
                               |  - Concatenation: u = [e_v ; e_a] ∈ R^512|
                               |  - Dynamic Softmax: alpha_v, alpha_a    |
                               |  - Fallback (Missing Audio):            |
                               |      alpha_v = 1.000, alpha_a = 0.000   |
                               |  - Fused: e_fused = [a_v·e_v ; a_a·e_a]  |
                               +-----------------------------------------+
                                                    |
                                                    v
                               +-----------------------------------------+
                               |        DEEPFAKE CLASSIFIER HEAD         |
                               |  - Linear(512, 256) -> BN -> ReLU -> DO |
                               |  - Linear(256, 128) -> BN -> ReLU -> DO |
                               |  - Linear(128, 2) -> Logits [z_0, z_1]  |
                               |  - Raw Softmax: P(Real), P(Fake)        |
                               +-----------------------------------------+
                                                    |
                                                    v
                               +-----------------------------------------+
                               |    POST-HOC TEMPERATURE CALIBRATION     |
                               |  - Optimal Temperature: T = 0.7924      |
                               |  - Calibrated: q_k = Softmax(z_k / T)   |
                               |  - Lowers ECE from 5.25% to 3.21%       |
                               +-----------------------------------------+
                                                    |
                                                    v
                               +-----------------------------------------+
                               |    OOD & FORENSIC RELIABILITY DETECTOR  |
                               |  - Mahalanobis Representation Distance  |
                               |  - Threshold: 27.93 (95th pct on Val)   |
                               |  - Reliability Tier: HIGH / MED / LOW   |
                               +-----------------------------------------+
                                                    |
                                                    v
                               +-----------------------------------------+
                               |     STRUCTURED EVIDENCE CONTRACT        |
                               |  - Tamper-Proof Immutable JSON Payload  |
                               |  - Grounded Visual, Audio & AV Telemetry|
                               +-----------------------------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
       +------------------------------------+                +------------------------------------+
       |      LLM FORENSIC REASONER         |                |     FASTAPI SINGLETON SERVICE      |
       |  - Provider-Agnostic Engine        |                |  - REST Endpoints (/analyze)       |
       |    (Groq / OpenAI / Gemini)        |                |  - Single Startup Model Load       |
       |  - Strict Anti-Hallucination Prompt|                |  - Real-Time Async Execution       |
       |  - Rule 7: Cannot alter verdict    |                |  - Interactive Web Dashboard       |
       |  - Deterministic Fallback Engine   |                |  - Professional Forensic UI        |
       +------------------------------------+                +------------------------------------+
```

---

## 2. Step-by-Step End-to-End Operational Pipeline

### Step 1: Input Media Ingestion & Validation
- **Module**: `app/video_processor.py` (`VideoProcessor.validate_file`)
- **Action**: Verifies file extension (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) and file size ($\le 100\text{ MB}$).
- **Failure Mode**: Rejects unsupported formats or oversized payloads immediately before decoding.

### Step 2: Temporal Video Frame Sampling
- **Module**: `app/video_processor.py` (`VideoProcessor.extract_frames`)
- **Action**: Uses OpenCV (`cv2.VideoCapture`) to calculate total frame count $N_{\text{frames}}$. Samples exactly $T = 16$ frames uniformly across the video timeline:
  $$t_i = \left\lfloor i \cdot \frac{N_{\text{frames}} - 1}{T - 1} \right\rfloor, \quad i \in \{0, 1, \dots, 15\}$$
- **Boundary Handling**: If video has $<16$ frames, the final frame is duplicated to preserve temporal length.

### Step 3: Facial Landmark Localization & Cropping
- **Module**: `app/video_processor.py` (`VideoProcessor.process_video_frames`)
- **Action**: InsightFace SCRFD (`buffalo_l` detector) localizes primary face bounding box $(x_1, y_1, x_2, y_2)$.
- **Context Margin**: Expands bounding box by $15\%$ context margin:
  $$\Delta x = 0.15 \cdot (x_2 - x_1), \quad \Delta y = 0.15 \cdot (y_2 - y_1)$$
  $$x_1' = \max(0, x_1 - \Delta x), \quad y_1' = \max(0, y_1 - \Delta y), \quad x_2' = \min(W, x_2 + \Delta x), \quad y_2' = \min(H, y_2 + \Delta y)$$
- **Normalization**: Crops are resized to $224 \times 224$ and normalized via ImageNet distribution ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$).
- **Fallback**: If no face is detected, a centered portrait square crop is extracted.
- **Output Tensor**: Shape $(1, 16, 3, 224, 224)$.

### Step 4: Perioral Mouth ROI Extraction
- **Module**: `models/face_mouth_encoder.py` (`FaceMouthEncoder.forward`)
- **Action**: Dynamically slices the lower facial anatomy across all 16 frames:
  $$\text{ROI} = \text{Frame}\left[\frac{H}{2} : H, \; \frac{W}{4} : \frac{3W}{4}\right] \implies [112 : 224, \; 56 : 168]$$
- **Bilinear Resizing**: Bilinearly interpolated back to $(224, 224)$ to match the receptive field of ResNet-18.

### Step 5: Audio Ingestion & Energy Validation
- **Module**: `app/video_processor.py`, `preprocessing/audio_detector.py`
- **Action**: FFmpeg/imageio demuxes audio stream, converting to 16,000 Hz monophonic single-channel PCM.
- **Verification**: `AudioDetector.check_waveform` checks:
  1. Stream existence and duration ($\ge 0.5\text{ s}$).
  2. Non-zero RMS energy ($\max(|x|) \ge 10^{-7}$).
  3. Finite numeric values ($\text{NaN}/\text{Inf}$ rejection).
- **Formatting**: Padded or trimmed to exactly 160,000 samples (10 seconds) with shape $(1, 160000)$.
- **Status Assignment**: If invalid or missing, flags `audio_available = False`.

### Step 6: 6-Branch Deep Feature Extraction
- **Module**: `models/multimodal_model.py` (`MultimodalDeepfakeDetector.forward`)
- **Visual Stream (Parallel Execution)**:
  - `EfficientNet-B0`: Processes $(16, 3, 224, 224) \to$ temporal mean pooling $\to \mathbf{f}_{\text{eff}} \in \mathbb{R}^{1280}$.
  - `Video Swin-Tiny`: Permutes to $(1, 3, 16, 224, 224) \to$ 3D shifted window attention $\to \mathbf{f}_{\text{swin}} \in \mathbb{R}^{768}$.
  - `Face/Mouth ResNet-18`: ResNet-18 on mouth ROIs $\to$ temporal mean pooling $\to \mathbf{f}_{\text{mouth}} \in \mathbb{R}^{512}$.
- **Acoustic Stream (Parallel Execution / Bypassed if `audio_available == False`)**:
  - `Whisper Base.en`: Computes 80-channel log-Mel $\to$ 6 Transformer blocks $\to$ pooling $\to \mathbf{f}_{\text{whisper}} \in \mathbb{R}^{512}$.
  - `Wav2Vec2 Base`: Temporal convolution $\to$ 12 Transformer blocks $\to$ pooling $\to \mathbf{f}_{\text{wav2vec2}} \in \mathbb{R}^{768}$.
  - `Spectrogram CNN`: 80-bin STFT log-Mel $\to$ 4-stage 2D CNN $\to$ pooling $\to \mathbf{f}_{\text{spec}} \in \mathbb{R}^{256}$.

### Step 7: Feature Aggregation & Metric Projection
- **Module**: `models/vision_feature_aggregation.py`, `models/audio_feature_aggregation.py`, `models/projection_heads.py`
- **Vision Stream**:
  $$\mathbf{f}_{v,\text{concat}} = [\mathbf{f}_{\text{eff}} \,\|\, \mathbf{f}_{\text{swin}} \,\|\, \mathbf{f}_{\text{mouth}}] \in \mathbb{R}^{2560} \xrightarrow{\mathbf{W}_{\text{agg},v}} \mathbf{f}_v \in \mathbb{R}^{1280} \xrightarrow{\text{Projector}_v} \mathbf{e}_v \in \mathbb{R}^{256} \quad (\|\mathbf{e}_v\|_2 = 1.0)$$
- **Acoustic Stream**:
  $$\mathbf{f}_{a,\text{concat}} = [\mathbf{f}_{\text{whisper}} \,\|\, \mathbf{f}_{\text{wav2vec2}} \,\|\, \mathbf{f}_{\text{spec}}] \in \mathbb{R}^{1536} \xrightarrow{\mathbf{W}_{\text{agg},a}} \mathbf{f}_a \in \mathbb{R}^{512} \xrightarrow{\text{Projector}_a} \mathbf{e}_a \in \mathbb{R}^{256} \quad (\|\mathbf{e}_a\|_2 = 1.0)$$

### Step 8: Cross-Modal Consistency Evaluation
- **Module**: `models/contrastive_module.py`, `agent/reasoning.py`
- **Computation**: Inner product of $L_2$-normalized unit embeddings:
  $$S_{AV} = \mathbf{e}_v^T \mathbf{e}_a = \cos(\mathbf{e}_v, \mathbf{e}_a) \in [-1.0, +1.0]$$
- **Forensic Interpretation**:
  - $S_{AV} \ge 0.1177$: Normal audio-visual alignment (lip motion matches phonetics).
  - $S_{AV} < 0.1177$: Audio-visual desynchronization, dubbing, or audio track replacement.

### Step 9: Adaptive Multimodal Attention Fusion
- **Module**: `models/adaptive_fusion.py` (`AdaptiveFusion.forward`)
- **Normal Execution (`audio_available == True`)**:
  $$\mathbf{u} = [\mathbf{e}_v \,\|\, \mathbf{e}_a] \in \mathbb{R}^{512} \xrightarrow{\text{MLP}} \mathbf{z} \in \mathbb{R}^2 \xrightarrow{\text{Softmax}} [\alpha_v, \alpha_a]$$
  $$\mathbf{e}_{\text{fused}} = [\alpha_v \mathbf{e}_v \,\|\, \alpha_a \mathbf{e}_a] \in \mathbb{R}^{512}$$
- **Missing Audio Fallback (`audio_available == False`)**:
  $$\alpha_v = 1.000, \quad \alpha_a = 0.000 \implies \mathbf{e}_{\text{fused}} = [\mathbf{e}_v \,\|\, \mathbf{0}_{256}] \in \mathbb{R}^{512}$$

### Step 10: Deepfake Classification Logits & Probabilities
- **Module**: `models/classifier.py` (`DeepfakeClassifier.forward`)
- **Classifier**: 2-layer MLP with BatchNorm and Dropout(0.3) maps $\mathbf{e}_{\text{fused}} \in \mathbb{R}^{512} \to \mathbf{z}_{\text{out}} \in \mathbb{R}^2$.
- **Logit Interpretation**: $z_0 = \text{Real Logit}, z_1 = \text{Fake Logit}$.
- **Softmax Probabilities**:
  $$P(\text{Real}) = \frac{e^{z_0}}{e^{z_0} + e^{z_1}}, \quad P(\text{Fake}) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$
- **Verdict**: If $P(\text{Fake}) \ge 0.50 \implies \text{FAKE}$; else $\text{REAL}$.
- **Raw Confidence**: $C_{\text{raw}} = \max(P(\text{Real}), P(\text{Fake}))$.

### Step 11: Post-Hoc Temperature Scaling Calibration
- **Module**: `app/inference_service.py`, `experiments/phase8/calibration/temperature_scaling.py`
- **Action**: Applies learned temperature $T = 0.7924$:
  $$q_k = \frac{\exp(z_k / T)}{\exp(z_0 / T) + \exp(z_1 / T)}$$
- **Calibrated Confidence**: $C_{\text{cal}} = \max(q_0, q_1)$. Preserves 100% decision boundary invariance.

### Step 12: Input Quality Analysis & OOD Reliability Detection
- **Module**: `app/input_quality.py`, `experiments/phase8/ood/reliability_detector.py`
- **Input Quality Metrics**:
  - Laplacian frame variance (sharpness).
  - Face bounding box area ratio ($A_{\text{face}} / A_{\text{frame}}$).
  - Audio SNR and zero-crossing rate.
- **OOD Representation Distance**: Computes standardized distance from validation centroids $\boldsymbol{\mu}_{\text{Real}}, \boldsymbol{\mu}_{\text{Fake}}$:
  $$d_{\min} = \min_{c \in \{0, 1\}} \sqrt{ \sum_{k=1}^{512} \frac{(e_{\text{fused},k} - \mu_{c,k})^2}{\sigma_{c,k}^2 + 10^{-6}} }$$
- **Reliability Tier**:
  - **HIGH**: $d_{\min} \le 27.93$, face area $>15\%$, high sharpness, matched audio.
  - **MEDIUM**: $d_{\min} \le 27.93$ with mild degradation, missing audio, or dubbing.
  - **LOW / POSSIBLY OOD**: $d_{\min} > 27.93$, face area $<5\%$, heavy screen-capture compression.

### Step 13: Grounded Structured Evidence Packaging
- **Module**: `agent/structured_evidence.py` (`EvidenceSynthesizer.build_evidence_object`)
- **Action**: Packages detector verdict, confidence, calibrated score, modality weights, consistency score, input quality, and reliability status into an immutable `StructuredForensicEvidence` dataclass.

### Step 14: LLM Forensic Explanation & Narrative Generation
- **Module**: `agent/llm_forensic_agent.py`, `agent/reasoning.py`
- **Action**: Formats structured evidence into strict system prompt sent to LLM (Groq LLaMA-3.3-70B / OpenAI GPT-4o-mini).
- **Rule 7 Invariance**: LLM prompt explicitly forbids altering the classifier verdict or hallucinating non-existent artifacts.
- **Offline Fallback**: If LLM API key is unconfigured or network times out, `ForensicReasoningEngine` generates deterministic grounded explanations instantly.

### Step 15: Web Dashboard UI Rendering
- **Module**: `app/static/index.html`, `app/static/js/app.js`
- **Output**: Presents verdict badge, confidence bar, calibrated score, reliability badge, key forensic observations, and collapsible technical telemetry panels.

---

## 3. Module-by-Module Technical Responsibilities

| Subsystem Module | File Location | Primary Function & Responsibilities | Core Dependencies |
| :--- | :--- | :--- | :--- |
| **API Entry Point** | `app/main.py` | FastAPI application, CORS, static routes, `/analyze`, `/analyze-demo`, lifespan singleton loading. | `fastapi`, `uvicorn` |
| **Inference Orchestrator**| `app/inference_service.py` | Singleton pattern loading 3V3 detector once into VRAM, executing forward pass, applying temperature scaling. | `torch`, `models/multimodal_model.py` |
| **Video Preprocessor** | `app/video_processor.py` | 16-frame uniform extraction, InsightFace SCRFD face cropping, 15% margin expansion, FFmpeg 16kHz audio. | `cv2`, `insightface`, `torchaudio` |
| **Quality Analyzer** | `app/input_quality.py` | Real-time computation of Laplacian sharpness, face area ratio, aspect ratio, audio SNR, and silence detection. | `cv2`, `numpy`, `scipy` |
| **OOD Reliability** | `experiments/phase8/ood/` | Centroid distance computation from 2,414 validation representations, assigning HIGH/MED/LOW reliability. | `torch`, `reference_stats.pt` |
| **Vision Backbone 1** | `models/efficientnet.py` | Frozen EfficientNet-B0 extracting 1280-D spatial texture representations. | `torchvision.models` |
| **Vision Backbone 2** | `models/video_swin.py` | Frozen Video Swin-Tiny extracting 768-D 3D spatio-temporal self-attention features across 16 frames. | `torchvision.models.video` |
| **Vision Backbone 3** | `models/face_mouth_encoder.py`| Frozen ResNet-18 extracting 512-D perioral lip articulation features from dynamically cropped mouth ROIs. | `torchvision.models` |
| **Audio Backbone 4** | `models/whisper_encoder.py` | Frozen Whisper Base.en extracting 512-D semantic speech tokens from 80-channel log-Mel spectrograms. | `openai-whisper` |
| **Audio Backbone 5** | `models/wav2vec2_encoder.py` | Frozen Wav2Vec2 Base extracting 768-D acoustic speech waveform tokens and vocoder synthesis artifacts. | `torchaudio.pipelines` |
| **Audio Backbone 6** | `models/spectrogram_cnn.py` | 4-stage 2D CNN extracting 256-D time-frequency harmonic features from log-Mel spectrograms. | `torchaudio.transforms` |
| **Vision Aggregator** | `models/vision_feature_aggregation.py` | Trainable linear projection layer ($2560 \to 1280$) unifying the three visual streams. | `torch.nn.Linear` |
| **Audio Aggregator** | `models/audio_feature_aggregation.py` | Trainable linear projection layer ($1536 \to 512$) unifying the three acoustic streams. | `torch.nn.Linear` |
| **Projection Heads** | `models/projection_heads.py` | Linear + BatchNorm + ReLU + Linear + $L_2$ normalization mapping features to shared 256-D metric space. | `torch.nn` |
| **Contrastive Module** | `models/contrastive_module.py` | InfoNCE symmetric contrastive loss ($\tau = 0.07$) and cosine alignment metric $S_{AV} = \mathbf{e}_v^T \mathbf{e}_a$. | `torch.nn.functional` |
| **Adaptive Fusion** | `models/adaptive_fusion.py` | Modality attention network predicting $\alpha_v, \alpha_a$; implements zero-bypassing missing audio fallback. | `torch.nn` |
| **Classifier Head** | `models/classifier.py` | 2-layer MLP with BatchNorm and Dropout(0.3) mapping 512-D fused embedding to binary logits $[z_0, z_1]$. | `torch.nn` |
| **Evidence Synthesizer**| `agent/structured_evidence.py` | Formulates immutable `StructuredForensicEvidence` payload from detector telemetry. | `dataclasses` |
| **Forensic Reasoner** | `agent/reasoning.py` | Deterministic multi-stage rule engine mapping telemetry to grounded visual, acoustic, and consistency bullets. | Internal heuristics |
| **LLM Agent Layer** | `agent/llm_forensic_agent.py` | Provider-agnostic client (Groq, OpenAI, Gemini) translating structured evidence into natural language report. | `requests`, `dotenv` |
| **Report Generator** | `agent/report_generator.py` | Generates formatted Markdown and printable forensic analysis reports. | Text templating |
| **Web Dashboard UI** | `app/static/` | Single-page HTML5/CSS3/Vanilla JS application providing interactive drag-and-drop and visualization. | Browser native |

---

## 4. Algorithmic Pseudocode: End-to-End Multimodal Forensic Inference

```python
def end_to_end_multimodal_inference(video_file_path):
    """
    Complete algorithmic pipeline for the 3V3 Agentic Multimodal Deepfake Detector.
    """
    # -------------------------------------------------------------
    # 1. Ingestion & Quality Validation
    # -------------------------------------------------------------
    is_valid, msg = validate_video_file(video_file_path)
    if not is_valid:
        raise ValueError(f"Invalid input file: {msg}")

    # -------------------------------------------------------------
    # 2. Visual & Acoustic Preprocessing
    # -------------------------------------------------------------
    # Extract 16 uniform frames (T=16, C=3, H=224, W=224)
    raw_frames = opencv_sample_uniform_frames(video_file_path, num_frames=16)
    face_boxes = insightface_scrfd_detect(raw_frames)
    expanded_boxes = [expand_bounding_box(box, padding=0.15) for box in face_boxes]
    cropped_faces = [crop_and_resize(f, b, size=(224, 224)) for f, b in zip(raw_frames, expanded_boxes)]
    images_tensor = imagenet_normalize(cropped_faces)  # Shape: (1, 16, 3, 224, 224)

    # Extract lower facial mouth ROIs
    mouth_crops = [f[:, 112:224, 56:168] for f in images_tensor[0]]
    mouth_crops_resized = bilinear_resize(mouth_crops, size=(224, 224)) # Shape: (1, 16, 3, 224, 224)

    # Ingest and validate 16 kHz audio waveform
    waveform, sample_rate = ffmpeg_extract_mono_audio(video_file_path, target_sr=16000)
    audio_available = check_audio_validity(waveform, min_energy=1e-7, min_sec=0.5)
    waveform_tensor = pad_or_trim(waveform, length=160000)  # Shape: (1, 160000)

    # Extract input quality metrics
    quality_metrics = analyze_input_quality(raw_frames, expanded_boxes, waveform, audio_available)

    # -------------------------------------------------------------
    # 3. Deep Feature Extraction (6 Foundation Encoders)
    # -------------------------------------------------------------
    with torch.inference_mode():
        # Vision Encoders (Frozen)
        f_eff   = efficientnet_b0_forward(images_tensor)            # (1, 1280)
        f_swin  = video_swin_tiny_forward(images_tensor)            # (1, 768)
        f_mouth = resnet18_mouth_forward(mouth_crops_resized)       # (1, 512)

        # Vision Aggregation & Projection
        f_v_concat = concat([f_eff, f_swin, f_mouth], dim=1)        # (1, 2560)
        f_v        = vision_aggregator_linear(f_v_concat)           # (1, 1280)
        e_v        = l2_normalize(vision_projector(f_v))            # (1, 256)

        # Audio Encoders (Frozen, with Dynamic Fallback)
        if audio_available:
            f_whisper = whisper_base_forward(waveform_tensor)       # (1, 512)
            f_w2v2    = wav2vec2_base_forward(waveform_tensor)      # (1, 768)
            f_spec    = spectrogram_cnn_forward(waveform_tensor)    # (1, 256)

            f_a_concat = concat([f_whisper, f_w2v2, f_spec], dim=1) # (1, 1536)
            f_a        = audio_aggregator_linear(f_a_concat)        # (1, 512)
            e_a        = l2_normalize(audio_projector(f_a))         # (1, 256)

            # Audio-Visual Consistency
            av_consistency = dot_product(e_v, e_a)                  # Cosine S_AV in [-1, +1]

            # Adaptive Attention Fusion
            attention_weights = softmax(attention_mlp(concat([e_v, e_a], dim=1)))
            alpha_v, alpha_a  = attention_weights[0, 0], attention_weights[0, 1]
            e_fused           = concat([alpha_v * e_v, alpha_a * e_a], dim=1) # (1, 512)
        else:
            # Missing Audio Fallback: Route 100% to Visual Stream
            e_a            = zeros_like(e_v)
            av_consistency = None
            alpha_v        = 1.000
            alpha_a        = 0.000
            e_fused        = concat([1.0 * e_v, zeros_like(e_v)], dim=1)     # (1, 512)

        # ---------------------------------------------------------
        # 4. Classification & Post-Hoc Calibration
        # ---------------------------------------------------------
        logits = deepfake_classifier_mlp(e_fused)                   # (1, 2)
        raw_probs = softmax(logits, dim=1)
        p_real, p_fake = raw_probs[0, 0].item(), raw_probs[0, 1].item()

        verdict = "FAKE" if p_fake >= 0.50 else "REAL"
        confidence_raw = max(p_real, p_fake)

        # Temperature Scaling Calibration (T = 0.7924)
        calibrated_probs = softmax(logits / 0.7924, dim=1)
        confidence_calibrated = max(calibrated_probs[0, 0].item(), calibrated_probs[0, 1].item())

        # ---------------------------------------------------------
        # 5. OOD & Forensic Reliability Determination
        # ---------------------------------------------------------
        dist_ood = compute_mahalanobis_distance(e_fused, validation_centroids, validation_variances)
        is_ood = dist_ood > 27.93  # 95th-percentile threshold

        if is_ood or quality_metrics["face_area_ratio"] < 0.05:
            reliability = "LOW"
        elif not audio_available or (av_consistency is not None and av_consistency < 0.1177):
            reliability = "MEDIUM"
        else:
            reliability = "HIGH"

    # -------------------------------------------------------------
    # 6. Structured Evidence Synthesis & Forensic Reasoning
    # -------------------------------------------------------------
    structured_evidence = build_structured_evidence(
        verdict=verdict,
        confidence=confidence_raw,
        calibrated_confidence=confidence_calibrated,
        prob_real=p_real,
        prob_fake=p_fake,
        audio_available=audio_available,
        vision_weight=alpha_v,
        audio_weight=alpha_a,
        av_similarity=av_consistency,
        input_quality=quality_metrics,
        reliability=reliability,
        ood_distance=dist_ood,
    )

    # -------------------------------------------------------------
    # 7. LLM Narrative Explanation (Grounded, Anti-Hallucinatory)
    # -------------------------------------------------------------
    if llm_agent.is_configured():
        explanation = llm_agent.generate_explanation(structured_evidence)
    else:
        explanation = deterministic_rule_reasoner.generate_explanation(structured_evidence)

    return {
        "verdict": verdict,
        "confidence": confidence_raw,
        "calibrated_confidence": confidence_calibrated,
        "reliability": reliability,
        "structured_evidence": structured_evidence,
        "forensic_explanation": explanation,
    }
```

---

## 5. Flowcharts and Technical Architecture Figures for Report

When formatting the final academic report, the following five technical diagrams should be included:

### Figure 1: Comprehensive 3V3 Foundation Architecture
- **Description**: Detailed block diagram illustrating the two tripartite streams (3 Vision: EfficientNet-B0, Video Swin-Tiny, Face/Mouth ResNet-18; 3 Audio: Whisper Base, Wav2Vec2 Base, Spectrogram CNN), their aggregators, projection heads, InfoNCE contrastive alignment, adaptive attention fusion, and final MLP classifier.
- **Reference in Text**: Section 5 (System Architecture).

### Figure 2: End-to-End Operational Pipeline & Service Architecture
- **Description**: Multi-tier architecture diagram showing Client Web Browser $\to$ FastAPI Ingestion Gateway $\to$ VideoProcessor (SCRFD Face Detection + FFmpeg) $\to$ Singleton InferenceService $\to$ Temperature Calibration & OOD Detector $\to$ Structured Forensic Evidence Contract $\to$ LLM / Rule Reasoning Engine $\to$ Forensic Dashboard.
- **Reference in Text**: Section 10 (Application Architecture).

### Figure 3: Missing-Audio Dynamic Attention Fallback Mechanism
- **Description**: State machine showing how `AudioDetector` tests waveform energy ($\max(|x|) < 10^{-7}$), branches into normal multimodal mode vs. vision-only fallback, deterministically setting $\alpha_v = 1.0, \alpha_a = 0.0$ and appending zero vectors to prevent tensor shape corruption and ensure zero crashes.
- **Reference in Text**: Section 9 (Robustness & Missing Modalities).

### Figure 4: Cross-Modal Inconsistency vs. Generative Deepfake Disentanglement
- **Description**: Scatter plot and conceptual diagram illustrating the dual-axis forensic decision space: Axis X represents the Deepfake Classifier Logits $P(\text{Fake})$, and Axis Y represents the Audio-Visual Consistency Score $S_{AV}$. Shows four quadrants: (1) Authentic Synchronized, (2) Deepfake with Synthetic Audio, (3) Dubbed / Foreign Voice-Over Authentic Media ($S_{AV} < 0.1177$ but $P(\text{Fake}) < 0.50$), and (4) Out-of-Sync Deepfake.
- **Reference in Text**: Section 6 & 8 (Forensic Consistency Analysis).

### Figure 5: Reliability & OOD Distance Calibration Protocol
- **Description**: Visualization of the Mahalanobis representation distance distribution fitted on 2,414 validation samples, displaying the 95th-percentile (27.93) and 99th-percentile (33.17) decision thresholds, alongside reliability tier boundaries (HIGH, MEDIUM, LOW) and the reliability of external screen-captured test cases.
- **Reference in Text**: Section 8 & 9 (Phase 8 OOD & Calibration).
