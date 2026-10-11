# Final Agentic Multimodal Deepfake Detection Architecture (Full 3V3)

## 1. Architectural Diagram

```
                              INPUT VIDEO (MP4)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
       VISION STREAM                                    AUDIO STREAM
  [16 Uniform Video Frames +                       [160,000 Sample Raw Waveform
   16 Mouth/Lip Region Crops]                                (16 kHz)]
              │                                               │
  ┌───────────┼───────────┐                       ┌───────────┼───────────┐
  ▼           ▼           ▼                       ▼           ▼           ▼
EfficientNet  Video Swin  Face/Mouth             Whisper    Wav2Vec2    Spectrogram
  B0 (2D)     Transformer ResNet-18              Base-EN    2.0-Base    2D Audio CNN
 (1280-D)      (768-D)     (512-D)               (512-D)     (768-D)      (256-D)
 [Spatial]   [Temporal]   [Lip Sync]            [Semantic]  [Acoustic]  [Time-Freq]
  [Frozen]    [Frozen]    [Frozen]               [Frozen]    [Frozen]     [Frozen]
      │           │           │                      │           │           │
      └───────────┼───────────┘                      └───────────┼───────────┘
                  ▼                                              ▼
       Vision Feature Aggregation                     Audio Feature Aggregation
         Linear(2560 -> 1280)                           Linear(1536 -> 512)
                  │                                              │
                  ▼                                              ▼
        Vision Projection Head                         Audio Projection Head
         Linear(1280 -> 256)                            Linear(512 -> 256)
         + BatchNorm + L2 Norm                          + BatchNorm + L2 Norm
                  │                                              │
                  ├──────────────────────┬───────────────────────┤
                  │                      ▼                       │
                  │             Contrastive Module               │
                  │            InfoNCE Cross-Modal Loss          │
                  │         S_AV = cos(v_emb, a_emb)             │
                  │                                              │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                             Adaptive Multimodal Fusion
                             Attention Network(512 -> 2)
                        w_v, w_a = Softmax(Linear(combined))
                        Fused = [w_v * v_emb, w_a * a_emb]
                                         │
                                         ▼
                                Deepfake Classifier
                           Linear(512 -> 256) -> BN -> ReLU
                           Linear(256 -> 128) -> BN -> ReLU
                                Linear(128 -> 2)
                                         │
                                         ▼
                           Dual Forensic Output Streams
                            /                          \
             Deepfake Classification            Cross-Modal Inconsistency
             Probability & Real/Fake            Synchronization Score S_AV
```

---

## 2. Encoder Role Matrix

| Branch | Model Backbone | Dimension | Specialization & Artifact Targeted | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Vision 1** | `EfficientNet-B0` | 1280-D | Frame-level spatial textures, blending seams, warping | **FROZEN** |
| **Vision 2** | `VideoSwin-Tiny` | 768-D | 3D Spatio-temporal inter-frame coherence, head motion | **FROZEN** |
| **Vision 3** | `FaceMouth-ResNet18`| 512-D | Lip contour dynamics, mouth ROI articulation | **FROZEN** |
| **Audio 1** | `Whisper-Base.en` | 512-D | High-level speech semantics, phonetic structure | **FROZEN** |
| **Audio 2** | `Wav2Vec2-Base` | 768-D | Low-level acoustic synthesis artifacts, vocoder glitches | **FROZEN** |
| **Audio 3** | `Spectrogram 2D CNN`| 256-D | 128-band log-Mel time-frequency spectral anomalies | **FROZEN** |

---

## 3. Trainable Multimodal Components (8.76M Params)
1. **Vision Aggregation**: Compresses concatenated 2560-D vision vectors to 1280-D.
2. **Audio Aggregation**: Compresses concatenated 1536-D audio vectors to 512-D.
3. **Projection Heads**: Maps aggregated features into shared 256-D metric space for InfoNCE alignment.
4. **Adaptive Fusion**: Self-attention mechanism dynamically computing modality trust weights (w_v, w_a).
5. **Deepfake Classifier**: Multi-layer perceptron outputting classification logits.
