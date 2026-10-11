# Comprehensive Inference Latency Audit Report

## 1. Measurement Scope Clarification
The previously reported Phase 6 latency values (`0.20 ms/sample` for Multimodal Fusion and `0.01 ms/sample` for Vision-Only Fallback) represent **GPU In-Memory Feature Aggregation, Attention Fusion, and Classifier Logit Calculation** on precomputed feature tensors.

When operating on raw, unprocessed `.mp4` video files in deployment, inference encompasses three distinct computational stages:

```
[Raw MP4 Video & Audio]
          │
          ▼  Stage A: Preprocessing (~22.0 ms)
   [16 Video Frames + 16 Mouth ROIs + 16kHz Audio Waveform]
          │
          ▼  Stage B: 6 Deep Encoders (~367.1 ms)
   [EfficientNet + Video Swin + Face/Mouth + Whisper + Wav2Vec2 + Spectrogram CNN]
          │
          ▼  Stage C: Aggregation, Adaptive Fusion & Classifier (~3.84 ms)
   [Deepfake Prediction (Real/Fake) + Audio-Visual Consistency Score]
```

---

## 2. Empirical Benchmark Breakdown (Averaged Over 5 Videos)

| Pipeline Stage | Operation Description | Mean Latency | Percentage of Total Time |
| :--- | :--- | :---: | :---: |
| **Stage A: Preprocessing** | OpenCV video decoding, uniform 16-frame extraction, face/mouth landmark cropping, audio demuxing & 16kHz resampling | **`22.04 ms`** | `5.6%` |
| **Stage B: 6-Encoder Extraction** | Forward passes through EfficientNet-B0 (1280-D), Video Swin-T (768-D), Face/Mouth ResNet-18 (512-D), Whisper (512-D), Wav2Vec2 (768-D), Spectrogram CNN (256-D) | **`367.11 ms`** | `93.4%` |
| **Stage C: Fusion & Classifier** | Linear Aggregations (2560->1280, 1536->512), Projection Heads (1280->256, 512->256), Cross-modal Attention Fusion, Deepfake MLP | **`3.84 ms`** | `0.98%` |
| **Stage D: Complete End-to-End** | **Total Raw MP4 to Classification & Consistency Output** | **`392.98 ms`** | **`100.0%`** |

---

## 3. Deployment Throughput
- **End-to-End Raw Video Processing Rate**: `2.54 videos / second` (`393.0 ms / video`)
- **Cached-Feature Fusion Throughput**: `260.3 embeddings / second` (`3.84 ms / embedding`)
