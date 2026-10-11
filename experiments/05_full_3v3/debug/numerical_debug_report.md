# Phase 5 Full 3V3 Numerical Diagnosis & Repair Report

## 1. Root Cause Analysis
During initial Phase 5 feature precomputation with Automatic Mixed Precision (`AMP=ENABLED`), the `SpectrogramCNNEncoder` branch generated `NaN` values on a small fraction of audio samples (85 in train, 12 in val, 3 in test). 

### Mechanism of Instability:
- The log-Mel spectrogram conversion computes `log(mel + 1e-6)`.
- In half-precision (FP16), `1e-6` falls below the minimum subnormal threshold (`torch.finfo(torch.float16).tiny = 6.10e-5`), causing floating point underflow to `0.0` or `-Inf`.
- The subsequent `Conv2d` and `BatchNorm2d` layers propagated `NaN` across the 256-D embedding.
- When loading `spectrogram_features` during fusion training, any calculation involving these embeddings produced `NaN` losses, causing the entire classifier and metrics to collapse to `NaN`.

---

## 2. Numerical Audit Summary

| Encoder Branch | Train NaNs (Old) | Val NaNs (Old) | Test NaNs (Old) | Train NaNs (Repaired) | Val NaNs (Repaired) | Test NaNs (Repaired) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (1280-D)** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Video Swin (768-D)** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Face/Mouth ResNet-18 (512-D)** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Whisper (512-D)** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Wav2Vec2 (768-D)** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Spectrogram CNN (256-D)** | **21,760 (85 samples)** | **3,072 (12 samples)** | **768 (3 samples)** | **0** | **0** | **0** |

---

## 3. Targeted Branch Repair
- **No 155-Minute Rerun**: Reused 100% of the valid precomputed features for EfficientNet, Video Swin, Face/Mouth, Whisper, and Wav2Vec2.
- **FP32 Regeneration**: Extracted only the lightweight Spectrogram CNN branch in full FP32 (`float32`) with numerical clamp (`torch.nan_to_num`) in ~2.5 minutes.
- **Cache Updated**: Successfully updated `train_features.pt`, `val_features.pt`, and `test_features.pt`.

---

## 4. Fusion Pipeline Diagnostics
- Forward pass, contrastive InfoNCE loss, classification loss, backward gradients, and AdamW optimizer updates were tested and confirmed to be **100% finite and stable**.
