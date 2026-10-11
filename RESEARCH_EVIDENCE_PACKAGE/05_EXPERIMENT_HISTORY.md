# Comprehensive Experiment Progression & Historical Phases

## 1. Chronological Research Evolution
The development of the Agentic Multimodal Deepfake Detection framework progressed through distinct phases designed to isolate the contribution of each architectural component, solve extreme dataset class imbalance, and evaluate forensic robustness.

```
+----------------------------------------------------------------------------------------------------+
|                                  EXPERIMENT PROGRESSION TIMELINE                                   |
+----------------------------------------------------------------------------------------------------+
  Phase 1: Baseline (Unweighted 1V1)         --> Accuracy: 7.04%, F1: 9.22%, AUC: 68.65%
     |  (Collapsed due to 97.7% class imbalance; predicted all Fake samples as Real)
     v
  Phase 2A: Class Weighting (Weighted CE)    --> Accuracy: 97.68%, F1: 98.83%, AUC: 68.40%, Real F1: 0.0%
     |  (Loss weighting inverted collapse; predicted 100% Fake samples, 0% Real recall)
     v
  Phase 2B: Weighted Random Sampler (1V1)    --> Accuracy: 74.91%, F1: 85.40%, AUC: 70.10%, Real F1: 10.6%
     |  (Balanced batch exposure; stabilized training, established robust 1V1 foundation)
     v
  Phase 3: Video Swin Transformer (2V1A)     --> Accuracy: 97.52%, F1: 98.71%, AUC: 98.80%, Real F1: 68.97%
     |  (Added 3D Swin spatio-temporal features; +24.34% Accuracy & +38.81% AUC on same test set)
     v
  Phase 4: Partial Fine-Tuning (2V1A)        --> Accuracy: 97.52%, F1: 98.71%, AUC: 98.25%, Real F1: 69.39%
     |  (Unfroze top layers of backbones with differential LR; balanced accuracy reached 93.63%)
     v
  Phase 5: Full 3V3 Architecture (6 Models)  --> Accuracy: 97.52%, F1: 98.71%, AUC: 99.02%, Real F1: 69.39%
     |  (Full 3-Vision + 3-Audio: EffNet, Swin, ResNet-18, Whisper, Wav2Vec2, SpecCNN; highest AUC)
     v
  Phase 6: Forensic & Robustness Audits      --> Missing Audio: 97.68% Acc (+95.9% speedup)
                                             --> Dubbed Audio: S_AV separation margin = +0.2678
```

---

## 2. Detailed Phase-by-Phase Breakdown

### Phase 1: Unweighted Baseline (1V1)
- **Architecture**: `EfficientNet-B0` (1280-D) + `Whisper Base` (512-D), Adaptive Fusion, Focal Loss, No Sampler.
- **Dataset**: FakeAVCeleb (11,266 train, 2,414 val, 604 test; 97.7% Fake / 2.3% Real).
- **Epochs**: 5 epochs (Best epoch: 1).
- **Measured Metrics**:
  - Accuracy: **7.04%** (0.0704)
  - Precision: **100.00%** (1.0000)
  - Recall: **4.83%** (0.0483)
  - F1-Score: **9.22%** (0.0922)
  - ROC-AUC: **68.65%** (0.6865)
  - Real F1: **4.75%** | Fake F1: **9.22%**
- **Scientific Observation**: Due to severe class imbalance without sampling compensation, the gradients were overwhelmed by majority class dynamics, causing severe underfitting on the minority authentic class.

---

### Phase 2A: Class-Weighted Loss (1V1)
- **Architecture**: `EfficientNet-B0` + `Whisper Base`, Class-Weighted Cross-Entropy Loss ($w_{	ext{real}} = 21.75, w_{	ext{fake}} = 0.51$), No Sampler.
- **Epochs**: 6 epochs (Best epoch: 1).
- **Measured Metrics**:
  - Accuracy: **97.68%** (0.9768)
  - Precision: **97.68%**
  - Recall: **100.00%**
  - F1-Score: **98.83%**
  - ROC-AUC: **68.40%** (0.6840)
  - Real F1: **0.00%** (Recall Real: 0.0%) | Fake F1: **98.83%**
- **Scientific Observation**: While overall accuracy appeared high (97.68%), this metric was illusory; the model classified 100% of samples as Fake, failing completely to detect a single authentic video (Real Recall = 0.0%, Real F1 = 0.00%).

---

### Phase 2B: Weighted Random Sampler Baseline (1V1)
- **Architecture**: `EfficientNet-B0` + `Whisper Base`, Focal Loss ($lpha=0.25, \gamma=2.0$) + `WeightedRandomSampler`.
- **Epochs**: 10 epochs (Best epoch: 5).
- **Measured Metrics** (General Test / Same Test Set 604):
  - Accuracy: **74.91%** (73.18% on same-test-set 604)
  - Precision: **98.88%** (97.74%)
  - Recall: **75.16%** (74.02%)
  - F1-Score: **85.40%** (84.24%)
  - ROC-AUC: **70.10%** (59.99%)
  - Real F1: **10.62%** (10.00%) | Fake F1: **85.40%** (84.24%)
- **Scientific Observation**: Batch-level balanced sampling resolved minority collapse, achieving 64.29% recall on real samples and establishing a true learning baseline.

---

### Phase 3: Video Swin Transformer Spatio-Temporal Integration (2V1A)
- **Architecture**: `EfficientNet-B0` (1280-D) + `Video Swin Transformer` (768-D) + `Whisper Base` (512-D), Vision Aggregator (`Linear(2048, 1280)`), `WeightedRandomSampler`, Focal Loss.
- **Epochs**: 15 epochs (Best epoch: 12, Validation F1: 0.9828).
- **Measured Metrics** (604 Test Samples):
  - Accuracy: **97.52%** (0.9752)
  - Precision: **99.48%** (0.9948)
  - Recall: **97.95%** (0.9795)
  - F1-Score: **98.71%** (0.9871)
  - ROC-AUC: **98.80%** (0.9880)
  - Balanced Accuracy: **91.08%** (0.9108)
  - Real Precision / Recall / F1: **57.14% / 84.21% / 68.97%**
  - Fake Precision / Recall / F1: **99.48% / 97.95% / 98.71%**
- **Confusion Matrix**: $	ext{TN} = 16, 	ext{FP} = 3, 	ext{FN} = 12, 	ext{TP} = 573$
- **Scientific Gain**: Integrating 3D spatio-temporal attention alongside 2D textures improved Accuracy by **+24.34%**, F1 by **+14.47%**, and ROC-AUC by **+38.81%** on identical test data.

---

### Phase 4: Partial Fine-Tuning Strategy (2V1A)
- **Architecture**: `EfficientNet-B0` (unfrozen features[7..8]) + `Video Swin` (unfrozen stage 6 + norm) + `Whisper` (unfrozen blocks 4..5), Grouped AdamW (`lr=1e-4` for heads, `lr=1e-5` for backbones).
- **Epochs**: 15 epochs (Best epoch: 14, Validation F1: 0.9829).
- **Measured Metrics** (604 Test Samples):
  - Accuracy: **97.52%**
  - Precision: **99.65%**
  - Recall: **97.78%**
  - F1-Score: **98.71%**
  - ROC-AUC: **98.25%**
  - Balanced Accuracy: **93.63%**
  - Real F1: **69.39%** (Recall Real: **89.47%**, 17/19) | Fake F1: **98.71%**
- **Confusion Matrix**: $	ext{TN} = 17, 	ext{FP} = 2, 	ext{FN} = 13, 	ext{TP} = 572$
- **Scientific Gain**: Higher specificity on authentic videos ($	ext{Specificity} = 89.47\%$), reducing false alarms on real content.

---

### Phase 5: Full 3V3 Multimodal Architecture (6 Foundation Models)
- **Architecture**: 3 Vision (`EfficientNet-B0` 1280-D + `Video Swin` 768-D + `Face/Mouth ResNet-18` 512-D) + 3 Audio (`Whisper` 512-D + `Wav2Vec2` 768-D + `Spectrogram CNN` 256-D), Vision Aggregator (`Linear(2560, 1280)`), Audio Aggregator (`Linear(1536, 512)`), Adaptive Fusion, InfoNCE Loss.
- **Initial Training Run & Numerical Fix**: During initial extraction with FP16 mixed precision, subnormal underflow in `SpectrogramCNN` ($1e-6 < 6.10e-5$) produced NaNs on 85 audio samples. This was isolated and resolved by regenerating the Spectrogram CNN branch in full FP32 (`float32`) with `torch.nan_to_num`. Fusion training was executed smoothly to completion.
- **Epochs**: 5 epochs (Best epoch: 4, Validation F1: 0.9722).
- **Measured Metrics** (604 Test Samples):
  - Accuracy: **97.52%** (0.9752)
  - Precision: **99.65%** (0.9965)
  - Recall: **97.78%** (0.9778)
  - F1-Score: **98.71%** (0.9871)
  - ROC-AUC: **99.02%** (**Highest achieved in project**)
  - Balanced Accuracy: **93.63%** (0.9363)
  - Sensitivity (Fake Recall): **97.78%** (572 / 585)
  - Specificity (Real Recall): **89.47%** (17 / 19)
  - Real Precision / Recall / F1: **56.67% / 89.47% / 69.39%**
  - Fake Precision / Recall / F1: **99.65% / 97.78% / 98.71%**
- **Confusion Matrix**: $	ext{TN} = 17, 	ext{FP} = 2, 	ext{FN} = 13, 	ext{TP} = 572$
- **Selected Final Checkpoint**: `experiments/05_full_3v3/models/best_model.pth`

---

## 3. Comprehensive Metric Summary Table

| Phase | Vision Encoders | Audio Encoders | Imbalance Strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | Bal Acc | Real F1 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline** | EffNet | Whisper | None (Unweighted) | 7.04% | 100.0% | 4.83% | 9.22% | 68.65% | N/A | 4.75% |
| **2A. Class Wt** | EffNet | Whisper | Loss Weighting | 97.68% | 97.68% | 100.0% | 98.83% | 68.40% | N/A | 0.00% |
| **2B. Sampler** | EffNet | Whisper | WeightedRandomSampler | 74.91% | 98.88% | 75.16% | 85.40% | 70.10% | N/A | 10.62% |
| **3. Video Swin**| EffNet + Swin | Whisper | Sampler + Focal | 97.52% | 99.48% | 97.95% | 98.71% | 98.80% | 91.08% | 68.97% |
| **4. Partial FT**| EffNet + Swin | Whisper | Partial FT + Sampler | 97.52% | 99.65% | 97.78% | 98.71% | 98.25% | 93.63% | 69.39% |
| **5. Full 3V3** | EffNet+Swin+ResNet | Whisper+W2V2+Spec | 6-Model Aggregation | **97.52%** | **99.65%** | **97.78%** | **98.71%** | **99.02%** | **93.63%** | **69.39%** |
