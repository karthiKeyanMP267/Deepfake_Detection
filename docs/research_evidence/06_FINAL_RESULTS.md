# Final Experimental Results & Benchmark Performance

## 1. Definitive Research Model Identification
The definitive research model established across all project benchmarks is the **Phase 5 Full 3V3 Multimodal Deepfake Detector**.
- **Final Model Checkpoint**: `D:\Deepfake\Agentic-Multimodal-Deepfake-Detection\experiments\05_full_3v3\models\best_model.pth`
- **Optimal Training Epoch**: Epoch 4 (Validation F1: 0.9722)
- **Active Encoders (6 Total)**:
  - Vision 1: EfficientNet-B0 (1280-D) [Frozen]
  - Vision 2: Video Swin Transformer (768-D) [Frozen]
  - Vision 3: Face/Mouth ResNet-18 (512-D) [Frozen]
  - Audio 4: Whisper Base.en (512-D) [Frozen]
  - Audio 5: Wav2Vec2 Base (768-D) [Frozen]
  - Audio 6: Spectrogram 2D CNN (256-D) [Trainable]
- **Trainable Subsystems**: Vision Aggregator (2560 $	o$ 1280), Audio Aggregator (1536 $	o$ 512), Projection Heads (256-D), Adaptive Fusion (512-D), Deepfake Classifier MLP.

---

## 2. Test Set Benchmark Metrics (604 Samples)

The final model was evaluated on the exact 604 test samples from `test_manifest.csv` (Real: 19, Fake: 585) under a fixed classification threshold ($t = 0.50$):

```
===================================================================================================
METRIC                                  EXACT VALUE          PERCENTAGE          SCIENTIFIC INTERPRETATION
===================================================================================================
Accuracy                                0.975166             97.52%              Overall correct classification rate
F1-Score (Macro / Micro)                0.987058             98.71%              Harmonic mean of precision and recall
ROC-AUC (Area Under ROC Curve)          0.990238             99.02%              Discriminative ranking ability
Balanced Accuracy                       0.936257             93.63%              Unweighted mean of class recalls
Sensitivity (True Positive Rate)        0.977778             97.78%              Correct detection of manipulated media
Specificity (True Negative Rate)        0.894737             89.47%              Correct detection of authentic media
Positive Predictive Value (Fake Prec)   0.996516             99.65%              Probability that a flagged sample is Fake
Negative Predictive Value (Real Prec)   0.566667             56.67%              Probability that a cleared sample is Real
Fake F1-Score                           0.987058             98.71%              F1 performance on manipulated class
Real F1-Score                           0.693878             69.39%              F1 performance on minority authentic class
===================================================================================================
```

---

## 3. Confusion Matrix Breakdown

### Absolute Counts:
```
                         PREDICTED AUTHENTIC (0)       PREDICTED MANIPULATED (1)       TOTAL
ACTUAL AUTHENTIC (0)           TN = 17                        FP = 2                     19
ACTUAL MANIPULATED (1)         FN = 13                       TP = 572                   585
TOTAL                          30                             574                       604
```

### Normalized Rates:
- **True Negative Rate (Specificity)**: $rac{17}{19} = \mathbf{89.47\%}$
- **False Positive Rate (False Alarm on Real)**: $rac{2}{19} = \mathbf{10.53\%}$
- **False Negative Rate (Missed Deepfake)**: $rac{13}{585} = \mathbf{2.22\%}$
- **True Positive Rate (Sensitivity)**: $rac{572}{585} = \mathbf{97.78\%}$

---

## 4. Classification Report (Scikit-Learn Standard Format)

```
              precision    recall  f1-score   support

        Real     0.5667    0.8947    0.6939        19
        Fake     0.9965    0.9778    0.9871       585

    accuracy                         0.9752       604
   macro avg     0.7816    0.9363    0.8405       604
weighted avg     0.9830    0.9752    0.9778       604
```

---

## 5. Execution Latency & Computational Efficiency

An exhaustive latency audit (`experiments/06_forensic_analysis/reports/latency_audit.md`) established the distinction between raw GPU tensor throughput and end-to-end media inference:

```
===================================================================================================
PROCESSING STAGE                                      LATENCY (PER SAMPLE)        THROUGHPUT
===================================================================================================
Stage A: Raw Video/Audio Ingestion & Preprocessing     22.04 ms                    --
Stage B: 6 Foundation Encoders Feature Extraction     367.11 ms                   --
Stage C: GPU Multimodal Fusion & Classifier Pass        3.84 ms                   260.3 samples/sec
---------------------------------------------------------------------------------------------------
Total End-to-End Raw MP4 Inference Latency:           392.98 ms / video           2.54 FPS / video
GPU In-Memory Precomputed 1D Feature Latency:           0.20 ms / sample        5,000.0 samples/sec
===================================================================================================
```
