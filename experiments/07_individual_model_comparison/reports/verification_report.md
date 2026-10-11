# Independent Verification Audit Report
## Frozen-Backbone Standalone Encoders (6 Models)

- **Audit Timestamp**: 2026-09-10 19:06:23
- **Verification Tolerance**: < 1e-6
- **Test Manifest Samples**: 604

| Model | Prediction Count | Samples Match | Labels Match | Metrics Discrepancy | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| EfficientNet-B0 | 604 | YES | YES | 0.00e+00 | **PASSED** |
| Video Swin-Tiny | 604 | YES | YES | 0.00e+00 | **PASSED** |
| Face/Mouth ResNet-18 | 604 | YES | YES | 0.00e+00 | **PASSED** |
| Whisper Base | 604 | YES | YES | 0.00e+00 | **PASSED** |
| Wav2Vec2 Base | 604 | YES | YES | 0.00e+00 | **PASSED** |
| Spectrogram CNN | 604 | YES | YES | 0.00e+00 | **PASSED** |

## Metric-by-Metric Numerical Tolerances

### EfficientNet-B0
- Confusion Matrix: TN=11, FP=8, FN=9, TP=576
- Accuracy: 0.971854
- F1-Score: 0.985458
- ROC-AUC: 0.917994
- Balanced Accuracy: 0.781781
- Status: VERIFIED

### Video Swin-Tiny
- Confusion Matrix: TN=11, FP=8, FN=10, TP=575
- Accuracy: 0.970199
- F1-Score: 0.984589
- ROC-AUC: 0.943455
- Balanced Accuracy: 0.780927
- Status: VERIFIED

### Face/Mouth ResNet-18
- Confusion Matrix: TN=12, FP=7, FN=12, TP=573
- Accuracy: 0.968543
- F1-Score: 0.983691
- ROC-AUC: 0.966667
- Balanced Accuracy: 0.805533
- Status: VERIFIED

### Whisper Base
- Confusion Matrix: TN=7, FP=12, FN=113, TP=472
- Accuracy: 0.793046
- F1-Score: 0.883068
- ROC-AUC: 0.768826
- Balanced Accuracy: 0.587629
- Status: VERIFIED

### Wav2Vec2 Base
- Confusion Matrix: TN=15, FP=4, FN=208, TP=377
- Accuracy: 0.649007
- F1-Score: 0.780538
- ROC-AUC: 0.780702
- Balanced Accuracy: 0.716959
- Status: VERIFIED

### Spectrogram CNN
- Confusion Matrix: TN=19, FP=0, FN=585, TP=0
- Accuracy: 0.031457
- F1-Score: 0.000000
- ROC-AUC: 0.303419
- Balanced Accuracy: 0.500000
- Status: VERIFIED

