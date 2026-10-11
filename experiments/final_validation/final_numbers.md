# Final Verified Project Numbers Reference Table

| Category | Parameter / Metric | Verified Value |
| :--- | :--- | :---: |
| **Model Architecture** | Total Encoders | **6 Encoders (3 Vision + 3 Audio)** |
| | Total Parameters | **166,440,678** |
| | Frozen Parameters | **157,682,658 (94.74%)** |
| | Trainable Parameters | **8,758,020 (5.26%)** |
| **Final Test Performance (3V3)** | Test Samples | **604 samples** |
| | **Accuracy** | **97.52%** |
| | **Precision** | **99.65%** |
| | **Recall** | **97.78%** |
| | **F1 Score** | **98.71%** |
| | **ROC-AUC** | **99.02%** |
| | **Balanced Accuracy** | **93.63%** |
| | **Real Class F1** | **69.39%** |
| | **Fake Class F1** | **98.71%** |
| | **Sensitivity / Specificity** | **97.78% / 89.47%** |
| **Missing Audio Fallback** | Vision-Only Accuracy | **97.68%** |
| | Vision-Only F1 Score | **98.80%** |
| | Vision-Only Real F1 | **65.00%** |
| | Assigned Weights (w_v / w_a) | **1.000 / 0.000** |
| **Forensic AV Dubbing** | Mean Synchronized S_AV | **+0.2618** |
| | Mean Mismatched S_AV | **-0.0060** |
| | Separation Margin | **+0.2678** |
| **Inference Latency Audit** | Preprocessing (Decode/Crop) | **22.04 ms / video** |
| | 6-Encoder Forward Passes | **367.11 ms / video** |
| | Fusion & Classifier Pass | **3.841 ms / sample** |
| | End-to-End Raw MP4 Latency | **392.98 ms / video** |
