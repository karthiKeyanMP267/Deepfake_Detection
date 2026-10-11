# Key Scientific & Engineering Research Contributions

## 1. Summary of Major Contributions

### 1. Hexa-Modal Foundation Architecture (Full 3V3)
Designed and implemented an end-to-end multimodal deepfake detector integrating 6 complementary foundation models:
- **Vision (3 Encoders)**: Spatial texture (`EfficientNet-B0`), 3D spatio-temporal self-attention (`Video Swin Transformer`), and perioral articulation (`Face/Mouth ResNet-18`).
- **Audio (3 Encoders)**: Semantic speech (`Whisper Base.en`), self-supervised acoustic representations (`Wav2Vec2 Base`), and time-frequency spectrograms (`Spectrogram 2D CNN`).
- **Performance**: Achieved **97.52% Test Accuracy**, **98.71% F1-Score**, and **99.02% ROC-AUC**.

### 2. Resolution of Severe Class Imbalance (42.5:1 Skew)
Formulated a combined `WeightedRandomSampler` and `Focal Loss` ($\gamma = 2.0, lpha = 0.25$) optimization strategy that eliminated majority-class collapse, elevating Balanced Accuracy from 60.69% to **93.63%** and Authentic F1 from 10.00% to **69.39%**.

### 3. Statistically Rigorous Same-Test-Set Benchmarking
Conducted a strict, controlled comparison on the exact same 604 test samples, demonstrating that integrating 3D spatio-temporal features yielded **+24.34% Accuracy** and **+38.81% ROC-AUC** over the baseline with overwhelming statistical significance ($p < 10^{-15}$).

### 4. Non-Crashing Missing-Audio Fallback Protocol
Developed an autonomous audio detector and dynamic modality attention routing mechanism ($lpha_v = 1.0, lpha_a = 0.0$) that executes flawlessly when audio is absent or corrupted, preserving **97.68% accuracy** with a **+95.9% GPU fusion speedup**.

### 5. Disentanglement of Dubbing from Deepfakes
Formulated an embedding-level consistency metric $S_{AV} = \mathbf{e}_v^T \mathbf{e}_a$ providing a **+0.2678 separation margin** between synchronized ($+0.2618$) and dubbed speech ($-0.0060$), establishing a dual-channel forensic architecture that avoids misclassifying foreign translations as deepfakes.

### 6. Autonomous Agentic Reasoning & OOD Reliability Layer
Engineered a deterministic, rule-based forensic reasoning agent (`ForensicAgent`) that synthesizes grounded visual, acoustic, and cross-modal evidence into structured reports, featuring an autonomous **12-criterion domain reliability layer** that flags out-of-distribution screen captures.
