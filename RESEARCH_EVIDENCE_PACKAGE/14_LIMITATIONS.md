# Academic Limitations & Future Research Directions

## 1. Scientific & Operational Limitations

### A. Minority Class Sample Size in Benchmark Split
While the full dataset comprises 14,284 samples, the stratified 15% testing split contains 604 samples (19 Real, 585 Fake). Although this reflects real-world operational distributions where fraudulent submissions dominate, the sample size of 19 authentic videos increases the discrete granularity of specificity measurements (each real video represents $pprox 5.26\%$ of the specificity metric).

### B. Cross-Dataset Generalization & In-the-Wild Domain Shift
As demonstrated in the OOD investigation (`10_OOD_ANALYSIS.md`), the 6 encoders were trained primarily on FakeAVCeleb manipulation signatures. When presented with external screen recordings or unseen generative engines (e.g., SadTalker, LivePortrait, ElevenLabs), lossy compression quantization smooths high-frequency boundary seams, causing neural softmax outputs to saturate toward authentic predictions.

### C. Embedding-Level vs. Phonemic Token-Level Synchronization
The audio-visual consistency metric $S_{AV} = \mathbf{e}_v^T \mathbf{e}_a$ evaluates global embedding alignment. It detects general speech-gesture correlation and cross-modal mismatch effectively ($S_{AV} = -0.0060$ vs $+0.2618$), but is not a replacement for token-level phoneme-viseme temporal alignment tools (e.g., SyncNet).

---

## 2. Recommended Future Research Directions
1. **Multi-Dataset Training**: Expand training corpora across FaceForensics++, DFDC (DeepFake Detection Challenge), Celeb-DF, and WildDeepfake to expose encoders to diverse generator architectures.
2. **Domain-Generalization Data Augmentation**: Integrate JPEG compression jitter, Gaussian blur filtering, spatial downscaling, and audio codec perturbations during training.
3. **Uncertainty Calibration & Temperature Scaling**: Implement Monte Carlo Dropout or conformal prediction to prevent softmax overconfidence on out-of-distribution media.
4. **Phoneme-Viseme Cross-Attention**: Incorporate fine-grained cross-attention mechanisms between video frame tokens and audio Mel-spectrogram timeframes.
