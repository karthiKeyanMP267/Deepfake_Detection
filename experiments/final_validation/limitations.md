# Empirical Limitations & Research Constraints

## 1. Dataset Imbalance & Minority Class Sample Size
- The benchmark dataset exhibits severe class imbalance (97.7% Fake vs. 2.3% Real). In the standard 604-sample test split, only 19 Real samples exist.
- While balanced accuracy (93.63%) and specificity (89.47%) are high, the Real F1 score (69.39%) is sensitive to small misclassifications (e.g. 2 False Positives).

## 2. Synthetic Mismatch vs. Natural Dubbing Benchmark
- The dubbed audio experiment utilized a controlled speaker-swap pairing manifest. While rigorously establishing cross-modal separation, evaluation on a dedicated in-the-wild multi-lingual dubbed dataset remains future work.

## 3. Embedding-Level vs. Phoneme-Viseme Lip Sync
- The consistency score S_AV measures 256-D cross-modal semantic/acoustic feature correlation; it is not a phoneme-level viseme verification system.

## 4. Latency Scope
- In-memory feature fusion runs in 0.20 ms/sample on GPU, but complete end-to-end raw video decoding and 6-encoder feature extraction requires ~300–450 ms/sample on consumer hardware.
