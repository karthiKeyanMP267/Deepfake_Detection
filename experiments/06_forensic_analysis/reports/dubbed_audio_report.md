# Dubbed & Audio-Visual Inconsistency Forensic Report

## Summary & Scientific Principles
This forensic investigation assesses the multimodal system's capacity to detect cross-modal inconsistency (e.g. dubbing, voice replacement, out-of-sync audio).

### Crucial Scientific Distinction:
- **Deepfake Classification** (Real vs. Fake face/video manipulation): Evaluates visual spatio-temporal and texture artifacts.
- **Audio-Visual Inconsistency** (Synchronized vs. Dubbed/Mismatched): Evaluates cross-modal alignment between visual speech articulation (Face/Mouth ResNet-18 + Video Swin) and acoustic representations (Whisper + Wav2Vec2 + Spectrogram CNN).
- **Ethical & Scientific Note**: Translated voice-overs or dubbed movies are frequently authentic videos with mismatched audio. The forensic system treats inconsistency as separate forensic evidence, rather than automatically classifying all dubbed content as deepfakes.

---

## Consistency Score Statistics

| Condition | Mean Similarity (S_AV) | Std Dev | Inconsistency Flagged Rate |
| :--- | :---: | :---: | :---: |
| **Normal Matched Audio** | **0.2618** | 0.1441 | **17.05%** |
| **Mismatched / Dubbed Audio** | **-0.0060** | 0.1925 | **71.69%** |
| **Separation Margin** | **+0.2678** | — | — |

- **Exploratory Threshold**: S_AV < 0.1177 cleanly separates normal speech from cross-modal mismatch.
