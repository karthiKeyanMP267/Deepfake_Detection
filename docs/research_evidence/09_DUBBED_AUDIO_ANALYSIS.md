# Dubbed & Audio-Visual Inconsistency Forensic Analysis

## 1. Research Motivation & Core Scientific Distinction
A fundamental challenge in multimodal forensics is distinguishing between:
1. **Visual Deepfakes (Generative Manipulation)**: Face swaps, neural facial reenactments, diffusion edits, or synthesized audio vocoders.
2. **Audio-Visual Inconsistency (Desynchronization / Dubbing)**: Authentic videos paired with foreign language translations, delayed audio tracks, or replaced speakers (e.g., dubbed movies, interviews).

> [!IMPORTANT]
> **Crucial Scientific Rule**: Translated voice-overs or dubbed movies are frequently authentic videos with mismatched audio. A forensic system must **NEVER** conflate cross-modal inconsistency with visual deepfakes. Dubbing detection is an evidence-grounded **consistency analysis**, not a binary deepfake classifier.

---

## 2. Experimental Methodology: Controlled Mismatch Evaluation
To measure the sensitivity of cross-modal representations to dubbing without retraining:
- **Test Cohort**: 604 test videos from `test_manifest.csv`.
- **Condition 1 (Normal Synchronized)**: Natural video and matched audio from the same recording.
- **Condition 2 (Controlled Mismatched / Dubbed)**: Audio waveforms were deterministically cyclically shifted across sample IDs by offset $k = 302$:
  $$	ext{Audio}_{	ext{mismatched}}(i) = 	ext{Audio}\left( (i + 302) \pmod{604} ight)$$
  pairing every video sequence with an unrelated speaker's speech track.

---

## 3. Audio-Visual Consistency Formulation
Cross-modal alignment is measured via the cosine similarity between unit-normalized visual and acoustic projection embeddings:
$$S_{AV} = \cos(\mathbf{e}_v, \mathbf{e}_a) = \mathbf{e}_v^T \mathbf{e}_a \in [-1.0, +1.0]$$
where $\mathbf{e}_v \in \mathbb{R}^{256}$ represents aggregated visual features (EfficientNet + Video Swin + Face/Mouth ResNet) and $\mathbf{e}_a \in \mathbb{R}^{256}$ represents aggregated speech features (Whisper + Wav2Vec2 + Spectrogram CNN).

---

## 4. Quantitative Results & Distribution Statistics

```
========================================================================================================================
CONDITION                  SAMPLES    MEAN S_AV     STD DEV       MIN S_AV      MAX S_AV      SEPARATION    FLAGGED %
========================================================================================================================
Normal Matched Audio       604        +0.2618       0.1441        -0.1781       +0.6012       Reference     17.05%
Controlled Dubbed Audio    604        -0.0060       0.1925        -0.4525       +0.4704       +0.2678       71.69%
========================================================================================================================
```

### Exploratory Threshold & Separation Margin:
- **Consistency Separation Margin**: $\Delta = \mu_{	ext{matched}} - \mu_{	ext{mismatched}} = +0.2618 - (-0.0060) = \mathbf{+0.2678}$.
- **Exploratory Decision Threshold**: $S_{	ext{threshold}} = 0.1177$ (derived from midpoint separation).
- **Flagged Inconsistent Rate**:
  - Controlled Mismatched / Dubbed: **71.69%** flagged (433 / 604).
  - Normal Synchronized Audio: **17.05%** flagged (103 / 604).

---

## 5. Visual Classifier Robustness Under Dubbing
When evaluated on mismatched/dubbed audio, the visual classifier maintained high deepfake detection performance:
- Accuracy: **97.68%**
- Precision: **99.48%**
- Recall: **98.12%**
- F1-Score: **98.80%**
- ROC-AUC: **98.33%**
- Confusion Matrix: $	ext{TN} = 16, 	ext{FP} = 3, 	ext{FN} = 11, 	ext{TP} = 574$.

---

## 6. Forensic Reporting Disentanglement
The autonomous agent implements a **Dual-Channel Output Protocol**:
1. **Channel 1 (Deepfake Classification)**: Reports whether the visual facial boundaries and acoustic textures contain generative AI manipulation (`REAL` vs. `FAKE`).
2. **Channel 2 (Cross-Modal Synchronization)**: Reports whether the speech track matches the lip articulation (`Synchronized` vs. `Desynchronization / Dubbing Detected`).
3. **Forensic Explanation Example**: If an authentic video is dubbed with English translation, the system outputs:
   - Verdict: `REAL (Authentic Visuals)`
   - Consistency Status: `Desynchronization / Dubbing Detected (S_AV = -0.05)`
   - Recommendation: "Visual content appears authentic, but cross-modal desynchronization was detected. Treat as potential translation voice-over rather than visual forgery."
