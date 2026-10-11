# Autonomous Agentic Forensic Reasoning System

## 1. Definition & Justification of "Agentic" Architecture
The system is designated **"Agentic"** because it does not merely output a black-box probability. Instead, it operates as an **Autonomous Forensic Agent** (`agent/forensic_agent.py`) that executes goal-directed reasoning over structured multimodal evidence:

1. **Evidence Perception**: Ingests structured evidence dictionaries from 6 foundation encoders, modality attention weights ($lpha_v, lpha_a$), cross-modal similarity ($S_{AV}$), and 12 video container metadata fields.
2. **Autonomous Multi-Criteria Evaluation**: Evaluates domain reliability, cross-modal synchronization, and missing-modality state independently.
3. **Deterministic Reasoning Synthesis**: Generates grounded, hallucination-free forensic explanations using a verified deterministic rule engine (`agent/reasoning.py`).
4. **Autonomous Adaptive Actions**:
   - Triggers dynamic modality fallback during missing audio.
   - Disentangles language dubbing from visual deepfakes.
   - Activates OOD warning overlays when screen capture compression is detected.
5. **Actionable Intelligence Generation**: Produces publication-grade Markdown and JSON forensic reports (`agent/report_generator.py`).

---

## 2. Agent Data Structures & Workflow

```
+----------------------------------------------------------------------------------------------------+
|                                    RAW MULTIMODAL INFERENCE OUTPUT                                 |
|  { logits, vision_emb, audio_emb, alpha_v, alpha_a, S_AV, audio_available, video_metadata }        |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                               +------------------------------------+
                               |     ForensicAgent.process_evidence |
                               |     Source: agent/forensic_agent.py|
                               +------------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                            |                            |
                     v                            v                            v
       +--------------------------+ +--------------------------+ +--------------------------+
       | evaluate_visual_evidence | | evaluate_audio_evidence  | | evaluate_cross_modal...  |
       | EfficientNet, Swin, Mouth| | Whisper, W2V2, SpecCNN   | | S_AV vs Threshold 0.1177 |
       +--------------------------+ +--------------------------+ +--------------------------+
                     |                            |                            |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
                               +------------------------------------+
                               | evaluate_domain_reliability        |
                               | 12-Criterion OOD Scoring (0.0..1.0)|
                               +------------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
                     v                                                         v
       +--------------------------+                              +--------------------------+
       | determine_risk_level     |                              | generate_recommendation  |
       | CRITICAL, HIGH, MEDIUM,  |                              | Actionable operational   |
       | MODERATE-UNCERTAIN, LOW  |                              | guidance for examiners   |
       +--------------------------+                              +--------------------------+
                                                  |
                                                  v
                               +------------------------------------+
                               | synthesize_reasoning               |
                               | Grounded Multi-Paragraph Narrative |
                               +------------------------------------+
                                                  |
                                                  v
                               +------------------------------------+
                               | ForensicReportGenerator            |
                               | Structured Markdown & Audit Log    |
                               +------------------------------------+
```

---

## 3. Reliability & Risk Level Matrix

```
========================================================================================================================
PREDICTION    CONFIDENCE    AUDIO STATE     AV CONSISTENCY    OOD STATUS    RELIABILITY STATUS     RISK LEVEL
========================================================================================================================
FAKE          >= 85.0%      Available       Any               In-Dist       HIGH RELIABILITY       CRITICAL
FAKE          70.0 - 84.9%  Available       Any               In-Dist       HIGH RELIABILITY       HIGH
FAKE          < 70.0%       Available       Any               In-Dist       MEDIUM RELIABILITY     MEDIUM-HIGH
FAKE          Any           Missing         N/A (Bypassed)    In-Dist       MEDIUM RELIABILITY     HIGH
REAL          >= 80.0%      Available       Synchronized      In-Dist       HIGH RELIABILITY       LOW
REAL          < 80.0%       Available       Synchronized      In-Dist       MEDIUM RELIABILITY     MODERATE
REAL          Any           Available       Mismatched/Dubbed In-Dist       MEDIUM RELIABILITY     MEDIUM
REAL          Any           Missing         N/A (Bypassed)    In-Dist       MEDIUM RELIABILITY     LOW-MEDIUM
REAL          Any           Any             Any               OOD Shift     LOW RELIABILITY        MODERATE-UNCERTAIN
========================================================================================================================
```

---

## 4. Implemented Agent Decision Rules

### Visual Evidence Rules (`agent/reasoning.py:23-60`):
- If `prediction == "FAKE"`: Generates grounded observations on spatial blending boundaries (EfficientNet), inter-frame motion discontinuity (Video Swin), and perioral warping (Face/Mouth ResNet).
- If `vision_weight >= 0.70`: Highlights that adaptive fusion allocated primary discriminant attention to the visual stream.

### Audio Evidence Rules (`agent/reasoning.py:63-106`):
- If `not audio_available`: Documents that audio was absent/silent, audio encoders were safely bypassed, and notes that missing audio represents absence of evidence rather than manipulation.
- If `prediction == "FAKE"`: Documents synthetic vocoder spectral markers (Wav2Vec2 + Spectrogram CNN) and phonetic anomalies (Whisper).

### Cross-Modal Consistency Rules (`agent/reasoning.py:109-147`):
- Evaluates $S_{AV} = \mathbf{e}_v^T \mathbf{e}_a$. If $S_{AV} < 0.1177$, flags desynchronization / dubbing as a separate forensic observation.

### Domain Reliability Rules (`agent/reasoning.py:150-282`):
- Computes composite reliability score $R \in [0.10, 1.00]$. If $R < 0.50$ or screen recording is detected, flags `LOW RELIABILITY — POSSIBLE OUT-OF-DISTRIBUTION INPUT`.
