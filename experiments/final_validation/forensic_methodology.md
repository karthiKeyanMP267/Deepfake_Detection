# Forensic Audio-Visual Analysis Methodology

## 1. Multi-Condition Forensic Framework
The system operates beyond binary Real/Fake classification by performing dual-stream forensic inspection:

```
                      VIDEO INPUT
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    Visual Deepfake             Cross-Modal Alignment
       Detection                     Forensics
  (Texture + Motion + Lip)    (Audio-Visual Synchronization)
             │                           │
             ▼                           ▼
  [Deepfake Prob: 0.987]         [AV Consistency: 0.2618]
```

---

## 2. Missing & Degraded Audio Fallback Protocol
When `audio_available = False` (silent audio, unreadable track, corrupted codec, or missing file):
1. **Encoder Bypass**: `Whisper`, `Wav2Vec2`, and `Spectrogram CNN` are completely skipped, saving computational resources.
2. **Deterministic Weight Routing**: The adaptive fusion attention network assigns vision_weight = 1.000 and audio_weight = 0.000.
3. **Vision-Only Prediction**: The classifier predicts purely on the aggregated visual representation without numerical divergence.

---

## 3. Dubbed & Mismatched Audio Forensics
- **Cosine Similarity Alignment**: S_AV = dot(e_v, e_a) / (||e_v|| * ||e_a||) in [-1, 1]
- **Separation**: Normal synchronized speech produces positive alignment (mean = +0.2618), whereas mismatched/dubbed speech drops to baseline (mean = -0.0060, a +0.2678 separation).
- **CRITICAL SCIENTIFIC PRINCIPLE**: Low audio-visual consistency indicates cross-modal mismatch (e.g. translated dubbing, audio replacement), which is reported as **forensic inconsistency evidence** rather than falsely classifying authentic translated videos as deepfakes.
