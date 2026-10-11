# Forensic Agent Architecture Documentation

**System Title:** Agentic Forensic Intelligence & Evidence Grounding Layer  
**Role:** Autonomous Forensic Reasoner for 6-Encoder Multimodal Deepfake Detector  

---

## 1. Design Philosophy: Grounded Intelligence without Hallucinations

Deepfake forensic systems must produce verifiable, explainable determinations that can withstand legal, academic, and industrial scrutiny. The Forensic Agent serves as an evidence-grounded interpretation layer between deep neural network representations and human forensic analysts.

### Core Principles:
1. **Zero Hallucination Guarantee**: The agent only asserts forensic anomalies that are quantitatively grounded in the outputs of the 6 encoders, the adaptive fusion attention weights, or the cross-modal consistency metric.
2. **Deterministic Multi-Stage Reasoning**: Rules and heuristics map raw probabilistic outputs and embedding alignments to structured forensic narratives.
3. **Plug-and-Play LLM Extensibility**: The agent engine is built with clean interfaces (`agent/reasoning.py`, `agent/forensic_agent.py`), allowing local deterministic engines to operate standalone while supporting seamless integration with external or local LLMs (e.g. LLaMA, Gemini, OpenAI).

---

## 2. Structured Evidence Data Contract

Before generating any report, the detector constructs a standardized internal evidence object:

```json
{
  "final_prediction": "FAKE",
  "confidence": 0.9876,
  "probability_real": 0.0124,
  "probability_fake": 0.9876,
  "audio_available": true,
  "fusion_weights": {
    "vision": 0.6751,
    "audio": 0.3249
  },
  "av_consistency": 0.5233,
  "missing_audio": false
}
```

---

## 3. Autonomous Reasoning Engine Workflow

```
                   INTERNAL EVIDENCE CONTRACT
                               │
               ┌───────────────┴───────────────┐
               │                               │
        VISUAL REASONING                ACOUSTIC REASONING
    • Spatial texture anomalies        • Vocoder phase anomalies
    • Temporal motion continuity       • Vocal timbre & pitch
    • Perioral articulation            • Phonetic speech cadence
               │                               │
               └───────────────┬───────────────┘
                               │
                  CROSS-MODAL CONSISTENCY
             • Cosine similarity evaluation
             • S_AV threshold diagnostic (0.1177)
             • Synchronization vs. Dubbing detection
                               │
                    ADAPTIVE TRUST AUDIT
             • Attention weight allocation analysis
             • Modality reliability determination
                               │
                   RISK LEVEL & RECOMMENDATION
             • Tier classification (CRITICAL, HIGH, MEDIUM, LOW)
             • Actionable quarantine/approval guidance
                               │
                   STRUCTURED FORENSIC REPORT
```

---

## 4. Reasoning Taxonomy & Heuristics

### A. Visual Stream Heuristics:
- **Spatial Texture Evaluation**: If $\hat{y} = \text{FAKE}$, highlights high-frequency blending and boundary warping identified by EfficientNet-B0. If $\hat{y} = \text{REAL}$, confirms natural biological texture continuity.
- **Temporal Motion Evaluation**: Explains whether Video Swin-Tiny detected inter-frame jitter/warping or coherent continuous motion.
- **Articulation Evaluation**: Explains whether Face/Mouth ResNet-18 detected unnatural perioral boundary manipulation.

### B. Acoustic Stream Heuristics:
- **Audio Availability**: When `audio_available = False`, explicitly clarifies: *"Missing audio represents the absence of acoustic evidence, NOT proof of manipulation."*
- **Speech & Vocoder Markers**: Explains synthetic vocoder spectral noise (Wav2Vec2 + Spectrogram CNN) versus natural environmental reverberation.

### C. Cross-Modal Consistency & Dubbing:
- **Threshold Diagnostic**: Evaluates $S_{AV}$ against the Phase-6 threshold ($0.1177$).
- **Scientific Separation**: Separates **Deepfake Prediction** (synthetic video/audio) from **Cross-Modal Inconsistency** (authentic video with translated or dubbed audio).

### D. Risk Tier Mapping:
- **CRITICAL**: Fake with confidence $\ge 85\%$.
- **HIGH**: Fake with confidence $\ge 70\%$.
- **MEDIUM**: Real video with low consistency $S_{AV} < 0.1177$ (Dubbing / Voiceover).
- **LOW-MEDIUM**: Real video with missing audio (Vision-only verified).
- **LOW**: Real video with high confidence ($\ge 80\%$) and verified multimodal synchronization.

---

## 5. Extensibility: Plugging in an LLM

The agent layer can be upgraded to an LLM reasoner by passing the structured evidence contract into a prompt template:

```python
class LLMForensicAgent(ForensicAgent):
    def __init__(self, llm_client):
        super().__init__()
        self.llm = llm_client

    def generate_llm_narrative(self, evidence_contract: dict) -> str:
        prompt = f"""
        You are an expert digital forensics examiner.
        Analyze the following verified multimodal detector output:
        {json.dumps(evidence_contract, indent=2)}
        
        Generate an evidence-grounded forensic report.
        Do NOT invent anomalies not present in the quantitative output.
        """
        return self.llm.generate(prompt)
```
