# Demo Walkthrough & Presentation Guide

**Project:** Agentic Multimodal Deepfake Detection using Audio-Visual Forensic Analysis  
**Goal:** Step-by-step instructions for demonstrating the live web application and forensic agent.

---

## 1. Quick Start Commands

### A. Start the Backend Server:
```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### B. Access the Web Interfaces:
- **User Analysis Interface:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Agentic Forensic Analysis Pipeline:** [http://127.0.0.1:8000/agent](http://127.0.0.1:8000/agent)
- **System Architecture & Methodology:** [http://127.0.0.1:8000/architecture](http://127.0.0.1:8000/architecture)
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 2. Live Demonstration Scenarios

### Scenario 1: Live Video Upload & Multimodal Analysis
1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
2. Drag and drop any test video (e.g. from `dataset/FakeAVCeleb/.../*.mp4`) into the dropzone.
3. Click **"Analyze Forensic Evidence"**.
4. Observe the results cards:
   - **Verdict Badge**: REAL / FAKE with animated glow.
   - **Adaptive Fusion Meter**: Dual-bar visualizing visual vs. acoustic attention percentage.
   - **Audio Status**: Verifies active 16kHz mono acoustic channel across Whisper, Wav2Vec2, and Spectrogram CNN.
   - **Cross-Modal Consistency ($S_{AV}$)**: Quantifies embedding alignment against threshold $0.1177$.
   - **Forensic Agent Explanation**: Grounded reasoning narrative and actionable operational recommendation.
5. Click **"Download Report (.md)"** to save the generated markdown forensic audit.

---

### Scenario 2: Instant Pre-Indexed Benchmark Evaluation (1-Click Demo)
1. On the right side of the User Analysis page, select any of the pre-indexed demo cases:
   - **Case 1: Authentic Media** $\to$ Demonstrates verified real video with matched audio ($S_{AV} = 0.3842$, Low Risk).
   - **Case 2: Full Multimodal Deepfake** $\to$ Demonstrates faceswap + voice clone ($99.4\%$ confidence, Critical Risk).
   - **Case 3: Missing Audio Video** $\to$ Demonstrates automatic **Vision-Only Fallback** with 100% visual attention allocation without crashing.
   - **Case 4: Dubbed / Desynchronized Audio** $\to$ Demonstrates cross-modal mismatch ($S_{AV} = -0.0385 < 0.1177$) where visual authenticity is preserved while audio desynchronization is flagged.
2. Switch to the **"Agentic Forensic Analysis"** tab to view the live 6-stage pipeline flow:
   `VIDEO` $\to$ `VISION ANALYSIS` $\to$ `AUDIO ANALYSIS` $\to$ `CROSS-MODAL CONSISTENCY` $\to$ `ADAPTIVE TRUST` $\to$ `FORENSIC REASONING` $\to$ `FINAL VERDICT`.

---

### Scenario 3: System Architecture Presentation
1. Click the **"System Architecture"** tab.
2. Walk examiners through the complete 6-encoder topology:
   - Spatial texture (EfficientNet-B0, 1280-D)
   - Temporal dynamics (Video Swin-T, 768-D)
   - Perioral articulation (Face/Mouth ResNet-18, 512-D)
   - Semantic speech (Whisper, 512-D)
   - Acoustic speech (Wav2Vec2, 768-D)
   - Time-frequency spectrograms (Spectrogram CNN, 256-D)
   - Contrastive alignment and adaptive fusion attention mechanism.
