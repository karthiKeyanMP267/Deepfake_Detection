# REST API Documentation

**Application Name:** Agentic Multimodal Deepfake Forensic Analyzer  
**Base URL:** `http://localhost:8000`  
**OpenAPI Interactive Docs:** `http://localhost:8000/docs`  

---

## 1. System Health Endpoint

### `GET /api/health`

Returns server status, hardware device, CUDA availability, model load status, checkpoint metadata, and active encoder architectures.

#### Response:
```json
{
  "status": "healthy",
  "app_name": "Agentic Multimodal Deepfake Forensic Analyzer",
  "version": "1.0.0",
  "device": "cuda",
  "cuda_available": true,
  "model_loaded": true,
  "checkpoint_path": "D:\\Deepfake\\Agentic-Multimodal-Deepfake-Detection\\experiments\\05_full_3v3\\models\\best_model.pth",
  "checkpoint_epoch": 4,
  "checkpoint_val_f1": 0.9722,
  "encoders": {
    "vision": [
      "EfficientNet-B0 (1280-D)",
      "VideoSwin-Tiny (768-D)",
      "FaceMouth-ResNet18 (512-D)"
    ],
    "audio": [
      "Whisper (512-D)",
      "Wav2Vec2 (768-D)",
      "Spectrogram-CNN (256-D)"
    ]
  },
  "max_upload_size_mb": 100,
  "allowed_extensions": [".avi", ".mkv", ".mov", ".mp4", ".webm"]
}
```

---

## 2. Video Analysis Endpoint

### `POST /api/analyze`

Accepts a multipart video file upload, executes 6-encoder multimodal feature extraction, computes adaptive attention fusion and cross-modal consistency, and returns grounded agent reasoning.

#### Request:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `video`: Binary file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`), max 100 MB.

#### Response (HTTP 200):
```json
{
  "prediction": "FAKE",
  "confidence": 0.9876,
  "confidence_percentage": 98.76,
  "probability_real": 0.0124,
  "probability_fake": 0.9876,
  "audio_available": true,
  "fusion_weights": {
    "vision": 0.6751,
    "audio": 0.3249
  },
  "av_consistency": 0.5233,
  "vision_evidence": [
    "Visual branch detected spatio-temporal artifacts across facial regions (EfficientNet-B0 texture analysis).",
    "Temporal inter-frame dynamics exhibit motion inconsistency indicative of synthetic synthesis (Video Swin Transformer).",
    "Perioral and facial articulation patterns indicate unnatural boundary or warping anomalies (Face/Mouth ResNet-18)."
  ],
  "audio_evidence": [
    "Acoustic feature representations exhibit spectral characteristics typical of synthetic vocoders (Wav2Vec2 + Spectrogram CNN).",
    "Phonetic distribution analysis indicates potential voice cloning or artificial synthesis markers (Whisper encoder)."
  ],
  "forensic_evidence": [
    "Visual branch detected spatio-temporal artifacts across facial regions (EfficientNet-B0 texture analysis).",
    "Temporal inter-frame dynamics exhibit motion inconsistency indicative of synthetic synthesis (Video Swin Transformer).",
    "Perioral and facial articulation patterns indicate unnatural boundary or warping anomalies (Face/Mouth ResNet-18).",
    "Acoustic feature representations exhibit spectral characteristics typical of synthetic vocoders (Wav2Vec2 + Spectrogram CNN).",
    "Phonetic distribution analysis indicates potential voice cloning or artificial synthesis markers (Whisper encoder)."
  ],
  "agent_analysis": {
    "verdict": "FAKE",
    "confidence_percentage": 98.76,
    "probability_real": 0.0124,
    "probability_fake": 0.9876,
    "reasoning": "The multimodal detection system classified this media as FAKE with 98.8% confidence (deepfake probability: 0.9876). Adaptive fusion allocated 67.5% weight to the visual pathway and 32.5% to the acoustic pathway. Evidence from spatial textures (EfficientNet), temporal video motion (Video Swin), and facial articulations (Face/Mouth ResNet) combined with speech spectral features to establish synthetic manipulation.",
    "risk_level": "CRITICAL",
    "recommendation": "High-confidence deepfake detection. Quarantine media and flag as synthetic content across ingestion pipelines.",
    "visual_evidence": [...],
    "audio_evidence": [...],
    "cross_modal_analysis": {
      "score": 0.5233,
      "status": "Synchronized / Coherent",
      "is_mismatched": false,
      "threshold": 0.1177,
      "detail": "Audio-visual embedding consistency score (0.5233) meets the synchronization threshold (>= 0.1177), indicating alignment between visual speech articulation and acoustic representations."
    },
    "fusion_breakdown": {
      "vision_weight": 0.6751,
      "audio_weight": 0.3249,
      "mode": "Adaptive Multimodal (3V3)",
      "explanation": "Modality attention network assigned 67.5% to visual stream and 32.5% to acoustic stream."
    }
  },
  "markdown_report": "# Forensic Deepfake Analysis Report...",
  "inference_time_ms": 1916.2,
  "device": "cuda",
  "is_demo_sample": false
}
```

#### Error Responses:
- **HTTP 400 Bad Request**: Invalid file extension, empty file, corrupted video decoding failure.
- **HTTP 500 Internal Server Error**: Inference runtime exception.

---

## 3. Demo Benchmark Endpoints

### `GET /api/demo/samples`
Returns catalog of pre-indexed benchmark demo cases.

### `POST /api/demo/analyze/{sample_id}`
Executes instant forensic analysis on the selected benchmark case without file upload overhead.
