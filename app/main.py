"""
app/main.py

FastAPI Application Entry Point for Multimodal Deepfake Forensic Analysis.
Provides REST API endpoints and serves the Web Application interfaces.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    BASE_DIR,
    STATIC_DIR,
    TEMP_UPLOAD_DIR,
    CHECKPOINT_PATH,
    DEVICE,
    APP_NAME,
    APP_VERSION,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS,
)
from app.video_processor import VideoProcessor
from app.inference_service import InferenceService
from app.demo_service import DemoService


# -------------------------------------------------------------
# Lifespan Event Management (Single Model Loading on Startup)
# -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the 6-encoder multimodal detector into memory once
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    inference_service = InferenceService.get_instance()
    try:
        inference_service.load_model()
    except Exception as e:
        print(f"[!] Warning during model load at startup: {e}")

    yield

    # Shutdown: Clean temporary upload artifacts
    if TEMP_UPLOAD_DIR.exists():
        shutil.rmtree(TEMP_UPLOAD_DIR, ignore_errors=True)


# -------------------------------------------------------------
# Initialize FastAPI Application
# -------------------------------------------------------------
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Agentic Multimodal Deepfake Detection using Audio-Visual Forensic Analysis",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_processor = VideoProcessor()
inference_service = InferenceService.get_instance()
demo_service = DemoService()


# -------------------------------------------------------------
# Static Files & UI Routes
# -------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Multimodal Deepfake Detection Web App</h1><p>UI loading...</p>")


@app.get("/agent", response_class=HTMLResponse)
async def serve_agent_view():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Agentic Forensic Analysis</h1>")


@app.get("/architecture", response_class=HTMLResponse)
async def serve_architecture_view():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>System Architecture & Methodology</h1>")


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """
    Return backend status, CUDA availability, model load status, and checkpoint information.
    """
    return {
        "status": "healthy",
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "device": str(inference_service.device),
        "cuda_available": bool(inference_service.device.type == "cuda"),
        "model_loaded": inference_service.is_loaded,
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_epoch": inference_service.checkpoint_epoch,
        "checkpoint_val_f1": inference_service.checkpoint_val_f1,
        "encoders": {
            "vision": ["EfficientNet-B0 (1280-D)", "VideoSwin-Tiny (768-D)", "FaceMouth-ResNet18 (512-D)"],
            "audio": ["Whisper (512-D)", "Wav2Vec2 (768-D)", "Spectrogram-CNN (256-D)"],
        },
        "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
        "allowed_extensions": sorted(list(ALLOWED_EXTENSIONS)),
    }


@app.get("/api/demo/samples")
async def get_demo_samples() -> Dict[str, Any]:
    """
    Get catalog of pre-indexed benchmark demo cases.
    """
    samples = demo_service.get_catalog()
    return {"samples": samples}


@app.post("/api/demo/analyze/{sample_id}")
async def analyze_demo_sample(sample_id: str) -> Dict[str, Any]:
    """
    Execute instant forensic analysis on a pre-indexed demo case.
    """
    result = demo_service.get_demo_analysis(sample_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Demo sample '{sample_id}' not found.")
    return result


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Ingest uploaded video file, perform face and audio extraction,
    execute 6-encoder multimodal deepfake inference, compute cross-modal consistency,
    and generate evidence-grounded agent reasoning.
    """
    # 1. Validate filename and extension
    if not video.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    filename = video.filename
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Save to temporary directory
    session_id = str(uuid.uuid4())[:8]
    temp_dir = TEMP_UPLOAD_DIR / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_video_path = temp_dir / f"upload_{filename}"

    try:
        content = await video.read()
        file_size = len(content)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes).")
        if file_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {MAX_UPLOAD_SIZE_MB}MB limit.",
            )

        with open(temp_video_path, "wb") as f:
            f.write(content)

        # 3. Process video frames and audio
        try:
            images_tensor, waveform_tensor, audio_available, video_metadata = video_processor.process_video_and_audio(
                video_path=temp_video_path,
                temp_dir=temp_dir,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Video decoding and preprocessing failed: {str(e)}",
            )

        # 4. Run multimodal inference & agent reasoning
        try:
            analysis_result = inference_service.run_inference(
                images_tensor=images_tensor,
                waveform_tensor=waveform_tensor,
                audio_available=audio_available,
                video_filename=filename,
                video_metadata=video_metadata,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Inference execution failed: {str(e)}",
            )

        analysis_result["is_demo_sample"] = False
        return analysis_result

    finally:
        # 5. Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
