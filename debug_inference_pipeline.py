"""
debug_inference_pipeline.py

Comprehensive diagnostic tool to trace the entire Deepfake Inference Pipeline
from raw video decoding to classifier logits, probabilities, and final verdict.
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.config import (
    CHECKPOINT_PATH,
    DEVICE,
    CLASSIFIER_THRESHOLD,
    AV_CONSISTENCY_THRESHOLD,
)
from app.video_processor import VideoProcessor
from app.inference_service import InferenceService
from models.multimodal_model import MultimodalDeepfakeDetector


def debug_video(video_path_str: str, ground_truth: str = "UNKNOWN"):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"ERROR: Video file not found at {video_path}")
        return

    print("=" * 60)
    print("DEEPFAKE INFERENCE DIAGNOSTIC")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Ground Truth: {ground_truth}")

    # 1. Video Preprocessing
    processor = VideoProcessor()
    temp_dir = Path("app/temp_uploads/debug_run")
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. PREPROCESSING & FEATURE EXTRACTION ---")
    try:
        images_tensor, waveform_tensor, audio_available, metadata = processor.process_video_and_audio(
            video_path=video_path,
            temp_dir=temp_dir,
        )
    except Exception as e:
        print(f"ERROR in preprocessing: {e}")
        return

    print(f"Audio: {'AVAILABLE' if audio_available else 'MISSING'}")
    print(f"Resolution: {metadata.get('resolution')}")
    print(f"FPS: {metadata.get('fps')}")
    print(f"Total Video Frames: {metadata.get('total_frames')}")
    print(f"Images Tensor Shape: {tuple(images_tensor.shape)} (dtype: {images_tensor.dtype})")
    print(f"Images Range: min={images_tensor.min().item():.3f}, max={images_tensor.max().item():.3f}, mean={images_tensor.mean().item():.3f}")
    print(f"Waveform Tensor Shape: {tuple(waveform_tensor.shape)} (dtype: {waveform_tensor.dtype})")
    print(f"Waveform Range: min={waveform_tensor.min().item():.4f}, max={waveform_tensor.max().item():.4f}")

    # 2. Checkpoint & Model Loading
    print("\n--- 2. MODEL & CHECKPOINT STATE ---")
    service = InferenceService.get_instance()
    service.load_model()
    model = service.model
    device = service.device

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint Epoch: {service.checkpoint_epoch}")
    print(f"Model Mode: {'EVAL' if not model.training else 'TRAIN'}")
    print(f"Device: {device}")

    # 3. Model Forward Pass
    print("\n--- 3. DETECTOR FORWARD PASS ---")
    images_dev = images_tensor.to(device)
    waveform_dev = waveform_tensor.to(device) if audio_available else None

    with torch.inference_mode():
        outputs = model(
            images=images_dev,
            waveform=waveform_dev,
            audio_available=audio_available,
            return_loss=False,
            return_intermediate_features=True,
        )

    eff_f = outputs.get("efficientnet_features")
    swin_f = outputs.get("video_swin_features")
    fm_f = outputs.get("face_mouth_features")
    fused_v = outputs.get("fused_vision_features")
    v_proj = outputs.get("vision_embedding")

    w_f = outputs.get("whisper_features")
    w2v_f = outputs.get("wav2vec2_features")
    spec_f = outputs.get("spectrogram_features")
    fused_a = outputs.get("fused_audio_features")
    a_proj = outputs.get("audio_embedding")

    fused_emb = outputs.get("fused_embedding")
    attn = outputs.get("attention")
    logits = outputs.get("logits")

    print(f"Vision Features: EfficientNet={tuple(eff_f.shape) if eff_f is not None else None}, Swin={tuple(swin_f.shape) if swin_f is not None else None}, FaceMouth={tuple(fm_f.shape) if fm_f is not None else None}")
    print(f"Vision Aggregated: {tuple(fused_v.shape) if fused_v is not None else None}, Projected: {tuple(v_proj.shape) if v_proj is not None else None}")
    print(f"Audio Features: Whisper={tuple(w_f.shape) if w_f is not None else None}, Wav2Vec2={tuple(w2v_f.shape) if w2v_f is not None else None}, SpecCNN={tuple(spec_f.shape) if spec_f is not None else None}")
    print(f"Audio Aggregated: {tuple(fused_a.shape) if fused_a is not None else None}, Projected: {tuple(a_proj.shape) if a_proj is not None else None}")
    print(f"Fusion Features: {tuple(fused_emb.shape)}")
    print(f"Attention Weights: Vision={attn[0, 0].item():.4f}, Audio={attn[0, 1].item():.4f}")

    # Audio-Visual Consistency
    av_consistency = None
    if audio_available and a_proj is not None:
        v_norm = F.normalize(v_proj, p=2, dim=1)
        a_norm = F.normalize(a_proj, p=2, dim=1)
        s_av = (v_norm * a_norm).sum(dim=1).item()
        av_consistency = float(s_av)
    print(f"Audio-Visual Consistency (S_AV): {av_consistency:.4f}" if av_consistency is not None else "Audio-Visual Consistency: N/A")

    # 4. Classifier Head & Probabilities
    print("\n--- 4. CLASSIFICATION & DECISION LOGIC ---")
    raw_logits = logits.squeeze(0).cpu().numpy().tolist()
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    p_real = float(probs[0])
    p_fake = float(probs[1])

    print(f"Raw Logits: [Index 0 (Real): {raw_logits[0]:.4f}, Index 1 (Fake): {raw_logits[1]:.4f}]")
    print(f"P(REAL): {p_real:.4f} ({p_real*100:.2f}%)")
    print(f"P(FAKE): {p_fake:.4f} ({p_fake*100:.2f}%)")
    print(f"Decision Threshold: {CLASSIFIER_THRESHOLD:.2f}")

    # Threshold checks
    print("\nDecision at multiple thresholds:")
    for th in [0.30, 0.40, 0.50, 0.60, 0.70]:
        pred_th = "FAKE" if p_fake >= th else "REAL"
        print(f"  Threshold {th:.2f} -> {pred_th} (p_fake={p_fake:.4f} {'>=' if p_fake >= th else '<'} {th:.2f})")

    # Final service prediction
    final_analysis = service.run_inference(
        images_tensor=images_tensor,
        waveform_tensor=waveform_tensor,
        audio_available=audio_available,
        video_filename=video_path.name,
        video_metadata=metadata,
    )

    print("\n--- 5. FINAL INFERENCE SERVICE & AGENT OUTPUT ---")
    print(f"Prediction: {final_analysis['prediction']}")
    print(f"Confidence: {final_analysis['confidence']*100:.2f}%")
    print(f"Forensic Reliability: {final_analysis['agent_analysis'].get('forensic_reliability')}")
    print(f"OOD Warning: {final_analysis['agent_analysis'].get('domain_reliability', {}).get('is_ood')}")
    print(f"Agent Summary: {final_analysis['agent_analysis']['llm_explanation']['summary']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Deepfake Inference Pipeline")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--truth", type=str, default="UNKNOWN", help="Ground truth label")
    args = parser.parse_args()

    debug_video(args.video_path, args.truth)
