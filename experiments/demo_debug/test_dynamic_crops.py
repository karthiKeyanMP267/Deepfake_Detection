"""
experiments/demo_debug/test_dynamic_crops.py

Compare:
1. Static BBox Cropping (Current baseline)
2. Dynamic Per-Frame Face Tracking
Across:
- Demo Video 1
- Demo Video 2
- 5 In-distribution Real videos
- 5 In-distribution Fake videos
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from app.video_processor import VideoProcessor
from app.inference_service import InferenceService

service = InferenceService.get_instance()
service.load_model()
model = service.model

vp = VideoProcessor()
det = vp._get_face_detector()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_dynamic_face_frames(video_path):
    raw_frames = vp.extract_frames(video_path, num_frames=16)
    last_valid_bbox = None
    processed_tensors = []
    crops = []

    for i, frame in enumerate(raw_frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = det.get(bgr)
        if len(faces) > 0:
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = map(int, largest.bbox)
            bw, bh = x2 - x1, y2 - y1
            padx, pady = int(bw * 0.15), int(bh * 0.15)
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padx)
            y1 = max(0, y1 - pady)
            x2 = min(w, x2 + padx)
            y2 = min(h, y2 + pady)
            last_valid_bbox = (x1, y1, x2, y2)
            cropped = frame[y1:y2, x1:x2]
        elif last_valid_bbox is not None:
            x1, y1, x2, y2 = last_valid_bbox
            cropped = frame[y1:y2, x1:x2]
        else:
            h, w = frame.shape[:2]
            s = min(h, w)
            cropped = frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]

        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            h, w = frame.shape[:2]
            s = min(h, w)
            cropped = frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]

        cropped = cv2.resize(cropped, (224, 224))
        crops.append(cropped)
        pil_img = Image.fromarray(cropped)
        processed_tensors.append(transform(pil_img))

    return torch.stack(processed_tensors, dim=0).unsqueeze(0), crops


def main():
    print("=" * 70)
    print("TESTING DYNAMIC PER-FRAME FACE EXTRACTION")
    print("=" * 70)

    for label, fname in [("Video 1 (222131)", "video1_222131.mp4"), ("Video 2 (222513)", "video2_222513.mp4")]:
        vpath = Path("experiments/demo_debug/videos") / fname
        img_dyn, _ = extract_dynamic_face_frames(vpath)
        
        temp_dir = Path("experiments/demo_debug") / f"temp_dyn_{vpath.stem}"
        temp_dir.mkdir(exist_ok=True)
        try:
            _, waveform, audio_avail = vp.process_video_and_audio(vpath, temp_dir)

            with torch.inference_mode():
                out = model(
                    images=img_dyn.to(service.device),
                    waveform=waveform.to(service.device) if audio_avail else None,
                    audio_available=audio_avail,
                    return_loss=False,
                    return_intermediate_features=True,
                )
            probs = F.softmax(out["logits"], dim=1).squeeze(0).cpu().numpy()
            
            # Isolated vision
            with torch.inference_mode():
                out_vo = model(
                    images=img_dyn.to(service.device),
                    waveform=None,
                    audio_available=False,
                    return_loss=False,
                )
            probs_vo = F.softmax(out_vo["logits"], dim=1).squeeze(0).cpu().numpy()

            print(f"\n--- {label} ---")
            print(f"  Multimodal Logits : {out['logits'].cpu().numpy().tolist()}")
            print(f"  Multimodal P(Real): {probs[0]:.4f}, P(Fake): {probs[1]:.4f} -> Prediction: {'FAKE' if probs[1]>=0.5 else 'REAL'}")
            print(f"  Vision-Only P(Real): {probs_vo[0]:.4f}, P(Fake): {probs_vo[1]:.4f} -> Prediction: {'FAKE' if probs_vo[1]>=0.5 else 'REAL'}")
            print(f"  Vision Weight: {out['vision_weight'].item():.4f}, Audio Weight: {out['audio_weight'].item():.4f}")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
