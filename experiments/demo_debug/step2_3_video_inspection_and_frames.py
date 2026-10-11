"""
experiments/demo_debug/step2_3_video_inspection_and_frames.py

Step 2: Video Forensic Inspection & Metadata Extraction
Step 3: Visual Frame & Face-Crop Contact Sheet Generation
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import json
import subprocess
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchaudio

from app.video_processor import VideoProcessor

DEBUG_DIR = BASE_DIR / "experiments" / "demo_debug"
VIDEOS_DIR = DEBUG_DIR / "videos"
FRAMES_DIR = DEBUG_DIR / "frames"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def get_video_properties(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = (frame_count / fps) if fps > 0 else 0.0
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()

    # Audio inspection via torchaudio / imageio-ffmpeg
    audio_info = {}
    temp_wav = DEBUG_DIR / f"temp_probe_{video_path.stem}.wav"
    
    ffmpeg_cmd = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    cmd = [
        ffmpeg_cmd, "-i", str(video_path),
        "-vn", "-y", str(temp_wav)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        si, sr = torchaudio.load(str(temp_wav))
        audio_info["audio_present"] = True
        audio_info["channels"] = si.shape[0]
        audio_info["sample_rate"] = sr
        audio_info["samples"] = si.shape[1]
        audio_info["duration_sec"] = round(si.shape[1] / sr, 2)
        audio_info["max_amplitude"] = float(torch.max(torch.abs(si)).item())
    except Exception as e:
        audio_info["audio_present"] = False
        audio_info["error"] = str(e)
    finally:
        if temp_wav.exists():
            temp_wav.unlink()

    return {
        "filename": video_path.name,
        "file_size_bytes": video_path.stat().st_size,
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "aspect_ratio": f"{width/height:.2f}" if height > 0 else "N/A",
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": round(duration, 2),
        "fourcc": fourcc_str,
        "audio": audio_info,
    }


def generate_contact_sheet(frames, title, save_path):
    """
    Save 4x4 grid of 16 frames into a high-res PNG contact sheet.
    """
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    for i, ax in enumerate(axes.flat):
        if i < len(frames):
            frame = frames[i]
            if isinstance(frame, torch.Tensor):
                # Unnormalize tensor (C, H, W)
                img = frame.clone().detach().cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
                ax.imshow(img)
            else:
                ax.imshow(frame)
            ax.set_title(f"F#{i+1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved contact sheet -> {save_path}")


def inspect_and_visualize():
    video_processor = VideoProcessor()
    
    videos = [
        ("video1", VIDEOS_DIR / "video1_222131.mp4"),
        ("video2", VIDEOS_DIR / "video2_222513.mp4"),
    ]

    all_props = {}

    for vid_key, vid_path in videos:
        print("=" * 60)
        print(f"INSPECTING: {vid_path.name}")
        print("=" * 60)
        props = get_video_properties(vid_path)
        all_props[vid_key] = props
        print(json.dumps(props, indent=2))

        # Extract raw frames
        raw_frames = video_processor.extract_frames(vid_path, num_frames=16)
        generate_contact_sheet(
            raw_frames,
            f"{vid_path.name} - 16 Uniform Raw Frames",
            FRAMES_DIR / f"{vid_key}_frames.png"
        )

        # Detect face & extract face crops
        bbox = video_processor.detect_video_face_bbox(raw_frames)
        print(f"Detected Face Bounding Box ({vid_key}): {bbox}")
        
        face_crops = []
        for frame in raw_frames:
            crop = video_processor.extract_face_crop(frame, bbox=bbox)
            face_crops.append(crop)

        generate_contact_sheet(
            face_crops,
            f"{vid_path.name} - 16 Face-Cropped Model Inputs (BBox: {bbox})",
            FRAMES_DIR / f"{vid_key}_face_frames.png"
        )

        # Check for screen recording / background dominance
        if bbox is None:
            print(f"[!] WARNING for {vid_key}: Face detector failed to detect any face! Central crop fallback was used.")
        else:
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            face_area_ratio = (bw * bh) / (props["width"] * props["height"])
            print(f"Face Area vs Screen Area: {face_area_ratio * 100:.2f}% of video resolution")

    with open(DEBUG_DIR / "video_properties.json", "w") as f:
        json.dump(all_props, f, indent=4)
    print("\nSaved video_properties.json to experiments/demo_debug/")


if __name__ == "__main__":
    inspect_and_visualize()
