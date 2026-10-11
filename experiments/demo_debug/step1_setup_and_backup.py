"""
experiments/demo_debug/step1_setup_and_backup.py

Creates debugging directories, copies demonstration videos into debug folder,
and logs current system baseline state.
"""

import os
import shutil
import time
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEBUG_DIR = BASE_DIR / "experiments" / "demo_debug"
VIDEOS_DIR = DEBUG_DIR / "videos"
FRAMES_DIR = DEBUG_DIR / "frames"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Original source paths
src_vid1 = Path(r"C:\Users\karthikeyan\Videos\Screen Recordings\Screen Recording 2026-08-28 222131.mp4")
src_vid2 = Path(r"C:\Users\karthikeyan\Videos\Screen Recordings\Screen Recording 2026-08-28 222513.mp4")

dst_vid1 = VIDEOS_DIR / "video1_222131.mp4"
dst_vid2 = VIDEOS_DIR / "video2_222513.mp4"

if src_vid1.exists():
    shutil.copy2(src_vid1, dst_vid1)
    print(f"Copied Video 1 -> {dst_vid1} ({dst_vid1.stat().st_size:,} bytes)")
else:
    print(f"ERROR: {src_vid1} not found")

if src_vid2.exists():
    shutil.copy2(src_vid2, dst_vid2)
    print(f"Copied Video 2 -> {dst_vid2} ({dst_vid2.stat().st_size:,} bytes)")
else:
    print(f"ERROR: {src_vid2} not found")

# Log Checkpoint State
ckpt_path = BASE_DIR / "experiments" / "05_full_3v3" / "models" / "best_model.pth"
ckpt_stat = ckpt_path.stat()
ckpt_mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ckpt_stat.st_mtime))

ckpt_obj = torch.load(ckpt_path, map_location="cpu")

baseline_state = {
    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    "checkpoint_path": str(ckpt_path),
    "checkpoint_size_bytes": ckpt_stat.st_size,
    "checkpoint_last_modified": ckpt_mtime,
    "checkpoint_epoch": ckpt_obj.get("epoch", "Unknown"),
    "checkpoint_val_f1": float(ckpt_obj.get("best_f1", 0.0)),
    "classification_threshold": 0.50,
    "consistency_threshold": 0.1177,
    "video_1": {
        "original_path": str(src_vid1),
        "debug_path": str(dst_vid1),
        "size_bytes": dst_vid1.stat().st_size if dst_vid1.exists() else 0,
    },
    "video_2": {
        "original_path": str(src_vid2),
        "debug_path": str(dst_vid2),
        "size_bytes": dst_vid2.stat().st_size if dst_vid2.exists() else 0,
    }
}

with open(DEBUG_DIR / "baseline_state.json", "w") as f:
    import json
    json.dump(baseline_state, f, indent=4)

print("Baseline state saved to experiments/demo_debug/baseline_state.json")
