"""
==============================================================
03_detect_faces.py
--------------------------------------------------------------
Detect ONE face per video using SCRFD (InsightFace)
Reuse bounding box for all extracted frames

Author : Your Name
Project: Agentic Multimodal Deepfake Detection
==============================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from insightface.app import FaceAnalysis

from config import *
from utils import *


# ============================================================
# Create Output Folder
# ============================================================

ensure_dir(PROCESSED_FACES_DIR)


# ============================================================
# Initialize SCRFD
# ============================================================

def initialize_detector():

    providers = []

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

    except Exception:
        providers = ["CPUExecutionProvider"]

    app = FaceAnalysis(
        name="buffalo_l",
        root=INSIGHTFACE_ROOT,
        providers=providers
    )

    app.prepare(
        ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
        det_size=(640, 640)
    )

    print("=" * 60)
    print("SCRFD Initialized")
    print("Providers :", providers)
    print("=" * 60)

    return app


# ============================================================321
# Largest Face
# ============================================================

def largest_face(faces):

    if len(faces) == 0:
        return None

    largest = None
    area = 0

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        a = (x2 - x1) * (y2 - y1)

        if a > area:
            area = a
            largest = face

    return largest


# ============================================================
# Detect Face Once
# ============================================================

def detect_bbox(detector, frame_paths):

    for frame_path in frame_paths:

        image = cv2.imread(str(frame_path))

        if image is None:
            continue

        faces = detector.get(image)

        if len(faces) == 0:
            continue

        face = largest_face(faces)

        if face is None:
            continue

        x1, y1, x2, y2 = face.bbox.astype(int)

        return x1, y1, x2, y2

    return None


# ============================================================
# Expand Bounding Box
# ============================================================

def expand_bbox(bbox, width, height, padding=0.15):

    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    padx = int(bw * padding)
    pady = int(bh * padding)

    x1 = max(0, x1 - padx)
    y1 = max(0, y1 - pady)

    x2 = min(width, x2 + padx)
    y2 = min(height, y2 + pady)

    return x1, y1, x2, y2


# ============================================================
# Crop Face
# ============================================================

def crop_face(image, bbox):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = expand_bbox(
        bbox,
        w,
        h,
        padding=0.15
    )

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    face = cv2.resize(
        face,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return face


# ============================================================
# Process One Video
# ============================================================

def process_video(detector, frame_folder, output_folder):

    ensure_dir(output_folder)

    frame_paths = sorted(
        list(Path(frame_folder).glob("*.jpg"))
    )

    if len(frame_paths) == 0:
        return False

    bbox = detect_bbox(detector, frame_paths)

    if bbox is None:
        return False

    for frame in frame_paths:

        image = cv2.imread(str(frame))

        if image is None:
            continue

        face = crop_face(image, bbox)

        if face is None:
            continue

        save_path = output_folder / frame.name

        cv2.imwrite(
            str(save_path),
            face
        )

    return True

    # ============================================================
# Process Entire Dataset
# ============================================================

def process_dataset():

    detector = initialize_detector()

    video_folders = []

    # Collect all frame folders (each folder = one video)
    for root, dirs, files in os.walk(PROCESSED_FRAMES_DIR):

        jpgs = [f for f in files if f.lower().endswith(".jpg")]

        if len(jpgs) > 0:
            video_folders.append(Path(root))

    print("=" * 60)
    print(f"Videos Found : {len(video_folders)}")
    print("=" * 60)

    processed = 0
    skipped = 0
    failed = 0
    no_face = 0

    for frame_folder in tqdm(video_folders, desc="Processing Videos"):

        # Preserve folder structure
        relative = frame_folder.relative_to(PROCESSED_FRAMES_DIR)
        output_folder = PROCESSED_FACES_DIR / relative

        # Resume support
        if output_folder.exists():

            existing = list(output_folder.glob("*.jpg"))

            if len(existing) >= NUM_FRAMES:
                skipped += 1
                continue

        ensure_dir(output_folder)

        try:

            success = process_video(
                detector,
                frame_folder,
                output_folder
            )

            if success:
                processed += 1
            else:
                no_face += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break

        except Exception as e:

            print(f"\nError : {frame_folder}")
            print(e)

            failed += 1

    print("\n")
    print("=" * 60)
    print("FACE DETECTION FINISHED")
    print("=" * 60)
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"No Face   : {no_face}")
    print(f"Failed    : {failed}")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("SCRFD Face Detection")
    print("=" * 60)

    if not PROCESSED_FRAMES_DIR.exists():

        print("Frame directory not found.")
        print(PROCESSED_FRAMES_DIR)
        return

    process_dataset()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    set_seed()

    main()