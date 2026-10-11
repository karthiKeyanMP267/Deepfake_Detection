"""
=========================================================
Frame Extraction
Agent-Assisted Multimodal Deepfake Detection
=========================================================
"""

import os
import cv2

from pathlib import Path

from config import *
from utils import *

logger = get_logger("FrameExtractor")


# ==========================================================
# EXTRACT UNIFORMLY SAMPLED FRAMES
# ==========================================================

def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        logger.warning(f"Cannot open : {video_path}")

        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = sample_frames(total_frames)

    relative_path = Path(video_path).relative_to(DATASET_DIR)

    save_folder = FRAME_DIR / relative_path.parent / relative_path.stem

    create_dir(save_folder)

    saved = 0
    current = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if current in frame_indices:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb = resize_image(rgb)

            save_path = save_folder / f"{saved:03d}.jpg"

            save_image(save_path, rgb)

            saved += 1

        current += 1

    cap.release()


# ==========================================================
# MAIN
# ==========================================================

def main():

    logger.info("Searching videos...")

    videos = get_video_files(DATASET_DIR)

    logger.info(f"Videos Found : {len(videos)}")

    for video in progress(videos):

        relative = Path(video).relative_to(DATASET_DIR)

        output_folder = FRAME_DIR / relative.parent / relative.stem

        # Skip already processed videos

        if output_folder.exists():

            if len(list(output_folder.glob("*.jpg"))) >= MAX_FRAMES:

                continue

        try:

            extract_frames(video)

        except Exception as e:

            logger.error(f"{video}")

            logger.error(e)

    logger.info("Frame Extraction Completed")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    set_seed()

    main()
