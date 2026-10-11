"""
=========================================================
Utility Functions
Agent-Assisted Multimodal Deepfake Detection
=========================================================
"""

import os
import cv2
import random
import shutil
import logging
import subprocess
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from config import *

# ==========================================================
# RANDOM SEED
# ==========================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# LOGGER
# ==========================================================

def get_logger(name="Deepfake"):

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )

        console = logging.StreamHandler()

        console.setFormatter(formatter)

        logger.addHandler(console)

    return logger


logger = get_logger()


# ==========================================================
# CREATE DIRECTORY
# ==========================================================

def create_dir(path):

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# DELETE DIRECTORY
# ==========================================================

def delete_dir(path):

    if os.path.exists(path):

        shutil.rmtree(path)

def ensure_dir(path):
    create_dir(path)


# ==========================================================
# LIST VIDEO FILES
# ==========================================================

def get_video_files(directory):

    videos = []

    for root, _, files in os.walk(directory):

        for file in files:

            if file.lower().endswith(".mp4"):

                videos.append(
                    os.path.join(root, file)
                )

    return sorted(videos)


# ==========================================================
# EXTRACT VIDEO FPS
# ==========================================================

def get_video_fps(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return fps


# ==========================================================
# VIDEO LENGTH
# ==========================================================

def get_video_length(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return frames / fps


# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image(path):

    image = cv2.imread(str(path))

    if image is None:
        return None

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ==========================================================
# SAVE IMAGE
# ==========================================================

def save_image(path, image):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(str(path), image)


# ==========================================================
# RESIZE
# ==========================================================

def resize_image(image):

    return cv2.resize(

        image,

        (IMAGE_SIZE, IMAGE_SIZE)

    )


# ==========================================================
# NORMALIZE
# ==========================================================

def normalize(image):

    image = image.astype(np.float32)

    image /= 255.0

    return image


# ==========================================================
# FRAME SAMPLING
# ==========================================================

def sample_frames(total_frames):

    if total_frames <= MAX_FRAMES:

        return list(range(total_frames))

    return np.linspace(

        0,

        total_frames - 1,

        MAX_FRAMES,

        dtype=int

    )


# ==========================================================
# EXTRACT AUDIO
# ==========================================================

def extract_audio(video_path, output_path):

    cmd = [

        "ffmpeg",

        "-i",

        video_path,

        "-ac",

        str(AUDIO_CHANNELS),

        "-ar",

        str(AUDIO_SAMPLE_RATE),

        "-vn",

        output_path,

        "-y"

    ]

    subprocess.run(

        cmd,

        stdout=subprocess.DEVNULL,

        stderr=subprocess.DEVNULL

    )


# ==========================================================
# PROGRESS BAR
# ==========================================================

def progress(iterable):

    return tqdm(

        iterable,

        ncols=100

    )


# ==========================================================
# DEVICE
# ==========================================================

def get_device():

    return DEVICE


# ==========================================================
# COUNT FILES
# ==========================================================

def count_files(folder, extension=None):

    count = 0

    for root, _, files in os.walk(folder):

        for file in files:

            if extension is None:

                count += 1

            elif file.endswith(extension):

                count += 1

    return count


# ==========================================================
# PRINT INFO
# ==========================================================

def print_info():

    print("=" * 60)

    print("Dataset :", DATASET_DIR)

    print("Frames  :", FRAME_DIR)

    print("Faces   :", FACE_DIR)

    print("Audio   :", AUDIO_DIR)

    print("Device  :", DEVICE)

    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    set_seed()

    print_info()

    logger.info("Utilities Loaded Successfully")
