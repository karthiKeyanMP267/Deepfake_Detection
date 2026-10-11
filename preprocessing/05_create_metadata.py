"""
=========================================================
Metadata Generator
Agent-Assisted Multimodal Deepfake Detection
=========================================================
Creates:

metadata.csv

Columns

video_path
audio_path
face_folder
label
manipulation
=========================================================
"""

import os
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from config import *
from utils import *

logger = get_logger("Metadata")


# ==========================================================
# LABEL
# ==========================================================

def get_label(folder_name):

    if "Fake" in folder_name:

        return 1

    return 0


# ==========================================================
# MANIPULATION TYPE
# ==========================================================

def get_manipulation(folder_name):

    folder = folder_name.lower()

    if "realvideo-realaudio" in folder:

        return "RV_RA"

    elif "fakevideo-realaudio" in folder:

        return "FV_RA"

    elif "realvideo-fakeaudio" in folder:

        return "RV_FA"

    elif "fakevideo-fakeaudio" in folder:

        return "FV_FA"

    return "Unknown"


# ==========================================================
# CREATE METADATA
# ==========================================================

def create_metadata():

    logger.info("=" * 60)

    logger.info("Generating Metadata")

    logger.info("=" * 60)

    rows = []

    videos = get_video_files(DATASET_DIR)

    logger.info(f"Videos Found : {len(videos)}")

    for video in progress(videos):

        relative = Path(video).relative_to(DATASET_DIR)

        folder = relative.parent

        stem = relative.stem

        audio = AUDIO_DIR / folder / (stem + ".wav")

        faces = FACE_DIR / folder / stem

        if not audio.exists():

            continue

        if not faces.exists():

            continue

        frame_count = len(list(faces.glob("*.jpg")))

        if frame_count == 0:

            continue

        rows.append({

            "video_path": str(video),

            "audio_path": str(audio),

            "face_folder": str(faces),

            "frames": frame_count,

            "label": get_label(str(folder)),

            "manipulation": get_manipulation(str(folder))

        })

    df = pd.DataFrame(rows)

    logger.info(f"Samples : {len(df)}")

    df.to_csv(

        METADATA_FILE,

        index=False

    )

    logger.info("Metadata Saved")


    # ======================================================
    # Train Validation Test Split
    # ======================================================

    train_df, temp_df = train_test_split(

        df,

        test_size=0.30,

        random_state=SEED,

        stratify=df["label"]

    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        random_state=SEED,

        stratify=temp_df["label"]

    )

    train_df.to_csv(

        TRAIN_SPLIT,

        index=False

    )

    val_df.to_csv(

        VAL_SPLIT,

        index=False

    )

    test_df.to_csv(

        TEST_SPLIT,

        index=False

    )

    logger.info("Train/Validation/Test Split Saved")

    logger.info("=" * 60)

    logger.info(f"Train : {len(train_df)}")

    logger.info(f"Validation : {len(val_df)}")

    logger.info(f"Test : {len(test_df)}")

    logger.info("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    set_seed()

    create_metadata()
