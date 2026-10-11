"""
=========================================================
Resize & Normalize Faces
Agent-Assisted Multimodal Deepfake Detection
=========================================================
"""

import os
import cv2
import numpy as np

from pathlib import Path

from config import *
from utils import *

logger = get_logger("ResizeFaces")


# ==========================================================
# PROCESS SINGLE IMAGE
# ==========================================================

def process_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:

        return False

    try:

        image = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB

        )

    except:

        return False

    image = cv2.resize(

        image,

        (IMAGE_SIZE, IMAGE_SIZE)

    )

    image = image.astype(np.float32)

    image /= 255.0

    image = (image * 255).astype(np.uint8)

    image = cv2.cvtColor(

        image,

        cv2.COLOR_RGB2BGR

    )

    cv2.imwrite(

        str(image_path),

        image

    )

    return True


# ==========================================================
# PROCESS DATASET
# ==========================================================

def process_dataset():

    logger.info("=" * 60)

    logger.info("Resizing Face Images")

    logger.info("=" * 60)

    images = []

    for root, _, files in os.walk(FACE_DIR):

        for file in files:

            if file.endswith(".jpg"):

                images.append(

                    Path(root) / file

                )

    logger.info(f"Images Found : {len(images)}")

    success = 0

    failed = 0

    for image in progress(images):

        ok = process_image(image)

        if ok:

            success += 1

        else:

            failed += 1

    logger.info("=" * 60)

    logger.info(f"Success : {success}")

    logger.info(f"Failed  : {failed}")

    logger.info("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    set_seed()

    process_dataset()