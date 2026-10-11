"""
=========================================================
Audio Extraction
Agent-Assisted Multimodal Deepfake Detection
=========================================================
Extract audio from every video
Convert to:
    - Mono
    - 16kHz
    - WAV
=========================================================
"""

from pathlib import Path

from config import *
from utils import *

logger = get_logger("AudioExtractor")


# ==========================================================
# AUDIO EXTRACTION
# ==========================================================

def process_video(video_path):

    relative_path = Path(video_path).relative_to(DATASET_DIR)

    output_audio = AUDIO_DIR / relative_path.parent / (
        relative_path.stem + ".wav"
    )

    create_dir(output_audio.parent)

    # Skip if already processed
    if output_audio.exists():

        return

    extract_audio(

        str(video_path),

        str(output_audio)

    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("Audio Extraction Started")
    logger.info("=" * 60)

    videos = get_video_files(DATASET_DIR)

    logger.info(f"Videos Found : {len(videos)}")

    failed = 0

    for video in progress(videos):

        try:

            process_video(video)

        except Exception as e:

            failed += 1

            logger.error(video)

            logger.error(e)

    logger.info("=" * 60)
    logger.info("Completed")
    logger.info(f"Failed Videos : {failed}")
    logger.info("=" * 60)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    set_seed()

    main()
