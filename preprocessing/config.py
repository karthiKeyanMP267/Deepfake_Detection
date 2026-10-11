from pathlib import Path
import torch

# ==============================================================
# BASE DIRECTORY
# ==============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# ==============================================================
# PATHS
# ==============================================================
DATASET_DIR = BASE_DIR / "dataset" / "FakeAVCeleb"
FRAME_DIR = BASE_DIR / "processed" / "frames"
FACE_DIR = BASE_DIR / "processed" / "faces"
AUDIO_DIR = BASE_DIR / "processed" / "audio"

PROCESSED_FRAMES_DIR = FRAME_DIR
PROCESSED_FACES_DIR = FACE_DIR

METADATA_FILE = BASE_DIR / "processed" / "metadata.csv"
TRAIN_SPLIT = BASE_DIR / "processed" / "train.csv"
VAL_SPLIT = BASE_DIR / "processed" / "val.csv"
TEST_SPLIT = BASE_DIR / "processed" / "test.csv"

INSIGHTFACE_ROOT = str(BASE_DIR / ".insightface")


# ==============================================================
# HYPERPARAMETERS / SETTINGS
# ==============================================================
SEED = 42
MAX_FRAMES = 10
NUM_FRAMES = 10
IMAGE_SIZE = 224

AUDIO_CHANNELS = 1
AUDIO_SAMPLE_RATE = 16000

# ==============================================================
# PHASE-3 VISION ENCODER & EXPERIMENT CONFIGURATION
# ==============================================================
ENABLE_EFFICIENTNET = True
ENABLE_VIDEO_SWIN = True
VIDEO_SWIN_MODEL = "swin3d_t"
VIDEO_SWIN_FEATURE_DIM = 768
VISION_FUSION_DIM = 1280

EXPERIMENT_DIR = BASE_DIR / "experiments" / "03_vision_swin"
EXPERIMENT_03_DIR = EXPERIMENT_DIR


# ==============================================================
# HARDWARE
# ==============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"