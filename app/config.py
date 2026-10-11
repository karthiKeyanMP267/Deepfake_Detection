"""
app/config.py

Configuration settings for the Multimodal Deepfake Forensic Web Application.
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
TEMP_UPLOAD_DIR = BASE_DIR / "app" / "temp_uploads"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Checkpoint path
CHECKPOINT_PATH = EXPERIMENTS_DIR / "05_full_3v3" / "models" / "best_model.pth"
PHASE6_DIR = EXPERIMENTS_DIR / "06_forensic_analysis"

# Allowed upload extensions
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Video / Frame processing specs
NUM_FRAMES = 16
IMAGE_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH_SEC = 10

# Fixed Classifier & Forensic Thresholds
CLASSIFIER_THRESHOLD = 0.50
AV_CONSISTENCY_THRESHOLD = 0.1177  # Phase-6 validated exploratory threshold

# Hardware device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# App version & metadata
APP_NAME = "Agentic Multimodal Deepfake Forensic Analyzer"
APP_VERSION = "1.0.0"

# LLM Forensic Agent Configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or ""
)

_is_groq = LLM_API_KEY.startswith("gsk_") or os.getenv("LLM_PROVIDER", "").lower() == "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if _is_groq else "openai").lower()

if LLM_PROVIDER == "groq":
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
elif LLM_PROVIDER == "gemini":
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
else:
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "10.0"))

