"""Configuration constants and paths."""
import os
from pathlib import Path

# Project root (where this repo lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# GroundingDINO via Hugging Face transformers — self-contained, no external
# repo/weights checkout needed. Downloaded once and cached under
# ~/.cache/huggingface/hub. "tiny" backbone is CPU-friendly.
GROUNDINGDINO_MODEL_ID = os.environ.get(
    "GROUNDINGDINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny"
)

_LOCAL_MODELS = PROJECT_ROOT / "models"
# SAM weights (optional — not loaded by default to stay under VRAM budget)
SAM_WEIGHTS = _LOCAL_MODELS / "sam_vit_h_4b8939.pth"

# Output
OUTPUT_ROOT = PROJECT_ROOT / "output" / "sessions"

# LLM services
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")

# Server
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

# Detection thresholds
GROUNDINGDINO_BOX_THRESHOLD = 0.30
GROUNDINGDINO_TEXT_THRESHOLD = 0.25
SAM_TOP_K_BOXES = 5

# OCR
OCR_ENABLED = os.environ.get("OCR_ENABLED", "1") != "0"
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "en,ch_tra").split(",")
OCR_MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.3"))
OCR_MAX_RESULTS = int(os.environ.get("OCR_MAX_RESULTS", "15"))

# Timeouts
VLM_TIMEOUT_S = 60
GOAL_DECOMPOSE_TIMEOUT_S = 30


def ensure_output_dir(session_id: str) -> Path:
    p = OUTPUT_ROOT / session_id
    (p / "photo").mkdir(parents=True, exist_ok=True)
    (p / "annotated").mkdir(parents=True, exist_ok=True)
    return p
