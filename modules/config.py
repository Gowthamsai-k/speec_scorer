import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Dynamically resolves the root directory of the project repository,
    ensuring paths work in Google Colab, Linux, macOS, Windows, or Codespaces.
    """
    curr = Path(__file__).resolve().parent
    for parent in [curr] + list(curr.parents):
        if (parent / "modules").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()

PROJECT_ROOT = get_project_root()
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STANDARDIZED_AUDIO_DIR = DATA_DIR / "standardized_16k"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
STANDARDIZED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
