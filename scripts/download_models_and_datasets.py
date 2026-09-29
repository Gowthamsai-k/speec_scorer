import os
import json
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download

WORKSPACE_ROOT = Path("/workspaces/speec_scorer") if Path("/workspaces/speec_scorer").exists() else Path(os.getcwd())
MODELS_DIR = WORKSPACE_ROOT / "models"
DATASETS_DIR = WORKSPACE_ROOT / "datasets"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

models_to_download = [
    ("indic-conformer-600m-multilingual", "ai4bharat/indic-conformer-600m-multilingual"),
    ("indicwav2vec-hindi", "ai4bharat/indicwav2vec-hindi"),
    ("punctuation-multilingual", "oliverguhr/fullstop-punctuation-multilingual-sonar-base"),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
    ("distilroberta-base", "distilbert/distilroberta-base")
]

def download_all_models_and_datasets():
    print("=== Downloading HuggingFace Pretrained Models ===")
    for folder_name, repo_id in models_to_download:
        target_path = MODELS_DIR / folder_name
        print(f"\n[Downloading Model]: {repo_id} -> {target_path}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_path),
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
            )
            print(f"[SUCCESS]: {repo_id} downloaded.")
        except Exception as e:
            print(f"[WARN downloading {repo_id}]: {e}")

    print("\n=== Downloading Speech ocean 762 Dataset ===")
    ds_target = DATASETS_DIR / "speechocean762"
    try:
        snapshot_download(
            repo_id="misahub/speechocean762",
            repo_type="dataset",
            local_dir=str(ds_target)
        )
        print(f"[SUCCESS]: speechocean762 dataset downloaded to {ds_target}")
    except Exception as e:
        print(f"[WARN]: Could not download misahub/speechocean762 directly: {e}")

if __name__ == "__main__":
    download_all_models_and_datasets()
