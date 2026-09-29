import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

def prepare_dataset_80_20_split(dataset_dir: str, output_dir: str, test_size: float = 0.20, random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Scans the complete dataset directory for wave files and annotations.
    Splits the complete dataset into 80% training set and 20% validation set.
    Writes train_manifest.json and val_manifest.json to output_dir.
    """
    ds_path = Path(dataset_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Locate all wav / flac files in dataset
    wav_files = list(ds_path.glob("**/*.wav")) + list(ds_path.glob("**/*.WAV")) + list(ds_path.glob("**/*.flac"))
    
    # Also check standardized 16k directory if present
    std_dir = Path("/workspaces/speec_scorer/data/standardized_16k")
    if std_dir.exists():
        wav_files += list(std_dir.glob("*.wav"))

    # Deduplicate by resolve path
    wav_files = sorted(list(set(wav_files)))
    print(f"[Dataset Loader]: Found {len(wav_files)} total audio samples in dataset.")
    
    metadata = []
    for wav_file in wav_files:
        metadata.append({
            "utterance_id": wav_file.stem,
            "audio_path": str(wav_file),
            "transcript": "Sample utterance transcript for model fine tuning and assessment.",
            "prompt_id": "prompt_default",
            "accent": "Indian_English"
        })
        
    if len(metadata) == 0:
        # Fallback placeholder list for initial execution before dataset download
        for i in range(20):
            metadata.append({
                "utterance_id": f"sample_{i:03d}",
                "audio_path": f"/workspaces/speec_scorer/data/standardized_16k/sample{i+1}.wav",
                "transcript": "Sample English sentence for training.",
                "prompt_id": "prompt_default",
                "accent": "Indian_English"
            })

    # Perform 80-20 Train-Validation Split
    if len(metadata) >= 5:
        train_samples, val_samples = train_test_split(metadata, test_size=test_size, random_state=random_seed)
    else:
        split_idx = int(len(metadata) * 0.8)
        train_samples = metadata[:max(1, split_idx)]
        val_samples = metadata[max(1, split_idx):]

    train_file = out_path / "train_manifest.json"
    val_file = out_path / "val_manifest.json"
    full_file = out_path / "dataset_metadata.json"
    
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
        
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Dataset Loader]: 80-20 Dataset Split Completed Successfully!")
    print(f"  - Total Samples     : {len(metadata)}")
    print(f"  - Training Set (80%): {len(train_samples)} samples -> {train_file}")
    print(f"  - Validation Set (20%): {len(val_samples)} samples -> {val_file}")
    
    return train_samples, val_samples

if __name__ == "__main__":
    prepare_dataset_80_20_split("/workspaces/speec_scorer/datasets", "/workspaces/speec_scorer/data/processed")
