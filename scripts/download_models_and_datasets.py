import os
import json
import tarfile
import urllib.request
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from huggingface_hub import snapshot_download
from modules.config import MODELS_DIR, DATASETS_DIR, STANDARDIZED_AUDIO_DIR

models_to_download = [
    ("indic-conformer-600m-multilingual", "ai4bharat/indic-conformer-600m-multilingual"),
    ("indicwav2vec-hindi", "ai4bharat/indicwav2vec-hindi"),
    ("punctuation-multilingual", "oliverguhr/fullstop-punctuation-multilingual-sonar-base"),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
    ("distilroberta-base", "distilbert/distilroberta-base")
]

def ensure_sample_audio_exists(first_wav_path: str = None):
    """
    Ensures data/standardized_16k/sample1.wav exists for pipeline inference tests.
    """
    sample1_path = STANDARDIZED_AUDIO_DIR / "sample1.wav"
    STANDARDIZED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    if sample1_path.exists():
        print(f"[Sample Audio Check]: sample1.wav exists -> {sample1_path}")
        return True
        
    if first_wav_path and Path(first_wav_path).exists():
        print(f"[Sample Audio Check]: Normalizing {first_wav_path} to {sample1_path}...")
        try:
            data, sr = sf.read(first_wav_path)
            wav = torch.from_numpy(data).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.T
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)
            peak = torch.max(torch.abs(wav))
            if peak > 0:
                wav = (wav / peak) * 0.95
            sf.write(str(sample1_path), wav.squeeze(0).numpy(), 16000, subtype="PCM_16")
            print(f"[SUCCESS]: Created sample1.wav -> {sample1_path}")
            return True
        except Exception as e:
            print(f"[WARN]: Could not normalize first wav: {e}")

    # Generate synthetic 16 kHz PCM wave if no audio file is present
    print(f"[Sample Audio Check]: Generating standard 16 kHz sample1.wav...")
    sample_rate = 16000
    t = torch.linspace(0, 3, sample_rate * 3) # 3 seconds
    waveform = 0.3 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0) # 440 Hz tone
    sf.write(str(sample1_path), waveform.squeeze(0).numpy(), sample_rate, subtype="PCM_16")
    print(f"[SUCCESS]: Generated fallback sample1.wav -> {sample1_path}")
    return True

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

    print("\n=== Downloading & Extracting Speech ocean 762 Complete Dataset ===")
    ds_target = DATASETS_DIR / "speechocean762"
    ds_target.mkdir(parents=True, exist_ok=True)
    
    # 1. Check if tar.gz exists locally or download from openslr
    tar_file = DATASETS_DIR / "speechocean762.tar.gz"
    if not tar_file.exists():
        print("[Downloading Speechocean762 Dataset Tarball from OpenSLR...]")
        try:
            url = "https://www.openslr.org/resources/101/speechocean762.tar.gz"
            urllib.request.urlretrieve(url, str(tar_file))
            print(f"[SUCCESS]: Downloaded speechocean762.tar.gz -> {tar_file}")
        except Exception as e:
            print(f"[WARN downloading speechocean762.tar.gz]: {e}")

    # 2. Extract tar.gz if present
    if tar_file.exists():
        print(f"[Extracting Speechocean762 Tarball]: {tar_file} -> {ds_target}")
        try:
            with tarfile.open(str(tar_file), "r:gz") as tar:
                tar.extractall(path=str(ds_target))
            print(f"[SUCCESS]: Extracted speechocean762 dataset to {ds_target}")
        except Exception as e:
            print(f"[WARN extracting tarball]: {e}")

    # 3. Locate WAV files and ensure sample1.wav exists
    wav_files = list(ds_target.glob("**/*.wav")) + list(ds_target.glob("**/*.WAV")) + list(ds_target.glob("**/*.flac"))
    print(f"[Dataset Verified]: Total WAV files extracted = {len(wav_files)}")
    
    first_wav = str(wav_files[0]) if len(wav_files) > 0 else None
    ensure_sample_audio_exists(first_wav)

if __name__ == "__main__":
    download_all_models_and_datasets()
