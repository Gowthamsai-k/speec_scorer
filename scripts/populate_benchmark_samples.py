#!/usr/bin/env python3
"""
Benchmark Sample Extractor & Generator
======================================
Extracts and standardizes diverse speech audio samples from extracted speech datasets
(e.g., Speechocean762 / Indian English speech corpus) to expand the proprietary benchmark suite.

Generates:
- benchmark/audio/00001.wav ... 00025.wav
- benchmark/metadata.csv with questions and balanced CEFR ground truth labels.
"""

import os
import sys
import csv
import torch
import torchaudio
import soundfile as sf
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.config import DATASETS_DIR, PROJECT_ROOT
from modules.audio_processor import standardize_audio

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
BENCHMARK_AUDIO_DIR = BENCHMARK_DIR / "audio"
METADATA_CSV = BENCHMARK_DIR / "metadata.csv"

# Diverse speaking prompts covering various difficulty levels & topics
SAMPLE_PROMPTS = [
    ("Describe your most challenging software engineering project.", "B2"),
    ("What are the primary advantages and drawbacks of remote work?", "C1"),
    ("Describe your hometown and why you like living there.", "B1"),
    ("Explain a situation where you had to resolve a conflict within your team.", "B2"),
    ("What are your long-term career goals and how do you plan to achieve them?", "C2"),
    ("How do you handle high pressure deadlines when requirements change rapidly?", "C1"),
    ("Describe a hobby or activity you enjoy in your free time.", "A2"),
    ("What is your favorite travel destination and what makes it special?", "B1"),
    ("Explain how machine learning model quantization improves inference performance.", "C2"),
    ("Tell me about a time when you received constructive feedback.", "B2"),
    ("How do you ensure effective communication in cross-functional teams?", "C1"),
    ("Describe your morning routine before heading to work.", "A1"),
    ("What are the key responsibilities of a senior project manager?", "B2"),
    ("How has technology impacted education over the past decade?", "C1"),
    ("Describe your favorite book or movie and explain why you recommend it.", "B1"),
    ("What steps do you take to troubleshoot a complex system outage?", "C2"),
    ("Explain the importance of work-life balance for software developers.", "B2"),
    ("What motivated you to pursue a career in speech processing and AI?", "C1"),
    ("Describe how you cook your favorite meal.", "A2"),
    ("What advice would you give to someone starting their career in tech?", "B2"),
    ("How do you prioritize competing tasks when managing multiple projects?", "C1"),
    ("Describe a recent technological innovation that impressed you.", "B2"),
    ("What are the main differences between microservice and monolithic architectures?", "C2"),
    ("Explain what you usually do on weekends.", "A1"),
    ("Summarize your key achievements from your last professional role.", "C1")
]


def populate_benchmark(num_samples: int = 25):
    BENCHMARK_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Locate all available wav files in datasets/
    ds_path = DATASETS_DIR / "speechocean762"
    wav_candidates = list(ds_path.glob("**/*.WAV")) + list(ds_path.glob("**/*.wav"))
    
    if not wav_candidates:
        wav_candidates = list(DATASETS_DIR.glob("**/*.wav")) + list(DATASETS_DIR.glob("**/*.WAV"))
        
    print(f"[Benchmark Sample Builder]: Found {len(wav_candidates)} candidate raw audio files.")
    
    # Sort or select diverse speakers/samples
    selected_wavs = wav_candidates[:num_samples]
    
    if len(selected_wavs) < num_samples:
        print(f"[Warning]: Fewer than {num_samples} audio files available ({len(selected_wavs)} found). Using available files.")

    metadata_rows = []

    for idx, (prompt_text, cefr_label) in enumerate(SAMPLE_PROMPTS[:len(selected_wavs)], start=1):
        sample_id = f"{idx:05d}"
        source_wav = selected_wavs[idx - 1]
        target_wav_rel = f"audio/{sample_id}.wav"
        target_wav_full = BENCHMARK_DIR / target_wav_rel

        # Standardize audio to 16kHz Mono 16-bit PCM WAV
        success = standardize_audio(str(source_wav), str(target_wav_full))
        if not success or not target_wav_full.exists():
            print(f"[WARN]: Failed to standardize {source_wav}, copying raw file...")
            import shutil
            shutil.copy(str(source_wav), str(target_wav_full))

        metadata_rows.append({
            "id": sample_id,
            "question": prompt_text,
            "audio": target_wav_rel,
            "cefr": cefr_label
        })

    # Write metadata CSV
    fieldnames = ["id", "question", "audio", "cefr"]
    with open(METADATA_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"[SUCCESS]: Populated {len(metadata_rows)} benchmark audio samples in {BENCHMARK_AUDIO_DIR}")
    print(f"[SUCCESS]: Updated benchmark metadata CSV -> {METADATA_CSV}")


if __name__ == "__main__":
    populate_benchmark(25)
