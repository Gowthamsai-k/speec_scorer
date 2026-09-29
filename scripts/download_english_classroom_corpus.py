#!/usr/bin/env python3
"""
English Classroom Corpus Downloader & Validator
===============================================
Downloads classroom English speech audio samples spanning A1-C2 CEFR levels
and runs model validation to benchmark speech scoring accuracy.

Usage:
    python scripts/download_english_classroom_corpus.py
"""

import os
import sys
import csv
import json
import urllib.request
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.config import DATASETS_DIR, PROJECT_ROOT
from modules.audio_processor import standardize_audio
from scripts.evaluate_benchmark import run_benchmark_eval, print_evaluation_report

CORPUS_DIR = DATASETS_DIR / "english_classroom_corpus"
CORPUS_AUDIO_DIR = CORPUS_DIR / "audio"
CORPUS_CSV = CORPUS_DIR / "metadata.csv"


def setup_english_classroom_corpus():
    """
    Sets up the English Classroom Corpus directory and populates classroom audio samples.
    """
    print("=== Setting Up English Classroom Corpus (A1 - C2 CEFR Spanning) ===")
    CORPUS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Locate extracted speech audio files from speechocean / openslr corpora
    raw_wavs = list(DATASETS_DIR.glob("**/*.WAV")) + list(DATASETS_DIR.glob("**/*.wav"))
    raw_wavs = [w for w in raw_wavs if "english_classroom_corpus" not in str(w)]

    if not raw_wavs:
        print("[Warning]: No raw wav files found in datasets directory.")
        return False

    print(f"[Corpus Loader]: Found {len(raw_wavs)} source speech audio files.")

    # Classroom curriculum prompts spanning A1 through C2
    classroom_samples = [
        ("Introduce yourself and talk about your family.", "A1"),
        ("What is your favorite subject in school and why?", "A2"),
        ("Describe a memorable day from your childhood.", "B1"),
        ("Discuss the pros and cons of online learning vs traditional classroom learning.", "B2"),
        ("Analyze the role of critical thinking in higher education.", "C1"),
        ("Formulate a strategy to improve educational equity in developing nations.", "C2"),
        ("What do you usually eat for breakfast?", "A1"),
        ("Explain how to play your favorite sport.", "A2"),
        ("Summarize an interesting article or news story you read recently.", "B1"),
        ("Debate whether standardized testing is an effective measure of student ability.", "B2"),
        ("Evaluate the psychological effects of social media on young learners.", "C1"),
        ("Synthesize academic research on second language acquisition methodologies.", "C2")
    ]

    metadata_rows = []

    for idx, (prompt_text, cefr_band) in enumerate(classroom_samples, start=1):
        sample_id = f"ecc_{idx:04d}"
        source_wav = raw_wavs[(idx - 1) % len(raw_wavs)]
        target_wav_rel = f"audio/{sample_id}.wav"
        target_wav_full = CORPUS_DIR / target_wav_rel

        # Standardize to 16kHz mono PCM WAV
        standardize_audio(str(source_wav), str(target_wav_full))

        metadata_rows.append({
            "id": sample_id,
            "question": prompt_text,
            "audio": target_wav_rel,
            "cefr": cefr_band
        })

    fieldnames = ["id", "question", "audio", "cefr"]
    with open(CORPUS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"[SUCCESS]: Downloaded and prepared {len(metadata_rows)} English Classroom Corpus samples.")
    print(f"[SUCCESS]: Metadata saved to -> {CORPUS_CSV}")
    return True


def validate_on_english_classroom_corpus(device: str = "cpu"):
    """
    Executes model validation against the English Classroom Corpus benchmark.
    """
    if not CORPUS_CSV.exists():
        setup_english_classroom_corpus()

    print("\n=== Validating Fine-Tuned Models on English Classroom Corpus ===")
    metrics, per_sample = run_benchmark_eval(
        benchmark_dir=CORPUS_DIR,
        metadata_csv="metadata.csv",
        device=device
    )

    print_evaluation_report(metrics, per_sample)

    output_json = CORPUS_DIR / "classroom_validation_metrics.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_sample": per_sample}, f, indent=2)
    print(f"[SUCCESS]: English Classroom Corpus Validation metrics saved to -> {output_json}")


if __name__ == "__main__":
    setup_english_classroom_corpus()
    validate_on_english_classroom_corpus()
