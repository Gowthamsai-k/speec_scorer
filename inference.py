#!/usr/bin/env python3
"""
Automated Indian-English CEFR Speech Assessment Inference Engine
================================================================
Loads the fine-tuned speech evaluation models into memory once and
evaluates any audio recording against a speaking task/question prompt.

Supports CPU and GPU inference, automatic audio standardization (16 kHz Mono),
verbatim transcription, multimodal feature extraction, and continuous CEFR scoring.

Interactive Mode:
-----------------
    python inference.py

CLI Usage:
----------
    python inference.py --audio sample.wav --question "Explain the benefits of remote work." --device cpu
"""

import os
import sys
import json
import argparse
import tempfile
import torch
import soundfile as sf
import torchaudio
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.config import PROJECT_ROOT, STANDARDIZED_AUDIO_DIR
from modules.audio_processor import standardize_audio
from modules.evaluator import SpeechEvaluator


class SpeechInferenceEngine:
    """
    Persistent in-memory inference engine for Indian-English CEFR Speech Assessment.
    Loads ASR, semantic profiling, and stacking ensemble models once into memory.
    """

    def __init__(self, device: str = None, xgb_model_path: str = None):
        """
        Initialize and load models into memory.

        Args:
            device: 'cpu' or 'cuda' (defaults to 'cuda' if available, else 'cpu').
            xgb_model_path: Optional custom path to XGBoost head model JSON.
        """
        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device.lower()

        self.device = torch.device(self.device_str)
        print(f"[Inference Engine]: Initializing models into memory on device: {self.device_str.upper()}...")

        self.evaluator = SpeechEvaluator(xgb_model_path=xgb_model_path, device=self.device_str)
        print(f"[Inference Engine]: All models loaded and ready for inference.")

    def _prepare_audio(self, audio_path: str) -> str:
        """
        Validates and standardizes audio to 16 kHz Mono 16-bit PCM.
        Handles .wav, .mp3, .flac, .m4a, and stereo downmixing automatically.
        """
        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"Input audio file not found: {audio_path}")

        # Check if already 16kHz mono WAV
        try:
            info = sf.info(str(audio_file))
            if info.samplerate == 16000 and info.channels == 1 and info.format == "WAV":
                return str(audio_file)
        except Exception:
            pass

        # Standardize to temporary 16kHz mono WAV file
        temp_dir = Path(tempfile.gettempdir()) / "speec_scorer_cache"
        temp_dir.mkdir(parents=True, exist_ok=True)
        standardized_path = temp_dir / f"proc_{audio_file.stem}.wav"

        success = standardize_audio(str(audio_file), str(standardized_path))
        if not success or not standardized_path.exists():
            raise RuntimeError(f"Failed to standardize audio file: {audio_path}")

        return str(standardized_path)

    def predict(
        self,
        audio_path: str,
        question: str = "Describe a situation where you had to lead a project under tight deadlines."
    ) -> dict:
        """
        Executes end-to-end evaluation on the provided audio and question.

        Args:
            audio_path: Path to the candidate's speech recording (.wav, .mp3, .flac, etc.).
            question: The interview question / speaking prompt given to the candidate.

        Returns:
            dict containing assessment metadata, CEFR band, continuous score,
            quadrant breakdown (pronunciation, fluency, grammar, vocabulary),
            transcription, and phoneme diagnostics.
        """
        clean_audio = self._prepare_audio(audio_path)
        result = self.evaluator.evaluate(audio_path=clean_audio, prompt=question)
        return result

    def format_summary(self, result: dict) -> str:
        """
        Generates a human-readable text summary of the evaluation results.
        """
        scores = result.get("scores", {})
        quadrant = result.get("quadrant_breakdown", {})
        meta = result.get("assessment_metadata", {})
        transcript = result.get("transcript", {})

        pron = quadrant.get("pronunciation", {})
        flu = quadrant.get("fluency", {})
        gram = quadrant.get("grammar_and_syntax", {})
        vocab = quadrant.get("vocabulary_and_coherence", {})

        summary = [
            "=" * 68,
            "         INDIAN ENGLISH CEFR SPEECH ASSESSMENT REPORT           ",
            "=" * 68,
            f"Question Prompt   : {meta.get('target_prompt', 'N/A')}",
            f"Audio Duration    : {meta.get('duration_seconds', 0):.2f}s",
            "-" * 68,
            f"OVERALL CEFR BAND : {scores.get('cefr_band', 'N/A')}  (Continuous Score: {scores.get('cefr_continuous', 0.0):.2f} / 6.00)",
            f"95% Confidence    : [{scores.get('confidence_interval_95', [0, 0])[0]:.2f}, {scores.get('confidence_interval_95', [0, 0])[1]:.2f}]",
            "-" * 68,
            "QUADRANT BREAKDOWN:",
            f"  * Pronunciation   : {pron.get('overall_gop_accuracy', 0)}% GOP Accuracy (Allophones: {', '.join(pron.get('allophones_detected', []))})",
            f"  * Fluency         : {flu.get('speech_rate_sps', 0):.2f} sps | Articulation: {flu.get('articulation_rate_sps', 0):.2f} sps | Pause Ratio: {flu.get('pause_to_speech_ratio', 0):.2f}",
            f"  * Grammar/Syntax  : Max Tree Depth: {gram.get('max_dependency_tree_depth', 0)} | Clause Density: {gram.get('subordinate_clause_density', 0)}",
            f"  * Vocabulary/Coh. : Task Relevance: {vocab.get('task_relevance_cosine', 0):.3f} | Coherence: {vocab.get('inter_sentence_coherence', 0):.3f}",
            "-" * 68,
            f"TRANSCRIPT        : \"{transcript.get('punctuated', '')}\"",
            "=" * 68,
        ]
        return "\n".join(summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CEFR Speech Assessment Inference on Audio Recording and Question Prompt."
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        default=None,
        help="Path to the input speech recording (.wav, .mp3, .flac, etc.)."
    )
    parser.add_argument(
        "--question", "-q", "--prompt", "-p",
        type=str,
        default=None,
        help="Speaking prompt or question presented to the candidate."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device for model inference ('cpu' or 'cuda'). Default: 'cpu'."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional path to save full evaluation JSON output."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON output instead of human-readable summary."
    )
    return parser.parse_args()


def prompt_user_inputs(default_audio: str, default_question: str) -> tuple[str, str]:
    """
    Explicitly prompts the user in the terminal for the recording path and question.
    """
    print("\n" + "=" * 68)
    print("      INDIAN ENGLISH CEFR SPEECH ASSESSMENT - INFERENCE INPUT       ")
    print("=" * 68)

    # 1. Prompt for Audio Recording Path
    while True:
        audio_input = input(f"\nEnter the speech recording audio file path\n[Default: {default_audio}]: ").strip()
        if not audio_input:
            audio_path = default_audio
        else:
            audio_path = audio_input.strip("'\"")

        if Path(audio_path).exists():
            print(f"[Verified]: Audio file found -> {Path(audio_path).resolve()}")
            break
        else:
            print(f"[Error]: File '{audio_path}' does not exist. Please re-enter a valid file path.")

    # 2. Prompt for Question / Task
    question_input = input(f"\nEnter the speaking prompt / question presented to the speaker\n[Default: {default_question}]: ").strip()
    if not question_input:
        question = default_question
    else:
        question = question_input

    print(f"[Verified]: Question prompt -> \"{question}\"")
    print("=" * 68 + "\n")
    return audio_path, question


def main():
    args = parse_args()

    default_sample = str(STANDARDIZED_AUDIO_DIR / "sample1.wav")
    default_prompt = "Describe a situation where you had to lead a project under tight deadlines."

    # If --audio or --question is not provided via CLI flags, interactively ask the user
    if args.audio is None or args.question is None:
        audio_path, question = prompt_user_inputs(
            default_audio=args.audio or default_sample,
            default_question=args.question or default_prompt
        )
    else:
        audio_path = args.audio
        question = args.question

    engine = SpeechInferenceEngine(device=args.device)
    result = engine.predict(audio_path=audio_path, question=question)

    if args.output:
        out_file = Path(args.output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[SUCCESS]: Results saved to -> {out_file}")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + engine.format_summary(result))


if __name__ == "__main__":
    main()
