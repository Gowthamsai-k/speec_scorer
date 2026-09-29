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
import textwrap
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

    @staticmethod
    def _bar(score: float, max_score: float = 10.0, length: int = 10) -> str:
        filled = int(round((score / max_score) * length))
        filled = max(0, min(length, filled))
        return "█" * filled + "░" * (length - filled)

    @staticmethod
    def _generate_assessment_comment(band: str, score: float, task_fulfillment: str) -> str:
        if task_fulfillment == "OFF_TOPIC":
            return (
                "The response is largely off-topic relative to the given prompt. "
                "While speech delivery and grammatical structures may show baseline proficiency, "
                "task achievement is constrained."
            )
        elif score >= 5.0:
            return (
                "The speaker demonstrates exceptional fluency, sophisticated vocabulary, "
                "and complex grammatical structures with precise target task alignment."
            )
        elif score >= 3.5:
            return (
                "The speaker demonstrates effective communication with consistent fluency, "
                "clear pronunciation, and appropriate vocabulary for the prompt."
            )
        elif score >= 2.5:
            return (
                "The speaker communicates main ideas clearly with moderate fluency. "
                "Occasional pauses or simplified sentence structures are observed."
            )
        else:
            return (
                "Basic speech production detected. Expanding vocabulary depth, sentence length, "
                "and articulation rate will improve overall proficiency."
            )

    def format_report_card(self, result: dict) -> str:
        """
        Renders candidate-facing assessment report card formatted inside a clean box.
        """
        meta = result.get("assessment_metadata", {})
        scores = result.get("scores", {})
        skills = result.get("skill_scores", {})
        perf = result.get("performance", {})
        transcript = result.get("transcript", {}).get("punctuated", "")

        band = scores.get("cefr_band", "N/A")
        score_val = scores.get("cefr_continuous", 0.0)
        ci = scores.get("confidence_interval_95", [0.0, 0.0])
        task_ful = meta.get("task_fulfillment", "ON_TOPIC")

        W = 62  # inner width between borders

        def box_center(text: str) -> str:
            return f"│{text.center(W)}│"

        def box_line(left: str, right: str = "") -> str:
            if not right:
                return f"│  {left:<{W-4}}  │"
            space = W - 4 - len(left) - len(right)
            return f"│  {left}{' ' * space}{right}  │"

        def box_skill(label: str, score: float) -> str:
            bar_str = self._bar(score)
            score_str = f"{score:.1f}/10"
            right_side = f"{bar_str}  {score_str:>6}"
            space = W - 4 - len(label) - len(right_side)
            return f"│  {label}{' ' * space}{right_side}  │"

        lines = []
        lines.append("╭" + "─" * W + "╮")
        lines.append(box_center("SPEAKING ASSESSMENT"))
        lines.append(box_center("English Proficiency"))
        lines.append("├" + "─" * W + "┤")
        lines.append(box_center(""))
        lines.append(box_line("Overall Level", band))
        lines.append(box_line("Speaking Score", f"{score_val:.2f} / 6.00"))
        lines.append(box_line("Confidence", f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"))
        lines.append(box_center(""))
        lines.append("├" + "─" * W + "┤")
        lines.append(box_line("SKILL PROFILE"))
        lines.append(box_center(""))
        lines.append(box_skill("Pronunciation", skills.get("pronunciation", 8.7)))
        lines.append(box_skill("Fluency", skills.get("fluency", 7.9)))
        lines.append(box_skill("Grammar", skills.get("grammar", 7.8)))
        lines.append(box_skill("Vocabulary & Coherence", skills.get("vocabulary", 8.4)))
        lines.append(box_center(""))
        lines.append("├" + "─" * W + "┤")
        lines.append(box_line("SPEAKING PERFORMANCE"))
        lines.append(box_center(""))
        lines.append(box_line("Speech Rate", f"{perf.get('speech_rate_wpm', 128)} words/min"))
        lines.append(box_line("Articulation Rate", f"{perf.get('articulation_rate_wpm', 146)} words/min"))
        lines.append(box_line("Pause Level", perf.get('pause_level', 'Minimal Pauses')))
        lines.append(box_line("Task Relevance", perf.get('task_relevance_label', 'High Relevance')))
        lines.append(box_line("Coherence", perf.get('coherence_label', 'High Coherence')))
        lines.append(box_center(""))
        lines.append("├" + "─" * W + "┤")
        lines.append(box_line("TRANSCRIPT"))
        lines.append(box_center(""))

        wrapped_transcript = textwrap.wrap(f'"{transcript}"' if transcript else '""', width=W - 6)
        for tline in wrapped_transcript:
            lines.append(f"│   {tline:<{W-6}}   │")
        lines.append(box_center(""))

        lines.append("├" + "─" * W + "┤")
        lines.append(box_line("ASSESSMENT SUMMARY"))
        lines.append(box_center(""))

        comment = self._generate_assessment_comment(band, score_val, task_ful)
        wrapped_comment = textwrap.wrap(comment, width=W - 6)
        for cline in wrapped_comment:
            lines.append(f"│   {cline:<{W-6}}   │")
        lines.append(box_center(""))
        lines.append("╰" + "─" * W + "╯")

        return "\n".join(lines)

    def format_summary(self, result: dict) -> str:
        """Alias for format_report_card for backward compatibility."""
        return self.format_report_card(result)


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
        print("\n" + engine.format_report_card(result))


if __name__ == "__main__":
    main()
