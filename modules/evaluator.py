import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
import soundfile as sf

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ---------------------------------------------------------------------
# Project modules
# ---------------------------------------------------------------------

from modules.config import (
    PROJECT_ROOT,
    STANDARDIZED_AUDIO_DIR,
    DATASETS_DIR,
)

from modules.asr_engine import IndicConformerASR
from modules.feature_extractor import MultimodalFeatureExtractor
from modules.phonetic_scorer import calculate_aligned_gop
from scripts.download_models_and_datasets import ensure_sample_audio_exists


class SpeechEvaluator:
    """
    Production-oriented speech assessment engine.

    Pipeline:

        Audio
          ↓
        ASR
          ↓
        Multimodal Feature Extraction
          ↓
        XGBoost CEFR Prediction
          ↓
        Task-Relevance Adjustment
          ↓
        CEFR Mapping
          ↓
        Diagnostic Skill Profile
          ↓
        Structured Assessment JSON

    IMPORTANT
    ---------
    The CEFR level is determined by the trained XGBoost model.

    The skill-profile scores are diagnostic indicators and are NOT
    averaged together to determine the CEFR level.
    """

    # -----------------------------------------------------------------
    # CEFR scale
    # -----------------------------------------------------------------

    CEFR_LEVELS = {
        1: "A1",
        2: "A2",
        3: "B1",
        4: "B2",
        5: "C1",
        6: "C2",
    }

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def __init__(
        self,
        xgb_model_path: str = None,
        device: str = None,
    ):
        """
        Initialize the speech evaluation engine.

        Args:
            xgb_model_path:
                Optional path to trained XGBoost CEFR model.

            device:
                "cpu" or "cuda".
        """

        # -------------------------------------------------------------
        # Device
        # -------------------------------------------------------------

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(
            f"[Speech Evaluator] Initializing on "
            f"{self.device.type.upper()}..."
        )

        # -------------------------------------------------------------
        # XGBoost model
        # -------------------------------------------------------------

        if xgb_model_path is None:
            self.xgb_model_path = (
                PROJECT_ROOT / "cefr_xgboost_head.json"
            )
        else:
            self.xgb_model_path = Path(xgb_model_path)

        # -------------------------------------------------------------
        # ASR
        # -------------------------------------------------------------

        self.asr_engine = IndicConformerASR(
            device=self.device
        )

        # -------------------------------------------------------------
        # Feature extractor
        # -------------------------------------------------------------

        self.feature_extractor = MultimodalFeatureExtractor()

        # -------------------------------------------------------------
        # XGBoost CEFR head
        # -------------------------------------------------------------

        self.regressor = xgb.XGBRegressor()

        self.model_loaded = False

        if self.xgb_model_path.exists():

            self.regressor.load_model(
                str(self.xgb_model_path)
            )

            self.model_loaded = True

            print(
                "[Speech Evaluator] "
                f"Loaded CEFR XGBoost model: "
                f"{self.xgb_model_path}"
            )

        else:

            print(
                "[Speech Evaluator WARNING] "
                f"CEFR model not found: "
                f"{self.xgb_model_path}"
            )

        print(
            "[Speech Evaluator] "
            "Initialization complete."
        )

    # =================================================================
    # AUDIO
    # =================================================================

    def _resolve_audio(self, audio_path: str = None) -> str:
        """
        Resolve the requested audio path.

        If no valid audio is provided, falls back to sample1.wav.
        """

        if audio_path:

            path = Path(audio_path).expanduser().resolve()

            if path.exists():
                return str(path)

        # -------------------------------------------------------------
        # Fallback sample
        # -------------------------------------------------------------

        sample_path = (
            STANDARDIZED_AUDIO_DIR / "sample1.wav"
        )

        if not sample_path.exists():

            wav_candidates = (
                list(DATASETS_DIR.glob("**/*.wav"))
                + list(DATASETS_DIR.glob("**/*.WAV"))
            )

            first_wav = (
                str(wav_candidates[0])
                if wav_candidates
                else None
            )

            ensure_sample_audio_exists(first_wav)

        if not sample_path.exists():

            raise FileNotFoundError(
                "No valid input audio found and "
                "sample1.wav could not be created."
            )

        print(
            "[Speech Evaluator] "
            f"Using fallback audio: {sample_path}"
        )

        return str(sample_path)

    # -----------------------------------------------------------------

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """
        Read actual audio duration.
        """

        try:

            info = sf.info(audio_path)

            return round(
                float(info.duration),
                2,
            )

        except Exception as exc:

            print(
                "[Speech Evaluator WARNING] "
                f"Could not determine audio duration: {exc}"
            )

            return 0.0

    # =================================================================
    # SAFE FEATURE ACCESS
    # =================================================================

    @staticmethod
    def _feature(
        features: np.ndarray,
        index: int,
        default: float = 0.0,
    ) -> float:
        """
        Safely retrieve a feature value.
        """

        try:

            value = float(features[index])

            if not np.isfinite(value):
                return default

            return value

        except (
            IndexError,
            TypeError,
            ValueError,
        ):

            return default

    # =================================================================
    # CEFR
    # =================================================================

    @classmethod
    def _score_to_cefr(
        cls,
        score: float,
    ) -> str:
        """
        Convert continuous 1-6 CEFR score to discrete CEFR band.

        This preserves the scale assumed by the existing model:

            1 → A1
            2 → A2
            3 → B1
            4 → B2
            5 → C1
            6 → C2

        NOTE:
        This uses nearest CEFR level because the trained model outputs
        a continuous score centered around these levels.
        """

        score = float(
            np.clip(
                score,
                1.0,
                6.0,
            )
        )

        band_index = int(
            np.clip(
                np.floor(score + 0.5),
                1,
                6,
            )
        )

        return cls.CEFR_LEVELS[band_index]

    # =================================================================
    # TASK RELEVANCE
    # =================================================================

    @staticmethod
    def _apply_task_relevance_gate(
        continuous_score: float,
        task_relevance: float,
    ):
        """
        Apply task-relevance constraints.

        < 0.50
            Off-topic → maximum B1 region

        0.50–0.65
            Partially relevant → maximum B2 region

        >= 0.65
            No task-relevance cap
        """

        score = float(continuous_score)

        task_relevance = float(
            np.clip(
                task_relevance,
                0.0,
                1.0,
            )
        )

        # -------------------------------------------------------------
        # Off topic
        # -------------------------------------------------------------

        if task_relevance < 0.50:

            relevance_factor = max(
                0.0,
                (task_relevance - 0.15) / 0.35,
            )

            target_cap = (
                2.50
                + relevance_factor * 0.90
            )

            return (
                min(score, target_cap),
                "OFF_TOPIC",
            )

        # -------------------------------------------------------------
        # Partially relevant
        # -------------------------------------------------------------

        if task_relevance < 0.65:

            relevance_factor = (
                (task_relevance - 0.50)
                / 0.15
            )

            target_cap = (
                3.40
                + relevance_factor * 0.80
            )

            return (
                min(score, target_cap),
                "PARTIALLY_RELEVANT",
            )

        # -------------------------------------------------------------
        # On topic
        # -------------------------------------------------------------

        return (
            score,
            "ON_TOPIC",
        )

    # =================================================================
    # SKILL PROFILE
    # =================================================================

    def _calculate_skill_profile(
        self,
        features: np.ndarray,
    ) -> dict:
        """
        Calculate diagnostic skill indicators.

        IMPORTANT:
        These scores are NOT used to calculate the CEFR level.

        They are intended to explain the candidate's performance.
        """

        # -------------------------------------------------------------
        # Pronunciation
        # Feature 0 = GOP accuracy, 0-100
        # -------------------------------------------------------------

        gop_accuracy = self._feature(
            features,
            0,
        )

        pronunciation_score = round(
            float(
                np.clip(
                    gop_accuracy / 10.0,
                    1.0,
                    10.0,
                )
            ),
            1,
        )

        # -------------------------------------------------------------
        # Fluency
        # -------------------------------------------------------------

        speech_rate = self._feature(
            features,
            5,
        )

        articulation_rate = self._feature(
            features,
            6,
        )

        pause_ratio = self._feature(
            features,
            7,
        )

        fluency_score = round(
            float(
                np.clip(
                    (
                        (speech_rate / 4.2) * 8.0
                        + (
                            1.0
                            - min(
                                pause_ratio,
                                0.5,
                            )
                        )
                        * 2.0
                    ),
                    1.0,
                    10.0,
                )
            ),
            1,
        )

        # -------------------------------------------------------------
        # Grammar
        # -------------------------------------------------------------

        tree_depth = self._feature(
            features,
            11,
        )

        clause_density = self._feature(
            features,
            12,
        )

        grammar_score = round(
            float(
                np.clip(
                    (
                        (tree_depth / 5.0) * 5.0
                        + clause_density * 2.5
                    ),
                    1.0,
                    10.0,
                )
            ),
            1,
        )

        # -------------------------------------------------------------
        # Vocabulary / Coherence
        # -------------------------------------------------------------

        task_relevance = self._feature(
            features,
            18,
        )

        coherence = self._feature(
            features,
            19,
        )

        c1_c2_ratio = self._feature(
            features,
            16,
        )

        vocabulary_score = round(
            float(
                np.clip(
                    (
                        task_relevance * 4.0
                        + coherence * 4.0
                        + (c1_c2_ratio + 0.1) * 2.0
                    ),
                    1.0,
                    10.0,
                )
            ),
            1,
        )

        return {
            "pronunciation": pronunciation_score,
            "fluency": fluency_score,
            "grammar": grammar_score,
            "vocabulary": vocabulary_score,
        }

    # =================================================================
    # PERFORMANCE METRICS
    # =================================================================

    def _calculate_performance(
        self,
        features: np.ndarray,
        task_fulfillment: str,
    ) -> dict:
        """
        Generate human-readable performance indicators.
        """

        speech_rate = self._feature(
            features,
            5,
        )

        articulation_rate = self._feature(
            features,
            6,
        )

        pause_ratio = self._feature(
            features,
            7,
        )

        task_relevance = self._feature(
            features,
            18,
        )

        coherence = self._feature(
            features,
            19,
        )

        # -------------------------------------------------------------
        # Speech rate
        # -------------------------------------------------------------

        speech_wpm = int(
            round(
                speech_rate * 39.5
            )
        )

        articulation_wpm = int(
            round(
                articulation_rate * 39.5
            )
        )

        # -------------------------------------------------------------
        # Pause classification
        # -------------------------------------------------------------

        if pause_ratio < 0.15:

            pause_level = "Minimal"

        elif pause_ratio <= 0.30:

            pause_level = "Moderate"

        else:

            pause_level = "Frequent"

        # -------------------------------------------------------------
        # Task relevance
        # -------------------------------------------------------------

        if task_fulfillment == "ON_TOPIC":

            task_relevance_label = "High"

        elif task_fulfillment == "PARTIALLY_RELEVANT":

            task_relevance_label = "Moderate"

        else:

            task_relevance_label = "Low"

        # -------------------------------------------------------------
        # Coherence
        # -------------------------------------------------------------

        if coherence >= 0.75:

            coherence_label = "High"

        elif coherence >= 0.50:

            coherence_label = "Moderate"

        else:

            coherence_label = "Low"

        return {

            "speech_rate_wpm": speech_wpm,

            "articulation_rate_wpm": articulation_wpm,

            "pause_level": pause_level,

            "task_relevance": round(
                task_relevance,
                3,
            ),

            "task_relevance_label": task_relevance_label,

            "coherence": round(
                coherence,
                3,
            ),

            "coherence_label": coherence_label,
        }

    # =================================================================
    # PHONETIC DIAGNOSTICS
    # =================================================================

    def _phonetic_diagnostics(
        self,
        features: np.ndarray,
    ):
        """
        Generate phonetic diagnostics.

        NOTE:
        We intentionally DO NOT generate random logits.

        If your phonetic scorer later receives real acoustic/model
        logits, pass them here. Until then, return an explicit
        unavailable state rather than fabricated diagnostics.
        """

        return {
            "available": False,
            "message": (
                "Phoneme-level diagnostics require aligned acoustic "
                "model logits and are not generated from synthetic data."
            ),
        }

    # =================================================================
    # MAIN EVALUATION
    # =================================================================

    def evaluate(
        self,
        audio_path: str = None,
        prompt: str = None,
    ) -> dict:
        """
        Execute complete speech evaluation.

        Returns a structured assessment dictionary.
        """

        # -------------------------------------------------------------
        # Prompt
        # -------------------------------------------------------------

        if not prompt:

            prompt = (
                "Describe a situation where you had to "
                "lead a project under tight deadlines."
            )

        # -------------------------------------------------------------
        # Audio
        # -------------------------------------------------------------

        audio_path = self._resolve_audio(
            audio_path
        )

        audio_path = str(
            Path(audio_path).resolve()
        )

        duration = self._get_audio_duration(
            audio_path
        )

        print(
            "[Speech Evaluator] "
            "Transcribing audio..."
        )

        # -------------------------------------------------------------
        # ASR
        # -------------------------------------------------------------

        raw_text = self.asr_engine.transcribe(
            audio_path
        )

        if raw_text is None:
            raw_text = ""

        raw_text = str(raw_text).strip()

        # -------------------------------------------------------------
        # Feature extraction
        # -------------------------------------------------------------

        print(
            "[Speech Evaluator] "
            "Extracting speech features..."
        )

        features = self.feature_extractor.extract_features(
            audio_path,
            prompt,
            raw_text,
        )

        features = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(-1)

        # -------------------------------------------------------------
        # CEFR model prediction
        # -------------------------------------------------------------

        if self.model_loaded:

            model_score = float(
                self.regressor.predict(
                    features.reshape(1, -1)
                )[0]
            )

        else:

            raise RuntimeError(
                "CEFR XGBoost model is not loaded. "
                f"Expected model at: {self.xgb_model_path}"
            )

        model_score = float(
            np.clip(
                model_score,
                1.0,
                6.0,
            )
        )

        # -------------------------------------------------------------
        # Task relevance
        # -------------------------------------------------------------

        task_relevance = self._feature(
            features,
            18,
        )

        adjusted_score, task_fulfillment = (
            self._apply_task_relevance_gate(
                model_score,
                task_relevance,
            )
        )

        adjusted_score = round(
            float(
                np.clip(
                    adjusted_score,
                    1.0,
                    6.0,
                )
            ),
            2,
        )

        # -------------------------------------------------------------
        # CEFR band
        # -------------------------------------------------------------

        assigned_band = self._score_to_cefr(
            adjusted_score
        )

        # -------------------------------------------------------------
        # Confidence interval
        #
        # NOTE:
        # This is retained from your existing implementation.
        # It should eventually be replaced with a calibrated
        # model-specific uncertainty estimate.
        # -------------------------------------------------------------

        ci_lower = round(
            max(
                1.0,
                adjusted_score - 0.16,
            ),
            2,
        )

        ci_upper = round(
            min(
                6.0,
                adjusted_score + 0.16,
            ),
            2,
        )

        # -------------------------------------------------------------
        # Skill profile
        # -------------------------------------------------------------

        skill_scores = self._calculate_skill_profile(
            features
        )

        # -------------------------------------------------------------
        # Performance
        # -------------------------------------------------------------

        performance = self._calculate_performance(
            features,
            task_fulfillment,
        )

        # -------------------------------------------------------------
        # Phonetic diagnostics
        # -------------------------------------------------------------

        phoneme_diagnostics = (
            self._phonetic_diagnostics(
                features
            )
        )

        # -------------------------------------------------------------
        # Feature diagnostics
        # -------------------------------------------------------------

        gop_accuracy = self._feature(
            features,
            0,
        )

        speech_rate = self._feature(
            features,
            5,
        )

        articulation_rate = self._feature(
            features,
            6,
        )

        pause_ratio = self._feature(
            features,
            7,
        )

        filled_pause_rate = self._feature(
            features,
            22,
        )

        mean_run_length = self._feature(
            features,
            8,
        )

        max_tree_depth = self._feature(
            features,
            10,
        )

        mean_tree_depth = self._feature(
            features,
            11,
        )

        clause_density = self._feature(
            features,
            12,
        )

        passive_voice_ratio = self._feature(
            features,
            27,
        )

        academic_word_ratio = self._feature(
            features,
            28,
        )

        lexical_ttr = self._feature(
            features,
            29,
        )

        a1_a2 = self._feature(
            features,
            14,
        )

        b1_b2 = self._feature(
            features,
            15,
        )

        c1_c2 = self._feature(
            features,
            16,
        )

        coherence = self._feature(
            features,
            19,
        )

        # -------------------------------------------------------------
        # Transcript formatting
        # -------------------------------------------------------------

        punctuated_text = (
            raw_text[:1].upper()
            + raw_text[1:]
            + "."
            if raw_text
            else ""
        )

        # -------------------------------------------------------------
        # Final structured result
        # -------------------------------------------------------------

        result = {

            # =========================================================
            # Metadata
            # =========================================================

            "assessment_metadata": {

                "sample_id": Path(
                    audio_path
                ).stem,

                "audio_path": audio_path,

                "duration_seconds": duration,

                "target_prompt": prompt,

                "task_fulfillment": task_fulfillment,

            },

            # =========================================================
            # AUTHORITATIVE CEFR RESULT
            # =========================================================

            "scores": {

                "cefr_band": assigned_band,

                "cefr_continuous": adjusted_score,

                "model_score": round(
                    model_score,
                    3,
                ),

                "confidence_interval_95": [
                    ci_lower,
                    ci_upper,
                ],

                "scoring_scale": {
                    "minimum": 1.0,
                    "maximum": 6.0,
                    "levels": self.CEFR_LEVELS,
                },

            },

            # =========================================================
            # DIAGNOSTIC PROFILE
            # =========================================================

            "skill_scores": skill_scores,

            # =========================================================
            # PERFORMANCE
            # =========================================================

            "performance": performance,

            # =========================================================
            # DETAILED DIAGNOSTICS
            # =========================================================

            "quadrant_breakdown": {

                "pronunciation": {

                    "score": skill_scores[
                        "pronunciation"
                    ],

                    "overall_gop_accuracy": round(
                        gop_accuracy,
                        1,
                    ),

                    "indian_allophone_tolerance_applied": True,

                },

                "fluency": {

                    "score": skill_scores[
                        "fluency"
                    ],

                    "speech_rate_sps": round(
                        speech_rate,
                        2,
                    ),

                    "articulation_rate_sps": round(
                        articulation_rate,
                        2,
                    ),

                    "pause_to_speech_ratio": round(
                        pause_ratio,
                        3,
                    ),

                    "filled_pause_rate_per_min": round(
                        filled_pause_rate,
                        2,
                    ),

                    "mean_run_length_syllables": round(
                        mean_run_length,
                        1,
                    ),

                },

                "grammar_and_syntax": {

                    "score": skill_scores[
                        "grammar"
                    ],

                    "max_dependency_tree_depth": int(
                        round(max_tree_depth)
                    ),

                    "mean_tree_depth": round(
                        mean_tree_depth,
                        2,
                    ),

                    "subordinate_clause_density": round(
                        clause_density,
                        2,
                    ),

                    "passive_voice_ratio": round(
                        passive_voice_ratio,
                        2,
                    ),

                },

                "vocabulary_and_coherence": {

                    "score": skill_scores[
                        "vocabulary"
                    ],

                    "task_relevance_cosine": round(
                        task_relevance,
                        3,
                    ),

                    "inter_sentence_coherence": round(
                        coherence,
                        3,
                    ),

                    "academic_word_list_ratio": round(
                        academic_word_ratio,
                        3,
                    ),

                    "lemmatized_type_token_ratio": round(
                        lexical_ttr,
                        2,
                    ),

                    "lexical_distribution": {

                        "A1_A2": round(
                            a1_a2,
                            2,
                        ),

                        "B1_B2": round(
                            b1_b2,
                            2,
                        ),

                        "C1_C2": round(
                            c1_c2,
                            2,
                        ),

                    },

                },

            },

            # =========================================================
            # TRANSCRIPT
            # =========================================================

            "transcript": {

                "raw": raw_text,

                "punctuated": punctuated_text,

            },

            # =========================================================
            # PHONETIC DIAGNOSTICS
            # =========================================================

            "phoneme_diagnostics": phoneme_diagnostics,

            # =========================================================
            # MODEL INFORMATION
            # =========================================================

            "model_information": {

                "model_type": "XGBoost CEFR regression head",

                "model_path": str(
                    self.xgb_model_path
                ),

                "device": str(
                    self.device
                ),

                "cefr_source": "trained_model",

                "skill_profile_source": "diagnostic_feature_mapping",

            },

        }

        return result


# =====================================================================
# Standalone test
# =====================================================================

if __name__ == "__main__":

    evaluator = SpeechEvaluator()

    result = evaluator.evaluate()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )