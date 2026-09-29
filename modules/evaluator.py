import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import xgboost as xgb
import numpy as np
from pathlib import Path
from modules.asr_engine import IndicConformerASR
from modules.feature_extractor import MultimodalFeatureExtractor

class SpeechEvaluator:
    def __init__(self, xgb_model_path: str = "cefr_xgboost_head.json"):
        self.xgb_model_path = Path(xgb_model_path)
        self.asr_engine = IndicConformerASR()
        self.feature_extractor = MultimodalFeatureExtractor()
        
        self.regressor = xgb.XGBRegressor()
        if self.xgb_model_path.exists():
            self.regressor.load_model(str(self.xgb_model_path))
            print(f"[Evaluator]: Loaded XGBoost head from {self.xgb_model_path}")
        else:
            print(f"[Evaluator Warning]: Model path {self.xgb_model_path} not found. Using untrained initialization.")

    def evaluate(self, audio_path: str, prompt: str) -> dict:
        """
        Executes end-to-end evaluation: ASR -> NLP -> GOP -> 22D Feature Vector -> CEFR Score.
        """
        raw_text = self.asr_engine.transcribe(audio_path)
        features = self.feature_extractor.extract_features(audio_path, prompt, raw_text)

        if self.xgb_model_path.exists():
            continuous_score = float(self.regressor.predict(features.reshape(1, -1))[0])
        else:
            # Simulated continuous score default
            continuous_score = 3.50

        continuous_score = round(float(np.clip(continuous_score, 1.00, 6.00)), 2)
        band_index = int(round(np.clip(continuous_score, 1.0, 6.0)))
        cefr_levels = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1", 6: "C2"}
        assigned_band = cefr_levels[band_index]

        return {
            "cefr_band": assigned_band,
            "cefr_score_continuous": continuous_score,
            "transcript": raw_text,
            "task_relevance": round(float(features[18]), 3),
            "mean_gop_accuracy": round(float(features[0]), 1),
            "speech_rate_sps": round(float(features[5]), 2)
        }

if __name__ == "__main__":
    evaluator = SpeechEvaluator()
    sample_audio = "/workspaces/speec_scorer/data/standardized_16k/sample1.wav"
    prompt_text = "Describe a situation where you had to lead a project under tight deadlines."
    if Path(sample_audio).exists():
        res = evaluator.evaluate(sample_audio, prompt_text)
        print("\n--- INFERENCE RESULT SUMMARY ---")
        for k, v in res.items():
            print(f"{k:24}: {v}")
    else:
        print("[Speech Evaluator Module Ready]")
