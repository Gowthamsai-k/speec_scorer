import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoConfig, Wav2Vec2Model, Wav2Vec2PreTrainedModel
from peft import LoraConfig, get_peft_model
from modules.config import MODELS_DIR

ARPABET_VOCAB = [
    "<pad>", "<s>", "</s>", "<unk>", "|",
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", 
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", 
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", 
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
]
phone_to_id = {p: i for i, p in enumerate(ARPABET_VOCAB)}
id_to_phone = {i: p for i, p in enumerate(ARPABET_VOCAB)}

INDIAN_ALLOPHONE_MAP = {
    "T":  ["T", "D"],
    "D":  ["D", "T"],
    "TH": ["TH", "T"],
    "DH": ["DH", "D"],
    "W":  ["W", "V"],
    "V":  ["V", "W"],
    "IY": ["IY", "IH"],
    "IH": ["IH", "IY"],
    "R":  ["R"]
}

class IndicWav2VecPhonemeCTC(Wav2Vec2PreTrainedModel):
    _tied_weights_keys = []
    _keys_to_ignore_on_load_missing = [r"phoneme_head"]

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.phoneme_head = nn.Linear(config.hidden_size, len(ARPABET_VOCAB))
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.phoneme_head(hidden_states)
        return logits

def calculate_aligned_gop(
    logits: torch.Tensor, 
    aligned_phones: List[str], 
    gamma: float = 1.8
) -> List[Dict]:
    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    T_frames = log_probs.shape[0]
    
    results = []
    frames_per_phone = max(1, T_frames // max(len(aligned_phones), 1))
    
    for i, target_phone in enumerate(aligned_phones):
        target_phone_clean = target_phone.upper().strip("012")
        t_start = i * frames_per_phone
        t_end = min((i + 1) * frames_per_phone, T_frames)
        
        if t_end <= t_start:
            t_end = t_start + 1
            
        segment_probs = log_probs[t_start:t_end]
        
        allophone_set = INDIAN_ALLOPHONE_MAP.get(target_phone_clean, [target_phone_clean])
        allophone_ids = [phone_to_id[p] for p in allophone_set if p in phone_to_id]
        
        if not allophone_ids:
            allophone_ids = [phone_to_id.get(target_phone_clean, 0)]
            
        non_allophone_ids = [idx for idx in range(len(ARPABET_VOCAB)) if idx not in allophone_ids]
        
        max_allo_logprobs = np.max(segment_probs[:, allophone_ids], axis=-1)
        max_compete_logprobs = np.max(segment_probs[:, non_allophone_ids], axis=-1)
        
        raw_gop = float(np.mean(max_allo_logprobs - max_compete_logprobs))
        score_100 = 100.0 / (1.0 + np.exp(-gamma * raw_gop))
        
        if score_100 >= 75.0:
            status = "GREEN"
        elif score_100 >= 50.0:
            status = "YELLOW"
        else:
            status = "RED"

        results.append({
            "token": target_phone,
            "canonical_target": target_phone_clean,
            "matched_allophones": allophone_set,
            "raw_gop": round(raw_gop, 3),
            "gop_score": round(score_100, 1),
            "status": status
        })
        
    return results

def build_lora_phonetic_scorer(model_path_or_name: str = "ai4bharat/indicwav2vec-hindi", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    local_dir = MODELS_DIR / "indicwav2vec-hindi"
    load_target = str(local_dir) if local_dir.exists() else model_path_or_name

    print(f"[Phonetic Scorer]: Initializing Wav2Vec model from {load_target}...")
    try:
        try:
            cfg = AutoConfig.from_pretrained(load_target)
        except Exception:
            print(f"[Phonetic Scorer]: Fallback to open baseline 'facebook/wav2vec2-base-960h'...")
            load_target = "facebook/wav2vec2-base-960h"
            cfg = AutoConfig.from_pretrained(load_target)
            
        cfg.vocab_size = len(ARPABET_VOCAB)
        indic_w2v = IndicWav2VecPhonemeCTC.from_pretrained(load_target, config=cfg, ignore_mismatched_sizes=True)

        indic_w2v.wav2vec2.feature_extractor._freeze_parameters()
        for layer in indic_w2v.wav2vec2.encoder.layers[:8]:
            for param in layer.parameters():
                param.requires_grad = False

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        phonetic_scorer_model = get_peft_model(indic_w2v, peft_config).to(device)
        phonetic_scorer_model.print_trainable_parameters()
        return phonetic_scorer_model
    except Exception as e:
        print(f"[Phonetic Scorer Warning]: Model fallback: {e}")
        return None

if __name__ == "__main__":
    model = build_lora_phonetic_scorer()
    print("[Phonetic Scorer Module Ready]")
