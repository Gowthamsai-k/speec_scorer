import os
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoConfig, Wav2Vec2Model, Wav2Vec2PreTrainedModel
from peft import LoraConfig, get_peft_model

ARPABET_VOCAB = [
    "<pad>", "<s>", "</s>", "<unk>", "|",
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", 
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", 
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", 
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
]
phone_to_id = {p: i for i, p in enumerate(ARPABET_VOCAB)}
id_to_phone = {i: p for i, p in enumerate(ARPABET_VOCAB)}

class IndicWav2VecPhonemeCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.phoneme_head = nn.Linear(config.hidden_size, len(ARPABET_VOCAB))
        self.init_weights()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.phoneme_head(hidden_states)
        return logits

def build_lora_phonetic_scorer(model_path_or_name: str = "ai4bharat/indicwav2vec-hindi", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    local_dir = Path("/workspaces/speec_scorer/models/indicwav2vec-hindi")
    load_target = str(local_dir) if local_dir.exists() else model_path_or_name

    print(f"[Phonetic Scorer]: Initializing IndicWav2Vec model from {load_target}...")
    try:
        cfg = AutoConfig.from_pretrained(load_target)
        cfg.vocab_size = len(ARPABET_VOCAB)
        indic_w2v = IndicWav2VecPhonemeCTC.from_pretrained(load_target, config=cfg, ignore_mismatched_sizes=True)

        # Freeze lower feature extraction layers
        indic_w2v.wav2vec2.feature_extractor._freeze_parameters()
        for layer in indic_w2v.wav2vec2.encoder.layers[:8]:
            for param in layer.parameters():
                param.requires_grad = False

        # Attach LoRA adapters to attention projection layers
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
        print(f"[Phonetic Scorer Warning]: Could not load full model weights: {e}")
        return None

if __name__ == "__main__":
    model = build_lora_phonetic_scorer()
    print("[Phonetic Scorer Module Ready]")
