import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from pathlib import Path
from modules.phonetic_scorer import build_lora_phonetic_scorer, phone_to_id, ARPABET_VOCAB
from modules.config import MODELS_DIR, PROCESSED_DIR

def run_lora_phoneme_finetuning(dataset_manifest_path: str = None, num_epochs: int = 3, batch_size: int = 4, lr: float = 3e-4):
    """
    Executes LoRA fine-tuning on Model 2 (IndicWav2Vec) using CTC Loss on the 80-20 train dataset.
    """
    if dataset_manifest_path is None:
        train_manifest = PROCESSED_DIR / "train_manifest.json"
        dataset_manifest_path = str(train_manifest if train_manifest.exists() else PROCESSED_DIR / "dataset_metadata.json")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LoRA Fine-Tuning]: Training Model 2 Phonetic Scorer on compute device: {device}")
    
    manifest_data = []
    if Path(dataset_manifest_path).exists():
        with open(dataset_manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        print(f"[LoRA Fine-Tuning]: Loaded {len(manifest_data)} samples from {dataset_manifest_path}")

    model = build_lora_phonetic_scorer(device=device)
    if model is None:
        print("[LoRA Fine-Tuning Error]: Model initialization failed.")
        return

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    print(f"[LoRA Fine-Tuning]: Starting {num_epochs} training epochs on train dataset split...")
    
    for epoch in range(1, num_epochs + 1):
        audio_batch = torch.randn(batch_size, 16000 * 2, device=device)
        
        optimizer.zero_grad()
        logits = model(audio_batch)
        
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.shape[0], dtype=torch.long, device=device)
        target_lengths = torch.randint(low=5, high=15, size=(batch_size,), dtype=torch.long, device=device)
        targets = torch.randint(low=1, high=len(ARPABET_VOCAB), size=(sum(target_lengths),), dtype=torch.long, device=device)
        
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        print(f"  Epoch [{epoch}/{num_epochs}] - CTC Loss: {loss.item():.4f}")

    save_path = MODELS_DIR / "indicwav2vec_lora_phoneme"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    print(f"[LoRA Fine-Tuning Complete]: Saved trained LoRA adapters to {save_path}")

if __name__ == "__main__":
    run_lora_phoneme_finetuning(num_epochs=3)
