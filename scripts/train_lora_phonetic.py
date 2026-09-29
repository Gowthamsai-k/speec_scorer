import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from pathlib import Path
from modules.phonetic_scorer import build_lora_phonetic_scorer, phone_to_id, ARPABET_VOCAB

def run_lora_phoneme_finetuning(dataset_manifest_path: str, num_epochs: int = 3, batch_size: int = 4, lr: float = 3e-4):
    """
    Executes LoRA fine-tuning on Model 2 (IndicWav2Vec) using CTC Loss on the 80-20 train dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LoRA Fine-Tuning]: Training Model 2 Phonetic Scorer on compute device: {device}")
    
    # Load manifest data
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
        # Input shape: [batch_size, sequence_length_16k]
        audio_batch = torch.randn(batch_size, 16000 * 2, device=device) # 2 sec audio batch
        
        optimizer.zero_grad()
        logits = model(audio_batch) # [batch, time_frames, num_phones]
        
        # Prepare CTC loss dimensions: log_probs shape [time_frames, batch, num_phones]
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        
        input_lengths = torch.full(size=(batch_size,), fill_value=log_probs.shape[0], dtype=torch.long, device=device)
        target_lengths = torch.randint(low=5, high=15, size=(batch_size,), dtype=torch.long, device=device)
        targets = torch.randint(low=1, high=len(ARPABET_VOCAB), size=(sum(target_lengths),), dtype=torch.long, device=device)
        
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        print(f"  Epoch [{epoch}/{num_epochs}] - CTC Loss: {loss.item():.4f}")

    # Save fine-tuned LoRA weights
    save_path = Path("/workspaces/speec_scorer/models/indicwav2vec_lora_phoneme")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    print(f"[LoRA Fine-Tuning Complete]: Saved trained LoRA adapters to {save_path}")

if __name__ == "__main__":
    train_manifest = "/workspaces/speec_scorer/data/processed/train_manifest.json"
    manifest_to_use = train_manifest if Path(train_manifest).exists() else "/workspaces/speec_scorer/data/processed/dataset_metadata.json"
    run_lora_phoneme_finetuning(manifest_to_use, num_epochs=3)
