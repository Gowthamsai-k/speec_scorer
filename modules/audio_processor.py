import os
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from modules.config import DATA_DIR, STANDARDIZED_AUDIO_DIR

def standardize_audio(input_file: str, output_file: str) -> bool:
    """
    Downmixes multichannel audio to mono, resamples to 16 kHz,
    and applies peak amplitude normalization to -0.95 dBFS.
    """
    try:
        try:
            data, sample_rate = sf.read(input_file)
            waveform = torch.from_numpy(data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
        except Exception:
            waveform, sample_rate = torchaudio.load(input_file)

        # 1. Downmix channels to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Resample to 16,000 Hz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # 3. Peak normalization to avoid clipping during feature extraction
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            waveform = (waveform / peak) * 0.95

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), waveform.squeeze(0).cpu().numpy(), 16000, subtype="PCM_16")
        return True
    except Exception as err:
        print(f"[Audio Processing Failed] {input_file}: {err}")
        return False

def process_directory(raw_dir: str = None, proc_dir: str = None):
    raw_path = Path(raw_dir) if raw_dir else DATA_DIR / "raw"
    target_dir = Path(proc_dir) if proc_dir else STANDARDIZED_AUDIO_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(raw_path.glob("**/*.wav")) + list(raw_path.glob("**/*.WAV")) + list(raw_path.glob("**/*.flac"))
    print(f"Normalizing {len(audio_files)} files to 16 kHz Mono...")
    processed_count = 0
    for file in audio_files:
        target_path = target_dir / f"{file.stem}.wav"
        if standardize_audio(str(file), str(target_path)):
            processed_count += 1
    print(f"Audio normalization stage complete. Successfully processed {processed_count}/{len(audio_files)} files.")
    return processed_count

if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "raw")
    proc = sys.argv[2] if len(sys.argv) > 2 else str(STANDARDIZED_AUDIO_DIR)
    process_directory(raw, proc)
