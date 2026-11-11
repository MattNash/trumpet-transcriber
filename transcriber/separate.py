"""Step 2: split a song into separate instrument tracks ("stems").

We use Demucs, a neural network from Meta trained to pull a mixed song
apart into four stems:

  vocals  - the singer(s)
  drums   - the drum kit
  bass    - the bass guitar / bass line
  other   - everything else (guitar, piano, synths, horns, ...)

For most songs the melody is in "vocals", so that is what later steps
will transcribe by default. For instrumentals, "other" is usually right.

The first run downloads the model (about 80 MB) and caches it.
"""

from pathlib import Path

import soundfile as sf
import torch
from demucs.api import Separator

STEMS = ("vocals", "drums", "bass", "other")


def pick_device() -> str:
    """Use the NVIDIA GPU if PyTorch can see one, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def separate(wav_path: Path, output_dir: Path, model: str = "htdemucs") -> dict[str, Path]:
    """Split the WAV at `wav_path` into stems, saved as WAVs in `output_dir`.

    Returns a dictionary like {"vocals": Path(".../vocals.wav"), ...}.
    """
    wav_path = Path(wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Separating on {device.upper()} with model '{model}'...")

    # Read the audio. soundfile gives us an array shaped (samples, channels);
    # Demucs wants a PyTorch tensor shaped (channels, samples), hence the .T
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    audio_tensor = torch.from_numpy(audio.T.copy())

    # Load the model and run it. `progress=True` prints a progress bar.
    separator = Separator(model=model, device=device, progress=True)
    _original, stems = separator.separate_tensor(audio_tensor, sample_rate)

    # Save each stem. Turn the tensor back into (samples, channels) for soundfile.
    stem_paths = {}
    for name, stem_audio in stems.items():
        path = output_dir / f"{name}.wav"
        sf.write(path, stem_audio.cpu().numpy().T, separator.samplerate)
        stem_paths[name] = path

    return stem_paths
