"""Step 3: follow the pitch of the melody, frame by frame.

We use torchcrepe, a neural network trained to answer one question for
every short slice of audio: "what note is being sung or played right now,
and how sure are you?" It works best on a single melody line, which is
why we run it on a separated stem (usually vocals) rather than the full mix.

The result is a "pitch track": for every 10 milliseconds of audio,
  - the frequency in Hz (or NaN, "not a number", when nothing is sounding)
  - a confidence value from 0 (noise) to 1 (clear, steady pitch)
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe

SAMPLE_RATE = 16000              # torchcrepe works at 16 kHz
HOP_LENGTH = 160                 # 160 samples at 16 kHz = one frame every 10 ms
LOWEST_HZ = 65.0                 # about C2: below any vocal melody
HIGHEST_HZ = 1400.0              # about F6: above any vocal melody


@dataclass
class PitchTrack:
    times: np.ndarray       # time of each frame, in seconds
    frequency: np.ndarray   # Hz, NaN where there is no clear pitch
    confidence: np.ndarray  # 0..1, how sure the model is

    @property
    def midi(self) -> np.ndarray:
        """Frequency as MIDI note numbers (60 = middle C, 69 = A 440 Hz).

        Unlike note names these are fractional: 60.3 is "slightly sharp C".
        """
        return librosa.hz_to_midi(self.frequency)


def track_pitch(
    stem_path: Path,
    confidence_threshold: float = 0.5,
    silence_db: float = -40.0,
    model: str = "full",
) -> PitchTrack:
    """Find the pitch of the melody in `stem_path` every 10 ms.

    confidence_threshold: frames the model is less sure about than this are
        treated as "no note". Raise it if you get junk notes, lower it if
        real notes go missing.
    silence_db: frames this much quieter than the loudest part of the track
        are treated as silence. (Separated stems often contain faint
        "bleed" from other instruments, which this removes.)
    model: "full" (accurate) or "tiny" (fast, for quick experiments).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # librosa loads the file, mixes it to mono and resamples it in one go.
    audio, _ = librosa.load(stem_path, sr=SAMPLE_RATE, mono=True)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # shape (1, samples)

    frequency, confidence = torchcrepe.predict(
        audio_tensor,
        SAMPLE_RATE,
        HOP_LENGTH,
        LOWEST_HZ,
        HIGHEST_HZ,
        model,
        return_periodicity=True,  # "periodicity" is torchcrepe's word for confidence
        batch_size=2048,
        device=device,
    )
    # Smooth out single-frame wobbles in the confidence values.
    confidence = torchcrepe.filter.median(confidence, 3)

    frequency = frequency.squeeze(0).cpu().numpy()
    confidence = confidence.squeeze(0).cpu().numpy()

    # How loud is each frame, in decibels relative to the loudest frame?
    loudness = librosa.amplitude_to_db(
        librosa.feature.rms(y=audio, frame_length=1024, hop_length=HOP_LENGTH)[0],
        ref=np.max,
    )
    # torchcrepe and librosa can disagree by a frame on the length; trim to match.
    n = min(len(frequency), len(loudness))
    frequency, confidence, loudness = frequency[:n], confidence[:n], loudness[:n]

    # Blank out frames that are unsure or too quiet.
    no_note = (confidence < confidence_threshold) | (loudness < silence_db)
    frequency = np.where(no_note, np.nan, frequency)

    times = np.arange(n) * HOP_LENGTH / SAMPLE_RATE
    return PitchTrack(times=times, frequency=frequency, confidence=confidence)
