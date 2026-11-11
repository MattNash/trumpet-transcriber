"""Step 1: turn any audio or video file into a standard WAV file.

We hand the hard work to FFmpeg, which understands almost every media
format (MP3, WAV, FLAC, OGG, M4A, WebM, MKV, MP4, ...). For video files
FFmpeg simply ignores the picture and keeps the soundtrack.

The output is always:
  - WAV format (uncompressed, easy for Python to read)
  - 44,100 samples per second (what Demucs expects)
  - stereo (2 channels)
"""

import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 44100
CHANNELS = 2


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """Convert `input_path` (any media file) into a WAV at `output_path`.

    Returns the path to the new WAV file.
    Raises RuntimeError if FFmpeg is missing or can't read the file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed. On Ubuntu run: sudo apt install ffmpeg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The FFmpeg command, piece by piece:
    command = [
        "ffmpeg",
        "-y",                      # overwrite the output file if it exists
        "-loglevel", "error",      # only print real problems
        "-i", str(input_path),     # the input file
        "-vn",                     # "no video": drop any picture track
        "-ac", str(CHANNELS),      # number of audio channels
        "-ar", str(SAMPLE_RATE),   # sample rate
        "-c:a", "pcm_s16le",       # plain 16-bit WAV audio
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg could not read {input_path.name}:\n{result.stderr.strip()}")

    return output_path
