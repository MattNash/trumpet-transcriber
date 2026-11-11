"""Run the transcription pipeline from the terminal.

Usage:
    python cli.py path/to/song.mkv
    python cli.py path/to/song.mp3 --output my_results

Right now (Phase 1) this extracts the audio and separates it into stems.
Later phases will add the pitch tracking and sheet music steps.
"""

import argparse
import time
from pathlib import Path

from transcriber.extract_audio import extract_audio
from transcriber.separate import separate


def main():
    parser = argparse.ArgumentParser(description="Transcribe a song's melody for B-flat trumpet.")
    parser.add_argument("input", type=Path, help="Any audio or video file")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Folder for results (default: ./output)")
    args = parser.parse_args()

    # Each song gets its own subfolder, named after the file: output/<song name>/
    song_dir = args.output / args.input.stem
    start = time.time()

    print(f"[1/2] Extracting audio from {args.input.name}...")
    wav_path = extract_audio(args.input, song_dir / "audio.wav")
    print(f"      Saved {wav_path}")

    print("[2/2] Separating instruments...")
    stems = separate(wav_path, song_dir / "stems")
    for name, path in stems.items():
        print(f"      {name:<7} -> {path}")

    print(f"Done in {time.time() - start:.1f} seconds.")


if __name__ == "__main__":
    main()
