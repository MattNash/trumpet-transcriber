"""Run the transcription pipeline from the terminal.

Usage:
    python cli.py path/to/song.mkv
    python cli.py path/to/song.mp3 --stem other      # melody is an instrument
    python cli.py path/to/song.mp3 --force           # redo separation

Separation is slow, so if output/<song>/stems/ already exists it is
reused. That makes it quick to re-run while tuning the note settings.
"""

import argparse
import time
from pathlib import Path

from transcriber.extract_audio import extract_audio
from transcriber.notes import find_beats, quantize, split_into_notes
from transcriber.pitch import track_pitch
from transcriber.preview import plot_notes, write_csv, write_midi
from transcriber.separate import STEMS, separate


def main():
    parser = argparse.ArgumentParser(description="Transcribe a song's melody for B-flat trumpet.")
    parser.add_argument("input", type=Path, help="Any audio or video file")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Folder for results (default: ./output)")
    parser.add_argument("--stem", choices=STEMS, default="vocals",
                        help="Which separated track holds the melody (default: vocals)")
    parser.add_argument("--force", action="store_true",
                        help="Redo audio extraction and separation even if already done")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Pitch confidence needed to count as a note, 0-1 (default: 0.5)")
    parser.add_argument("--grid", type=int, choices=[2, 4], default=4,
                        help="Rhythm grid: 4 = 16th notes, 2 = 8th notes (default: 4)")
    args = parser.parse_args()

    song_dir = args.output / args.input.stem
    stem_dir = song_dir / "stems"
    wav_path = song_dir / "audio.wav"
    start = time.time()

    # Steps 1 and 2: skipped if a previous run already did them.
    if args.force or not (stem_dir / f"{args.stem}.wav").exists():
        print(f"[1/4] Extracting audio from {args.input.name}...")
        extract_audio(args.input, wav_path)
        print("[2/4] Separating instruments...")
        separate(wav_path, stem_dir)
    else:
        print("[1-2/4] Reusing audio and stems from a previous run (use --force to redo).")

    print(f"[3/4] Tracking the pitch of '{args.stem}'...")
    track = track_pitch(stem_dir / f"{args.stem}.wav", confidence_threshold=args.confidence)

    print("[4/4] Finding notes and rhythm...")
    notes = split_into_notes(track)
    tempo, beat_times = find_beats(wav_path)
    notes = quantize(notes, beat_times, subdivisions=args.grid)
    print(f"      Tempo about {tempo:.0f} BPM, {len(notes)} notes found.")

    midi_path = write_midi(notes, tempo, song_dir / "melody.mid")
    csv_path = write_csv(notes, song_dir / "notes.csv")
    plot_path = plot_notes(track, notes, song_dir / "pitch_plot.png")

    print("Results:")
    print(f"  {midi_path}   <- listen to this (concert pitch)")
    print(f"  {plot_path}   <- pitch curve vs detected notes")
    print(f"  {csv_path}")
    print(f"Done in {time.time() - start:.1f} seconds.")


if __name__ == "__main__":
    main()
