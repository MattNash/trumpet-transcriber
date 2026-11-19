"""Tools for checking the transcription by ear and by eye.

write_midi()  - a MIDI file of the detected melody, at concert pitch.
                Open it in MuseScore or any MIDI player and compare it
                with the original song.
plot_notes()  - a chart of the pitch curve (grey dots) with the detected
                notes drawn over it (blue bars). Where the two disagree
                is where the note detection needs tuning.
write_csv()   - the notes as a spreadsheet-friendly table.
"""

import csv
import math
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import mido

from transcriber.notes import Note
from transcriber.pitch import PitchTrack

TICKS_PER_BEAT = 480


def write_midi(notes: list[Note], tempo: float, path: Path) -> Path:
    """Save quantized notes as a one-track MIDI file."""
    midi_file = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))
    track.append(mido.Message("program_change", program=56))  # 56 = trumpet sound

    # MIDI stores the time *since the previous event*, not absolute times,
    # so we keep track of where the previous event was.
    events = []
    for note in notes:
        events.append((note.start_beat, "note_on", note.pitch))
        events.append((note.start_beat + note.beats, "note_off", note.pitch))
    events.sort(key=lambda e: (e[0], e[1] == "note_on"))  # note_offs first at the same time

    previous_tick = 0
    for beat, kind, pitch in events:
        tick = round(beat * TICKS_PER_BEAT)
        track.append(mido.Message(kind, note=pitch, velocity=90, time=tick - previous_tick))
        previous_tick = tick

    midi_file.save(path)
    return path


def write_csv(notes: list[Note], path: Path) -> Path:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["note", "midi", "start_sec", "end_sec", "start_beat", "beats"])
        for n in notes:
            writer.writerow([n.name, n.pitch, f"{n.start:.2f}", f"{n.end:.2f}", n.start_beat, n.beats])
    return path


def plot_notes(track: PitchTrack, notes: list[Note], path: Path, seconds_per_row: float = 20) -> Path:
    """Draw the pitch curve and notes, one row per 20 seconds of song."""
    duration = float(track.times[-1])
    rows = max(1, math.ceil(duration / seconds_per_row))
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.6 * rows), squeeze=False)
    midi = track.midi

    # Same vertical range on every row, labelled with note names.
    pitches = [n.pitch for n in notes] or [60]
    lowest, highest = min(pitches) - 2, max(pitches) + 2
    ticks = list(range(lowest, highest + 1))
    labels = [librosa.midi_to_note(p, unicode=False) for p in ticks]

    for row, ax in enumerate(axes[:, 0]):
        left, right = row * seconds_per_row, (row + 1) * seconds_per_row
        shown = (track.times >= left) & (track.times < right)
        ax.scatter(track.times[shown], midi[shown], s=1, color="0.6")
        for n in notes:
            if n.end > left and n.start < right:
                ax.hlines(n.pitch, n.start, n.end, color="tab:blue", linewidth=4)
        ax.set_xlim(left, right)
        ax.set_ylim(lowest - 0.5, highest + 0.5)
        ax.set_yticks(ticks, labels, fontsize=7)
        ax.set_ylabel("note (concert pitch)")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time (seconds)")

    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
