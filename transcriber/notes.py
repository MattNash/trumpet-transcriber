"""Step 4: turn a wobbly pitch curve into notes with a rhythm.

This happens in three parts:

1. split_into_notes(): decide where one note ends and the next begins.
   Singers slide between notes and use vibrato, so we only start a new
   note when the pitch moves away from the current note and *stays* away
   for a few frames.

2. find_beats(): find the song's beat (where you'd tap your foot) using
   librosa's beat tracker on the full mix.

3. quantize(): measure each note in beats and snap it to the nearest
   16th note, so 0.97 beats becomes a quarter note and 0.52 an eighth.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import medfilt

from transcriber.pitch import PitchTrack


@dataclass
class Note:
    pitch: int              # MIDI note number at concert pitch (60 = middle C)
    start: float            # seconds
    end: float              # seconds
    start_beat: float = 0.0  # filled in by quantize()
    beats: float = 0.0       # length in beats, filled in by quantize()

    @property
    def name(self) -> str:
        """Note name like 'C#4'."""
        return librosa.midi_to_note(self.pitch, unicode=False)


# ---------------------------------------------------------------------------
# 1. Pitch curve -> notes
# ---------------------------------------------------------------------------

def split_into_notes(
    track: PitchTrack,
    change_threshold: float = 0.75,
    frames_to_confirm: int = 4,
    min_duration: float = 0.08,
) -> list[Note]:
    """Group the frames of a pitch track into notes.

    change_threshold: how far (in semitones) the pitch must move from the
        current note to count as a new note. 0.75 ignores normal vibrato.
    frames_to_confirm: how many frames in a row (10 ms each) the pitch must
        stay at the new level before we believe it. Stops quick slides
        and glitches from becoming notes.
    min_duration: notes shorter than this (seconds) are thrown away.
    """
    midi = track.midi.copy()

    # A median filter replaces each value with the middle value of its
    # neighbours, which removes single-frame spikes. NaN (silence) frames
    # are left alone.
    voiced = ~np.isnan(midi)
    midi[voiced] = medfilt(midi[voiced], kernel_size=5)

    segments = []          # (first_frame, frame_after_last) for each note
    start = None           # first frame of the note we're currently in
    away = 0               # how many frames in a row we've been "away" from it

    for i, value in enumerate(midi):
        if np.isnan(value):
            # Silence ends the current note.
            if start is not None:
                segments.append((start, i))
                start = None
            away = 0
            continue

        if start is None:
            start, away = i, 0
            continue

        # The current note's pitch: the median of its last 10 frames,
        # not counting the frames that have already wandered away.
        settled = midi[max(start, i - away - 10): i - away]
        current_pitch = np.median(settled)

        away = away + 1 if abs(value - current_pitch) > change_threshold else 0

        if away >= frames_to_confirm:
            # The pitch has moved and stayed moved: the new note began
            # at the first frame that wandered away.
            new_start = i - away + 1
            segments.append((start, new_start))
            start, away = new_start, 0

    if start is not None:
        segments.append((start, len(midi)))

    hop = track.times[1] - track.times[0]
    notes = []
    for first, after_last in segments:
        begin, end = track.times[first], track.times[first] + (after_last - first) * hop
        if end - begin < min_duration:
            continue
        pitch = int(round(float(np.median(midi[first:after_last]))))
        notes.append(Note(pitch=pitch, start=float(begin), end=float(end)))

    return merge_repeated(notes)


def merge_repeated(notes: list[Note], max_gap: float = 0.03) -> list[Note]:
    """Join notes of the same pitch separated by only a tiny gap.

    These are usually one note that got briefly interrupted (a breath of
    noise, a consonant), not two separate notes.
    """
    merged: list[Note] = []
    for note in notes:
        if merged and merged[-1].pitch == note.pitch and note.start - merged[-1].end <= max_gap:
            merged[-1].end = note.end
        else:
            merged.append(note)
    return merged


# ---------------------------------------------------------------------------
# 2. Finding the beat
# ---------------------------------------------------------------------------

def find_beats(mix_path: Path) -> tuple[float, np.ndarray]:
    """Return (tempo in beats per minute, times of each beat in seconds).

    Uses the full mix, because drums and bass make the beat easiest to hear.
    """
    audio, sr = librosa.load(mix_path, sr=22050, mono=True)
    rough_tempo, beat_times = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    beat_times = fill_beats(np.asarray(beat_times), len(audio) / sr, float(np.atleast_1d(rough_tempo)[0]))
    # librosa's tempo number is a rough first guess; the spacing of the beats
    # it actually found is more accurate.
    tempo = 60 / average_beat_length(beat_times)
    return tempo, beat_times


def average_beat_length(beat_times: np.ndarray) -> float:
    """Seconds per beat, averaged over all the beats.

    The beat tracker measures time in ~23 ms steps, so individual gaps
    between beats jitter a little; the average over many beats doesn't.
    """
    return float((beat_times[-1] - beat_times[0]) / (len(beat_times) - 1))


def fill_beats(beat_times: np.ndarray, duration: float, tempo: float) -> np.ndarray:
    """Extend the beat grid to cover the whole song.

    The beat tracker often misses the quiet start and end of a song.
    We add evenly spaced beats before the first and after the last
    detected beat so every note can be measured in beats.
    """
    if len(beat_times) < 2:
        return np.arange(0.0, duration + 60 / tempo, 60 / tempo)
    period = average_beat_length(beat_times)
    # Stop half a beat before time 0, so we don't add an empty beat before the song.
    before = np.arange(beat_times[0] - period, -period / 2, -period)[::-1]
    after = np.arange(beat_times[-1] + period, duration + period, period)
    return np.concatenate([before, beat_times, after])


# ---------------------------------------------------------------------------
# 3. Snapping notes to the rhythm grid
# ---------------------------------------------------------------------------

def quantize(notes: list[Note], beat_times: np.ndarray, subdivisions: int = 4) -> list[Note]:
    """Fill in start_beat and beats for each note, snapped to the grid.

    subdivisions: grid steps per beat. 4 = 16th notes, 2 = 8th notes
        (simpler to read, less accurate).

    Beat 0 is the first beat in the grid, which isn't necessarily the
    first beat of a bar. Phase 3 works out where the bars go.
    """
    step = 1 / subdivisions
    beat_numbers = np.arange(len(beat_times), dtype=float)

    def to_grid(seconds: float) -> float:
        # Convert a time to a (fractional) beat number, then round to the grid.
        beat = np.interp(seconds, beat_times, beat_numbers)
        return round(beat * subdivisions) / subdivisions

    result: list[Note] = []
    for note in notes:
        start = to_grid(note.start)
        end = max(to_grid(note.end), start + step)  # every note lasts at least one step

        if result and start < result[-1].start_beat + result[-1].beats:
            previous = result[-1]
            if start <= previous.start_beat:
                # Both notes snapped to the same spot: keep the longer one.
                if note.end - note.start > previous.end - previous.start:
                    result[-1] = Note(note.pitch, note.start, note.end, start, end - start)
                continue
            # Overlap: shorten the previous note so it ends where this one starts.
            previous.beats = start - previous.start_beat

        result.append(Note(note.pitch, note.start, note.end, start, end - start))

    return result
