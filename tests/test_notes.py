"""Check that a known melody comes back out as the right notes.

We synthesize a short tune (with vibrato and a slide, like a singer), run
it through pitch tracking and note detection, and compare.
"""

import numpy as np
import soundfile as sf

from transcriber.notes import fill_beats, quantize, split_into_notes
from transcriber.pitch import track_pitch

# (MIDI pitch, length in beats) at 120 BPM: C4 D4 E4 G4(half) - rest - E4
TUNE = [(60, 1), (62, 1), (64, 1), (67, 2), (None, 1), (64, 1)]
BPM = 120
SR = 16000


def synthesize(path):
    """A voice-like tone: a few harmonics, 5 Hz vibrato, quick slides between notes."""
    seconds_per_beat = 60 / BPM
    pitch_curve = []
    for pitch, beats in TUNE:
        n = int(beats * seconds_per_beat * SR)
        pitch_curve.append(np.full(n, np.nan if pitch is None else float(pitch)))
    midi = np.concatenate(pitch_curve)

    t = np.arange(len(midi)) / SR
    voiced = ~np.isnan(midi)
    midi_filled = np.where(voiced, midi, 60.0)
    # 40 ms slide: smooth the steps between notes
    kernel = np.ones(int(0.04 * SR)) / int(0.04 * SR)
    midi_smooth = np.convolve(midi_filled, kernel, mode="same")
    midi_smooth += 0.3 * np.sin(2 * np.pi * 5 * t)          # vibrato, +/- 0.3 semitone
    freq = 440 * 2 ** ((midi_smooth - 69) / 12)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    audio = sum(np.sin(k * phase) / k for k in (1, 2, 3)) * 0.3 * voiced
    sf.write(path, audio.astype(np.float32), SR)


def test_tune_is_recovered(tmp_path):
    path = tmp_path / "tune.wav"
    synthesize(path)

    track = track_pitch(path, model="tiny")
    notes = split_into_notes(track)
    beat_times = fill_beats(np.array([0.0, 0.5, 1.0, 1.5]), 4.0, BPM)
    notes = quantize(notes, beat_times)

    assert [n.pitch for n in notes] == [60, 62, 64, 67, 64]
    assert [n.start_beat for n in notes] == [0, 1, 2, 3, 6]
    assert [n.beats for n in notes] == [1, 1, 1, 2, 1]
