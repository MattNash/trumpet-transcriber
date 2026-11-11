"""Check that extract_audio handles audio and video files.

We generate tiny test files with FFmpeg itself, so no real songs are needed.
Run with:  pytest
"""

import subprocess

import soundfile as sf

from transcriber.extract_audio import SAMPLE_RATE, extract_audio


def make_test_file(path, with_video=False):
    """Create a 1-second 440 Hz tone (optionally with a video track)."""
    command = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
    if with_video:
        command += ["-f", "lavfi", "-i", "color=black:size=64x64:duration=1"]
    command.append(str(path))
    subprocess.run(command, check=True)


def test_mp3_becomes_stereo_wav(tmp_path):
    source = tmp_path / "tone.mp3"
    make_test_file(source)
    wav = extract_audio(source, tmp_path / "out.wav")
    info = sf.info(wav)
    assert info.samplerate == SAMPLE_RATE
    assert info.channels == 2
    assert 0.9 < info.duration < 1.2


def test_video_file_keeps_only_audio(tmp_path):
    source = tmp_path / "clip.mkv"
    make_test_file(source, with_video=True)
    wav = extract_audio(source, tmp_path / "out.wav")
    assert sf.info(wav).channels == 2
