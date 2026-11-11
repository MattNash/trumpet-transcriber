# Trumpet Transcriber

Upload a song (audio or video) and get its melody written out for B♭ trumpet.

## How it works

```
any media file → audio → separate instruments → track the melody → notes → B♭ trumpet part → sheet music
```

| Step | File | Status |
|---|---|---|
| 1. Extract audio (FFmpeg) | `transcriber/extract_audio.py` | ✅ Phase 1 |
| 2. Separate instruments (Demucs) | `transcriber/separate.py` | ✅ Phase 1 |
| 3. Track the melody's pitch | `transcriber/pitch.py` | Phase 2 |
| 4. Turn pitch into notes and rhythm | `transcriber/notes.py` | Phase 2 |
| 5. Transpose for B♭ trumpet | `transcriber/trumpet.py` | Phase 3 |
| 6. Export MusicXML / PDF | `transcriber/export.py` | Phase 3 |
| Web app for the tablet | `web/` | Phase 4 |

## Setup (Ubuntu)

Install FFmpeg and Python's virtual environment support:

```bash
sudo apt install ffmpeg python3-venv
```

Create a virtual environment (an isolated folder for this project's Python packages) and install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check that PyTorch can see your NVIDIA GPU (should print `True`):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

```bash
source .venv/bin/activate
python cli.py ~/Music/some_song.mp3
```

Results go into `output/<song name>/`:

- `audio.wav`: the extracted audio
- `stems/vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`: the separated instruments

The first run downloads the Demucs model (about 80 MB).

## Tests

```bash
pytest
```
