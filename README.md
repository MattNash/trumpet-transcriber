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
| 3. Track the melody's pitch (torchcrepe) | `transcriber/pitch.py` | ✅ Phase 2 |
| 4. Turn pitch into notes and rhythm | `transcriber/notes.py` | ✅ Phase 2 |
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

**Using conda instead?** Replace the three commands above with these, and still use `pip` for the packages (don't `conda install pytorch` as well; mixing the two breaks GPU setups):

```bash
conda create -n trumpet python=3.12
conda activate trumpet
pip install -r requirements.txt
```

Check that PyTorch can see your NVIDIA GPU (should print `True`):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

```bash
python cli.py ~/Music/some_song.mp3
```

Results go into `output/<song name>/`:

| File | What it is |
|---|---|
| `melody.mid` | The detected melody at concert pitch. Open it in MuseScore and compare with the song. |
| `pitch_plot.png` | Grey dots = the pitch the model heard; blue bars = the notes it decided on. |
| `notes.csv` | Every note with its timing, as a table. |
| `audio.wav`, `stems/*.wav` | The extracted audio and separated instruments. |

The first run downloads the Demucs model (about 80 MB). Later runs on the same
song reuse the separated stems, so they only take a few seconds.

### Options

| Option | Use it when |
|---|---|
| `--stem other` | The melody is played by an instrument, not sung (also `bass`, `drums`). |
| `--confidence 0.7` | You get junk notes in quiet or noisy parts (default 0.5; lower it if real notes go missing). |
| `--grid 2` | The rhythms are too fussy; snaps to 8th notes instead of 16ths. |
| `--force` | Redo the audio extraction and separation. |

## Tests

```bash
pytest
```
