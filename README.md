# tutorial-narrator

Turn screen recordings into narrated tutorials, fully offline, with Kokoro TTS.

## What it does

You record your screen and write a short script with a timestamp on each line. `narrate.py` reads every line
aloud with [Kokoro-82M](https://github.com/hexgrad/kokoro), puts each clip at its timestamp, and writes a copy
of your video with the narration as its audio track. Nothing is uploaded. Once the model is downloaded, it
runs fully offline.

```text
00:00 | Let's upgrade a MikroTik router from Router O S six to version seven.
00:09 | First, check your router. Go to System, then Resources.
00:25 | Now open your browser and search for MikroTik download.
```

```bash
python narrate.py script.txt upgrade-router-os.mp4 --voice am_michael
# -> upgrade-router-os-narrated.mp4  (and upgrade-router-os-narration.wav)
```

## Features

- **Timestamp-synced segments.** Each line is generated separately and placed at its `MM:SS` mark.
- **Overflow warnings.** If a line's speech runs into the next line or past the end of the video, you get a
  warning saying how many seconds it overflows. Overflow is measured to the end of the speech, so Kokoro's
  trailing silence doesn't count. Lines are never sped up to make them fit.
- **No re-encoding.** The video stream is copied as-is. Any existing audio is replaced by the narration (AAC).
- **Voice blending.** Mix voices by weight, e.g. `am_michael:60,bm_george:40`.
- **Apple Silicon acceleration.** Runs on the GPU through MPS when available, with CPU and CUDA as options.
- **Quiet by default.** A normal run prints only progress and results. `--verbose` shows library warnings,
  logging and ffmpeg output.

## Requirements

- macOS (Apple Silicon recommended) or Linux
- Python 3.10–3.12
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) (Kokoro uses it for words it doesn't know)
- [ffmpeg](https://ffmpeg.org/) (includes `ffprobe`)

On newer macOS versions, Homebrew's Python can fail with
`Symbol not found: _XML_SetAllocTrackerActivationThreshold` (its `pyexpat` needs a newer system libexpat).
Fix: build Python with [pyenv](https://github.com/pyenv/pyenv), which bundles its own expat. See
[Troubleshooting](#troubleshooting).

## Install

```bash
git clone https://github.com/p8ul/tutorial-narrator.git
cd tutorial-narrator

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

brew install espeak-ng ffmpeg        # Linux: sudo apt install espeak-ng ffmpeg
```

If `ffmpeg` is missing on macOS, `narrate.py` runs `brew install ffmpeg` for you.

The first run downloads the model and voices (about 330 MB) into `~/.cache/huggingface`, which takes a few
minutes. After that, everything runs offline.

## Usage

### `speak.py`: quick TTS

```bash
.venv/bin/python speak.py "Hello there"                      # writes ./output.wav
.venv/bin/python speak.py "Hello there" --play               # write, then play (macOS afplay)
.venv/bin/python speak.py -f input.txt --out story.wav       # read text from a file
echo "From a pipe" | .venv/bin/python speak.py -f -          # read from stdin
.venv/bin/python speak.py "Cheerio" --voice bm_george --speed 1.1
.venv/bin/python speak.py "Blend" --voice "am_michael:60,bm_george:40"
```

| Option | Default | Notes |
| --- | --- | --- |
| `--voice` | `am_michael` | Voice name or weighted blend |
| `--speed` | `1.0` | Higher is faster |
| `--out` | `output.wav` | Relative to the current directory |
| `-f FILE` | | Read text from a file, `-` for stdin. Each line break starts a new chunk |
| `--play` | | Play the result with `afplay` (macOS only) |
| `--device` | `auto` | `auto`, `mps`, `cpu` or `cuda` |
| `--verbose` | | Show all library warnings and logging |

### `narrate.py`: the full workflow

```bash
.venv/bin/python narrate.py script.txt video.mp4 --voice am_michael
```

It writes two files next to the video:

- `<video>-narrated.mp4`: the original video stream with the narration as its only audio
- `<video>-narration.wav`: the narration alone (24 kHz mono)

`--voice`, `--speed`, `--device` and `--verbose` work the same as in `speak.py`.

The output lists each segment's start time and speech length, then an overflow summary:

```text
video upgrade-router-os.mp4: 01:41.50 (101.50s), 7 segment(s)
  00:00.00   8.56s  line 1  OVERFLOW +0.56s
  00:08.00  15.52s  line 2  OVERFLOW +0.52s
  ...
3 segment(s) overflow (not sped up: they overlap the next line, or get cut off at the end of the video):
```

### Shell alias (optional)

To run `speak` from any directory without activating the venv, add this to `~/.zshrc` (or `~/.bashrc`):

```bash
alias speak='~/tutorial-narrator/.venv/bin/python ~/tutorial-narrator/speak.py'
```

Change the paths to where you cloned the repo. On macOS this hides espeak-ng's own `speak` command. Run
`command speak` or `espeak-ng` if you need it.

## Writing a good script

- **Format.** One segment per line: `MM:SS | text`. `H:MM:SS` and fractional seconds (`00:12.5`) also work.
  Blank lines and lines starting with `#` are ignored. Timestamps must go up.
- **Budget about 2.4 words per second** at `--speed 1.0`. A 10-second gap fits about 24 words.
- **Spell out acronyms and versions the way you want them said.** Write "Router O S", "M M I P S", "six point
  four nine". Kokoro reads "RouterOS" and "6.49" unpredictably.
- **Leave about 0.5 s between lines** so the narration has room to breathe.
- **When a line overflows, move timestamps.** Shift the next line later or cut words. Speeding everything
  up changes the pace of the whole video.
- **Fill long silent stretches** (uploads, reboots) with a short line like "This takes a few minutes."

## Voices

Kokoro has many voices. For narration, start with `am_michael` (American) or `bm_george` (British).

| Voice | Accent | | Voice | Accent |
| --- | --- | --- | --- | --- |
| `am_michael` | American | | `bm_george` | British |
| `am_fenrir` | American | | `bm_fable` | British |
| `am_puck` | American | | `bm_lewis` | British |
| `am_adam` | American | | `bm_daniel` | British |
| `am_echo` | American | | | |
| `am_eric` | American | | | |
| `am_liam` | American | | | |
| `am_onyx` | American | | | |

The language comes from the voice prefix: `a` is American English, `b` is British. In a blend, the voice with
the highest weight decides it.

There are also female voices (`af_*`, `bf_*`, e.g. `af_bella`, `bf_emma`) and voices for other languages. See
the full list with quality grades in
[VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).

To compare voices, write the same sentence with all 12 male English voices into `samples/`:

```bash
./samples.sh                        # default sentence
./samples.sh "Your own sentence."
afplay samples/bm_george.wav
```

## Recommended folder layout

Keep each tutorial in its own folder under `work/<video-name>/` inside the repo:

```text
tutorial-narrator/
└── work/
    ├── routeros-upgrade/
    │   ├── routeros-upgrade-narration.txt
    │   ├── upgrade-router-os.mp4
    │   ├── upgrade-router-os-narrated.mp4      # generated
    │   └── upgrade-router-os-narration.wav     # generated
    └── routeros-downgrade/
        └── ...
```

Run from the repo root. The outputs are written next to the video, in the same folder:

```bash
.venv/bin/python narrate.py work/routeros-upgrade/routeros-upgrade-narration.txt \
  work/routeros-upgrade/upgrade-router-os.mp4 --voice am_michael
```

> **Note:** `work/` is in `.gitignore`, so nothing in it is committed. Your scripts and videos there
> are **not backed up by git**. Keep a copy elsewhere if you need one.

## Examples

[`examples/`](examples/) has two real scripts for MikroTik RouterOS tutorials:

- [`routeros-upgrade-narration.txt`](examples/routeros-upgrade-narration.txt): upgrading from v6 to v7 (7 lines)
- [`routeros-downgrade-narration.txt`](examples/routeros-downgrade-narration.txt): going back from v7 to v6
  (11 lines)

The videos aren't included. Use the scripts as templates for your own.

## Troubleshooting

**`ModuleNotFoundError: No module named 'numpy'` (or `kokoro`)**
A conda environment is shadowing the venv, so `python` is conda's. Call the venv's Python directly
(`.venv/bin/python narrate.py ...`) or run `conda deactivate`. To stop conda's base env from activating in
every new shell: `conda config --set auto_activate_base false`.

**`Symbol not found: _XML_SetAllocTrackerActivationThreshold` when creating the venv**
Homebrew's Python can't load the macOS system libexpat. Build Python with pyenv and use it for the venv:

```bash
brew install pyenv
pyenv install 3.12
~/.pyenv/versions/3.12.*/bin/python3.12 -m venv .venv
```

**Warnings about LSTM dropout, `weight_norm`, `jit.script`, or "Redirects are currently not supported"**
These come from torch and are harmless. They're hidden by default and only show with `--verbose`.

**The first run is slow**
It downloads the model and voices (about 330 MB, roughly 3 minutes). Later runs start in seconds and work
offline.

**MPS vs CPU**
`--device auto` uses Apple's GPU (MPS) when it can, then CUDA, then CPU. If you hit an MPS error, try
`--device cpu`. It's slower, but the output is the same.

## Credits

- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): the TTS model (Apache-2.0)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng): phonemizer fallback
- [FFmpeg](https://ffmpeg.org/): muxing and probing


