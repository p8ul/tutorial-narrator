#!/usr/bin/env python3
"""Kokoro TTS command-line wrapper.

Examples:
    python speak.py "Hello there"
    python speak.py -f input.txt --voice bm_george --speed 1.1 --out story.wav
    python speak.py "Blended voice" --voice "am_michael:60,bm_george:40" --play
    python speak.py "Debug me" --verbose

Quiet by default: library warnings and logging are hidden, so a normal run prints
only the "wrote ..." line. Real errors still print. --verbose turns it all back on.
"""
import argparse
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

# Must be set before torch is imported: lets ops that MPS lacks fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SAMPLE_RATE = 24000
REPO_ID = "hexgrad/Kokoro-82M"
MODEL_FILES = ("config.json", "kokoro-v1_0.pth")


def parse_voice_spec(spec: str) -> list[tuple[str, float]]:
    """Parse 'am_michael:60,bm_george:40' into [(name, weight), ...].

    A voice without ':weight' gets weight 1.
    """
    voices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition(":")
        name = name.strip()
        try:
            w = float(weight) if weight else 1.0
        except ValueError:
            sys.exit(f"error: bad weight in voice spec '{part}'")
        if w <= 0:
            sys.exit(f"error: weight must be positive in voice spec '{part}'")
        voices.append((name, w))
    if not voices:
        sys.exit("error: empty --voice")
    return voices


def hf_hub_cache() -> Path:
    """Same lookup order huggingface_hub uses for its cache directory."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME", "~/.cache")
    return Path(xdg).expanduser() / "huggingface" / "hub"


def all_cached(voice_names: list[str]) -> bool:
    """True if the model and every requested voice are already in the HF cache."""
    snapshots = hf_hub_cache() / f"models--{REPO_ID.replace('/', '--')}" / "snapshots"
    needed = list(MODEL_FILES) + [f"voices/{v}.pt" for v in voice_names if not v.endswith(".pt")]
    try:
        return any(all((snap / f).exists() for f in needed) for snap in snapshots.iterdir())
    except OSError:
        return False


def quiet_libraries(voice_names: list[str]) -> None:
    """Hide startup noise. Must run before torch / kokoro / huggingface_hub are imported."""
    # torch/distributed/elastic/multiprocessing/redirects.py logs
    # "NOTE: Redirects are currently not supported in MacOs." at WARNING when
    # torch is imported. It's a plain stdlib logger, so raising the level on it
    # (and its parent) before the import stops it.
    for name in ("torch.distributed.elastic",
                 "torch.distributed.elastic.multiprocessing.redirects"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # Known harmless warnings from Kokoro's model code.
    warnings.filterwarnings("ignore", message=r"dropout option adds dropout after all but last recurrent layer")
    warnings.filterwarnings("ignore", message=r"`torch\.nn\.utils\.weight_norm` is deprecated")
    warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated")

    # huggingface_hub logs the server's "unauthenticated requests" header at WARNING.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    # If everything is cached, don't touch the network at all (faster, and no
    # HF warning). Otherwise stay online so missing files can download.
    if all_cached(voice_names):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def lang_code_for(voices: list[tuple[str, float]], lang_codes: dict) -> str:
    """Pick lang_code from the voice prefix (am_* -> 'a', bm_* -> 'b').

    For a blend, the highest-weighted voice decides.
    """
    dominant = max(voices, key=lambda v: v[1])[0]
    code = Path(dominant).name[:1]
    if code not in lang_codes:
        sys.exit(f"error: can't infer language from voice '{dominant}' "
                 f"(known prefixes: {', '.join(lang_codes)})")
    others = {Path(n).name[:1] for n, _ in voices} - {code}
    if others:
        print(f"note: blending voices from different languages; "
              f"using lang_code '{code}' ({lang_codes[code]}) from {dominant}",
              file=sys.stderr)
    return code


def fetch_voices(voice_names: list[str]) -> None:
    """Download (or find in cache) each voice file, failing with a clear message."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError

    for name in voice_names:
        if name.endswith(".pt"):
            if not Path(name).is_file():
                sys.exit(f"error: voice file not found: {name}")
            continue
        try:
            hf_hub_download(repo_id=REPO_ID, filename=f"voices/{name}.pt")
        except RemoteEntryNotFoundError:
            sys.exit(f"error: unknown voice '{name}' "
                     f"(see https://huggingface.co/{REPO_ID}/blob/main/VOICES.md)")
        except LocalEntryNotFoundError:
            sys.exit(f"error: voice '{name}' isn't cached and couldn't be downloaded "
                     f"(offline?)")


def build_voice(pipeline, voices: list[tuple[str, float]]):
    """Return a voice name for a single voice, or a weighted-average tensor for a blend."""
    if len(voices) == 1:
        return voices[0][0]
    total = sum(w for _, w in voices)
    blended = sum(pipeline.load_voice(name) * (w / total) for name, w in voices)
    return blended.float()


def pick_device(requested: str, torch) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_text(args) -> str:
    if args.file:
        if args.file == "-":
            return sys.stdin.read()
        try:
            return Path(args.file).read_text(encoding="utf-8")
        except FileNotFoundError:
            sys.exit(f"error: file not found: {args.file}")
    return args.text


def main() -> None:
    p = argparse.ArgumentParser(description="Generate speech with Kokoro TTS.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("text", nargs="?", help="text to speak")
    src.add_argument("-f", "--file", help="read text from file ('-' for stdin)")
    p.add_argument("--voice", default="am_michael",
                   help="voice name, or weighted blend like 'am_michael:60,bm_george:40'")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", default="output.wav",
                   help="output wav path (relative paths are from the current directory)")
    p.add_argument("--play", action="store_true", help="play result with afplay")
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--verbose", action="store_true",
                   help="show all library warnings and logging (for debugging)")
    args = p.parse_args()

    text = read_text(args).strip()
    if not text:
        sys.exit("error: no text to speak")

    voices = parse_voice_spec(args.voice)
    voice_names = [name for name, _ in voices]

    if not args.verbose:
        quiet_libraries(voice_names)

    # Heavy imports happen only after the logging/warning setup above.
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline
    from kokoro.pipeline import LANG_CODES

    if args.verbose:
        from loguru import logger
        logger.enable("kokoro")  # kokoro disables its own loguru output by default

    lang_code = lang_code_for(voices, LANG_CODES)
    fetch_voices(voice_names)
    device = pick_device(args.device, torch)

    pipeline = KPipeline(lang_code=lang_code, repo_id=REPO_ID, device=device)
    voice = build_voice(pipeline, voices)

    chunks = []
    for result in pipeline(text, voice=voice, speed=args.speed, split_pattern=r"\n+"):
        if result.audio is not None:
            chunks.append(result.audio.detach().cpu().numpy())
    if not chunks:
        sys.exit("error: no audio generated")

    audio = np.concatenate(chunks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, SAMPLE_RATE)

    model_device = next(pipeline.model.parameters()).device
    print(f"wrote {out} ({len(audio) / SAMPLE_RATE:.2f}s, {len(chunks)} chunk(s), "
          f"voice={args.voice}, lang={lang_code}, device={model_device})")

    if args.play:
        subprocess.run(["afplay", str(out)], check=False)


if __name__ == "__main__":
    main()
