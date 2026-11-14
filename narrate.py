#!/usr/bin/env python3
"""Narrate a video with Kokoro TTS from a timestamped script.

Script format, one segment per line (blank lines and lines starting with # are ignored):

    00:00 | Welcome. In this video we upgrade RouterOS.
    00:12 | First, open System, then Packages.
    01:05 | Click Check For Updates.

Timestamps are MM:SS (H:MM:SS and fractional seconds like 00:12.5 also work).

Each segment is generated separately and placed at its timestamp on a silent track the
length of the video. A clip whose speech runs past the next segment's start is NOT sped
up: it overlaps the next clip, and you get a warning saying how many seconds it overflows.
Speech running past the end of the video is warned about the same way and cut off there.
Kokoro's trailing silence doesn't count toward overflow.

Outputs, next to the video:
    <name>-narrated.mp4   video stream copied (not re-encoded), audio replaced by narration
    <name>-narration.wav  the narration track alone

Usage:
    python narrate.py routeros-upgrade-narration.txt upgrade-router-os.mp4 --voice am_michael
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# speak.py sits next to this file; reuse its voice/device/quiet helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import speak  # noqa: E402  (imports no heavy libraries at module level)

SAMPLE_RATE = speak.SAMPLE_RATE
SILENCE = 0.01  # ~-40 dBFS; quieter samples count as silence when measuring speech length
LINE_RE = re.compile(r"^\s*(?P<ts>[0-9:.]+)\s*\|\s*(?P<text>.*?)\s*$")


@dataclass
class Segment:
    line_no: int
    start: float  # seconds
    text: str


def fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(ts)
    secs = float(parts[-1])
    mins = int(parts[-2])
    hours = int(parts[0]) if len(parts) == 3 else 0
    if secs >= 60 or (len(parts) == 3 and mins >= 60):
        raise ValueError(ts)
    return hours * 3600 + mins * 60 + secs


def parse_script(path: Path) -> list[Segment]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        sys.exit(f"error: script not found: {path}")

    segments: list[Segment] = []
    for n, raw in enumerate(lines, 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = LINE_RE.match(raw)
        if not m:
            sys.exit(f"error: {path.name}:{n}: expected 'MM:SS | text', got: {raw.strip()!r}")
        try:
            start = parse_timestamp(m["ts"])
        except ValueError:
            sys.exit(f"error: {path.name}:{n}: bad timestamp {m['ts']!r} (want MM:SS)")
        if not m["text"]:
            sys.exit(f"error: {path.name}:{n}: no text after '|'")
        if segments and start <= segments[-1].start:
            sys.exit(f"error: {path.name}:{n}: timestamp {m['ts']} is not after the previous "
                     f"segment ({fmt_ts(segments[-1].start)}, line {segments[-1].line_no})")
        segments.append(Segment(n, start, m["text"]))

    if not segments:
        sys.exit(f"error: no segments in {path}")
    return segments


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    brew = shutil.which("brew")
    if not brew:
        sys.exit("error: ffmpeg not found and Homebrew isn't installed; install ffmpeg first")
    print("ffmpeg not found; installing with: brew install ffmpeg", file=sys.stderr)
    subprocess.run([brew, "install", "ffmpeg"], check=True)
    if not shutil.which("ffmpeg"):
        sys.exit("error: ffmpeg still not on PATH after brew install")


def video_duration(video: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"error: ffprobe couldn't read {video}:\n{r.stderr.strip()}")
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (KeyError, ValueError):
        sys.exit(f"error: couldn't get duration of {video}")


def mux(video: Path, wav: Path, out: Path, verbose: bool) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info" if verbose else "error", "-nostats",
           "-i", str(video), "-i", str(wav),
           "-map", "0:v", "-map", "1:a",  # all video from the original, audio only from narration
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"error: ffmpeg failed muxing {out} (exit {r.returncode})")


def main() -> None:
    p = argparse.ArgumentParser(description="Narrate a video with Kokoro TTS from a timestamped script.")
    p.add_argument("script", help="text file, one 'MM:SS | text' segment per line")
    p.add_argument("video", help="video to narrate")
    p.add_argument("--voice", default="am_michael",
                   help="voice name, or weighted blend like 'am_michael:60,bm_george:40'")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--verbose", action="store_true",
                   help="show all library warnings, logging and ffmpeg output")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep progress and warnings in order when piped

    script = Path(args.script)
    video = Path(args.video)
    if not video.is_file():
        sys.exit(f"error: video not found: {video}")
    segments = parse_script(script)

    ensure_ffmpeg()
    duration = video_duration(video)

    voices = speak.parse_voice_spec(args.voice)
    voice_names = [name for name, _ in voices]
    if not args.verbose:
        speak.quiet_libraries(voice_names)

    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline
    from kokoro.pipeline import LANG_CODES

    if args.verbose:
        from loguru import logger
        logger.enable("kokoro")

    lang_code = speak.lang_code_for(voices, LANG_CODES)
    speak.fetch_voices(voice_names)
    device = speak.pick_device(args.device, torch)
    pipeline = KPipeline(lang_code=lang_code, repo_id=speak.REPO_ID, device=device)
    voice = speak.build_voice(pipeline, voices)

    total = int(round(duration * SAMPLE_RATE))
    track = np.zeros(total, dtype=np.float32)
    overflows: list[tuple[Segment, float, str]] = []

    print(f"video {video.name}: {fmt_ts(duration)} ({duration:.2f}s), {len(segments)} segment(s)")
    for i, seg in enumerate(segments):
        if seg.start >= duration:
            print(f"warning: line {seg.line_no} at {fmt_ts(seg.start)} starts after the video ends "
                  f"({fmt_ts(duration)}); skipped", file=sys.stderr)
            continue

        chunks = [r.audio.detach().cpu().numpy()
                  for r in pipeline(seg.text, voice=voice, speed=args.speed, split_pattern=None)
                  if r.audio is not None]
        if not chunks:
            print(f"warning: line {seg.line_no}: no audio generated; skipped", file=sys.stderr)
            continue
        clip = np.concatenate(chunks).astype(np.float32)
        # Kokoro pads clips with trailing silence; measure overflow from the last
        # audible sample so padding alone never counts as running into the next line.
        audible = np.nonzero(np.abs(clip) > SILENCE)[0]
        clip_len = (audible[-1] + 1 if len(audible) else len(clip)) / SAMPLE_RATE
        end = seg.start + clip_len

        nxt = segments[i + 1] if i + 1 < len(segments) else None
        limit, limit_name = ((nxt.start, f"next segment at {fmt_ts(nxt.start)} (line {nxt.line_no})")
                             if nxt and nxt.start < duration
                             else (duration, f"end of video at {fmt_ts(duration)}"))
        status = ""
        if end > limit:
            over = end - limit
            overflows.append((seg, over, limit_name))
            status = f"  OVERFLOW +{over:.2f}s"
            print(f"warning: line {seg.line_no} at {fmt_ts(seg.start)} has {clip_len:.2f}s of speech and "
                  f"runs {over:.2f}s into the {limit_name}", file=sys.stderr)
        print(f"  {fmt_ts(seg.start)}  {clip_len:5.2f}s  line {seg.line_no}{status}")

        a = int(round(seg.start * SAMPLE_RATE))
        b = min(a + len(clip), total)  # anything past the end of the video is cut
        track[a:b] += clip[: b - a]   # overlapping clips are summed, not sped up

    np.clip(track, -1.0, 1.0, out=track)

    stem = video.with_suffix("")
    wav_out = Path(f"{stem}-narration.wav")
    mp4_out = Path(f"{stem}-narrated.mp4")
    sf.write(wav_out, track, SAMPLE_RATE)
    print(f"wrote {wav_out}")
    mux(video, wav_out, mp4_out, args.verbose)
    print(f"wrote {mp4_out}")

    if overflows:
        print(f"\n{len(overflows)} segment(s) overflow (not sped up: they overlap the next line, "
              f"or get cut off at the end of the video):", file=sys.stderr)
        for seg, over, limit_name in overflows:
            print(f"  line {seg.line_no} at {fmt_ts(seg.start)}: +{over:.2f}s into the {limit_name}",
                  file=sys.stderr)
    else:
        print("no overflows")


if __name__ == "__main__":
    main()
