#!/usr/bin/env python3
"""
Download a video from any URL (YouTube, YouTube Shorts, etc.) and
convert it to the format expected by the VACE workflow:
  - Resolution: 832×480 (or nearest AR-preserving crop)
  - Frame rate: 16 fps
  - Max frames: 49 (~3 seconds)
  - Codec: h264 MP4
  - Output: ComfyUI/input/<name>.mp4

Usage:
    python3 scripts/download_video.py <url> [output_name] [--frames N] [--start SS]

Examples:
    python3 scripts/download_video.py https://www.youtube.com/shorts/03WNMH5Az2c guitar_bird
    python3 scripts/download_video.py https://youtu.be/abc123 my_clip --frames 81 --start 5
"""

import argparse
import subprocess
import sys
import tempfile
import os
from pathlib import Path

COMFYUI_INPUT = Path.home() / "projects/ComfyUI/input"
TARGET_W, TARGET_H = 832, 480
TARGET_FPS = 16
DEFAULT_FRAMES = 49


def run(cmd, **kwargs):
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)
    return result.stdout.strip()


def probe_video(path):
    """Return (width, height, fps, duration_s) of a video file."""
    import json
    out = run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", str(path)
    ])
    stream = json.loads(out)["streams"][0]
    w = int(stream["width"])
    h = int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(stream.get("duration", 0))
    return w, h, fps, duration


def build_vf_filter(src_w, src_h, tgt_w, tgt_h, start_s, num_frames, fps):
    """
    Build an ffmpeg -vf filter string that:
      1. Crops to target aspect ratio
         - Landscape source (wider than target): centre crop on width
         - Portrait source (taller than target): top-biased crop on height
           (subject is usually near the top in phone/Shorts videos)
      2. Scales to target resolution
      3. Limits to num_frames frames at target fps
    """
    src_ar = src_w / src_h
    tgt_ar = tgt_w / tgt_h

    if src_ar > tgt_ar:
        # Source is wider — centre crop width
        crop_h = src_h
        crop_w = int(src_h * tgt_ar)
        crop_x = (src_w - crop_w) // 2
        crop_y = 0
    else:
        # Source is taller (portrait) — top-biased crop height
        # Take from the top 20% down so subject head is not cut off
        crop_w = src_w
        crop_h = int(src_w / tgt_ar)
        crop_x = 0
        # Start 10% from top instead of centre — keeps heads in frame
        crop_y = int(src_h * 0.10)

    duration_s = num_frames / fps
    filters = [
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        f"scale={tgt_w}:{tgt_h}:flags=lanczos",
        f"fps={fps}",
        f"trim=start={start_s}:duration={duration_s}",
        "setpts=PTS-STARTPTS",
    ]
    return ",".join(filters)


def download(url, output_name, num_frames, start_s):
    COMFYUI_INPUT.mkdir(parents=True, exist_ok=True)
    out_path = COMFYUI_INPUT / f"{output_name}.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.%(ext)s"
        print(f"Downloading: {url}")
        run([
            "yt-dlp",
            "--no-playlist",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", str(raw),
            url,
        ])

        # Find what was downloaded
        downloaded = list(Path(tmp).glob("raw.*"))
        if not downloaded:
            print("ERROR: yt-dlp produced no output file", file=sys.stderr)
            sys.exit(1)
        raw_file = downloaded[0]
        print(f"Downloaded: {raw_file.name}  ({raw_file.stat().st_size / 1e6:.1f} MB)")

        src_w, src_h, src_fps, duration = probe_video(raw_file)
        print(f"Source: {src_w}×{src_h}  {src_fps:.2f}fps  {duration:.1f}s")

        vf = build_vf_filter(src_w, src_h, TARGET_W, TARGET_H, start_s, num_frames, TARGET_FPS)
        print(f"Converting → {TARGET_W}×{TARGET_H} @ {TARGET_FPS}fps, {num_frames} frames "
              f"(start={start_s}s) ...")

        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(raw_file),
            "-vf", vf,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-an",  # no audio
            str(out_path),
        ], check=True, capture_output=True)

    w, h, fps, dur = probe_video(out_path)
    size_kb = out_path.stat().st_size // 1024
    print(f"\nSaved: {out_path}")
    print(f"       {w}×{h}  {fps:.0f}fps  {dur:.2f}s  {size_kb}KB")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="Video URL (YouTube, Shorts, etc.)")
    ap.add_argument("output_name", nargs="?", default="source",
                    help="Output filename without extension (default: source)")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                    help=f"Number of frames to extract (default: {DEFAULT_FRAMES} = ~3s at 16fps)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Start time in seconds (default: 0)")
    args = ap.parse_args()

    out = download(args.url, args.output_name, args.frames, args.start)
    print(f"\nNext: set  \"video\": \"{out.name}\"  in workflows/vace_prop_addition.json node 1")


if __name__ == "__main__":
    main()
