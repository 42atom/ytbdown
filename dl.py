#!/usr/bin/env python3
"""YouTube 视频下载"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
COOKIE = ROOT / "yt-cookies" / "cookie.txt"
OUTDIR = ROOT / "downloads"

QUALITY_MAP = {
    "2160": "bv*[height<=2160]+ba/b[height<=2160]",
    "1440": "bv*[height<=1440]+ba/b[height<=1440]",
    "1080": "bv*[height<=1080]+ba/b[height<=1080]",
    "720":  "bv*[height<=720]+ba/b[height<=720]",
    "480":  "bv*[height<=480]+ba/b[height<=480]",
    "360":  "bv*[height<=360]+ba/b[height<=360]",
}

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <url> [quality]")
        print(f"  quality: {' '.join(QUALITY_MAP)} (default: 1080)")
        sys.exit(1)

    url = sys.argv[1]
    quality = sys.argv[2] if len(sys.argv) > 2 else "1440"

    if quality not in QUALITY_MAP:
        print(f"不支持的画质: {quality}，可选: {' '.join(QUALITY_MAP)}")
        sys.exit(1)

    OUTDIR.mkdir(exist_ok=True)

    cmd = [
        "yt-dlp",
        "--cookies", str(COOKIE),
        "-f", QUALITY_MAP[quality],
        "--merge-output-format", "mp4",
        "-o", str(OUTDIR / "%(title)s [%(id)s].%(ext)s"),
        url,
    ]

    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
