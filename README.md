# Vertical-Video

竖屏视频工具库 — 下载、语义分割、横转竖、图片动态化，完整的短视频制作管线。

## Tools

### dl.py — Video Download

```bash
python dl.py <url> [quality]
# quality: 2160 1440 1080 720 480 360 (default: 1440)
```

Requires [yt-dlp](https://github.com/yt-dlp/yt-dlp) and ffmpeg. Place cookies at `yt-cookies/cookie.txt`.

### scene_cut.py — Three-Layer Semantic Scene Segmentation

| Layer | Engine | Role |
|-------|--------|------|
| L1 | PySceneDetect | HSV pixel-level shot boundary detection |
| L2 | Local VLM | Semantic scene clustering (merge shots of same scene) |
| L3 | ffmpeg | Lossless split with automatic black-frame trimming |

```bash
python scene_cut.py <video> [--split]
# --split  output scene clips to scenes/
```

Requires a local VLM endpoint (e.g. [lmstudio](https://lmstudio.ai)) running at `http://localhost:1234/v1/chat/completions`.

### reframe.py — Intelligent Auto-Reframe

Crops landscape video to portrait (9:16) with saliency-guided tracking, Kalman-filtered smoothing, and cubic-spline interpolation for cinema-grade camera movement.

**Pipeline:** Saliency detection → Center-biased tracking → Kalman filter → CubicSpline interpolation → Pipe-stream render

```bash
python reframe.py <video> [ratio] [--apply] [--speed=N]
# ratio:  9:16 1:1 4:5 16:9 (default: 9:16)
# --apply  render output (analysis only if omitted)
# --speed  playback speed factor (default: 0.5 = 2x slow motion)
```

**Key parameters** (tunable in source):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CENTER_BIAS` | 0.7 | Crop anchoring: 0.7 = 70% weight on frame center, 30% on saliency |
| `FOLLOW_SPEED` | 0.08 | Tracking responsiveness per sample step |
| `MAX_DELTA_PER_SAMPLE` | 50 | Max crop-center movement per sample (px) |
| `PLAYBACK_SPEED` | 0.5 | Output speed (0.5 = slow motion with frame interpolation) |

**Frame interpolation:** Uses ffmpeg `minterpolate` by default. For higher quality (especially smoke/mist/water scenes), install [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) into `./rife-ncnn-vulkan/`.

## Test Sample

```bash
python dl.py "https://www.youtube.com/watch?v=fJ4jfvS5v4o"
python scene_cut.py downloads/<video>.mp4 --split
python reframe.py downloads/scenes/<video>_scene_001.mp4 9:16 --apply --speed=0.5
```

Test video: [Indonesia 4K Drone Aerial](https://www.youtube.com/watch?v=fJ4jfvS5v4o) — a 4-minute aerial drone footage with diverse landscapes (volcanoes, waterfalls, coastline), ideal for testing all three tools.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

External dependencies: [ffmpeg](https://ffmpeg.org), [yt-dlp](https://github.com/yt-dlp/yt-dlp)

## Architecture

```
ytbdown/
├── dl.py                # Video download
├── scene_cut.py         # Three-layer scene segmentation
├── reframe.py           # Intelligent auto-reframe
├── requirements.txt     # Python dependencies
├── rife-ncnn-vulkan/    # (optional) RIFE frame interpolation
├── yt-cookies/          # YouTube cookies
└── downloads/           # Downloaded & processed videos
```

## License

[MIT](LICENSE)
