#!/usr/bin/env python3
"""
视频语义场景分割 — 三层架构

L1 物理切分: PySceneDetect ContentDetector + ThresholdDetector
L2 语义聚类: 本地 VLM 判断相邻 Shot 是否同一语义场景（并发+重试）
L3 合并输出: 同场景 Shot 合并，ffmpeg 切割输出

数据流: 全程 cv2 内存解码，零 ffprobe 子进程，零 ffmpeg 抽帧子进程。
"""

import base64
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, ThresholdDetector

API = "http://localhost:1234/v1/chat/completions"
MODEL = "huihui-glm-4.6v-flash-abliterated-mlx"
FRAME_WIDTH = 512
MAX_WORKERS = 4

# L1 检测参数
CONTENT_THRESHOLD = 27
MIN_SCENE_LEN_FRAMES = 15

# L3 切割模式: copy=快但关键帧对齐, reencode=慢但帧精确
SPLIT_MODE = "copy"

# L2 VLM 重试
VLM_MAX_RETRIES = 3
VLM_RETRY_DELAY = 2.0

# 运行时统计
_vlm_stats = {"total": 0, "success": 0, "retry": 0, "fail": 0}
_vlm_lock = __import__("threading").Lock()


# // 工具函数

def get_video_meta(video_path: str) -> dict:
    """cv2 一次性取视频元信息，替代 ffprobe 子进程"""
    cap = cv2.VideoCapture(video_path)
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    meta["duration"] = meta["frames"] / meta["fps"] if meta["fps"] > 0 else 0.0
    cap.release()
    return meta


def encode_frame_jpeg(frame, quality: int = 85) -> str:
    """OpenCV 帧 → JPEG → base64，纯内存"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


# ////// L1: 物理镜头边界检测

def detect_shots(video_path: str) -> list[tuple[float, float]]:
    """PySceneDetect 检测镜头边界，返回 [(start_s, end_s), ...]"""
    meta = get_video_meta(video_path)
    fps = meta["fps"]

    video = open_video(video_path)
    scene_mgr = SceneManager()
    scene_mgr.add_detector(ContentDetector(
        threshold=CONTENT_THRESHOLD,
        min_scene_len=MIN_SCENE_LEN_FRAMES,
    ))
    scene_mgr.add_detector(ThresholdDetector(
        threshold=16,
        min_scene_len=MIN_SCENE_LEN_FRAMES,
    ))
    scene_mgr.detect_scenes(video, show_progress=True)
    scene_list = scene_mgr.get_scene_list()

    if not scene_list:
        dur = meta["duration"]
        print(f"L1 未检测到镜头边界，视为单个镜头 ({dur:.1f}s)")
        return [(0.0, dur)]

    shots = []
    for scene in scene_list:
        start_s = scene[0].get_frames() / fps
        end_s = scene[1].get_frames() / fps
        shots.append((start_s, end_s))

    print(f"L1 检测到 {len(shots)} 个镜头:")
    for i, (s, e) in enumerate(shots):
        print(f"  镜头{i+1}: {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")
    return shots


# ////// L2: 语义场景聚类

def extract_mid_frames_encoded(
    video_path: str, shots: list[tuple[float, float]]
) -> list[str]:
    """
    cv2 内存解码，每个镜头抽中间帧，缩放后 base64 预编码。
    数据流: VideoCapture → numpy → resize → JPEG encode → base64 (一次性)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    encoded = []

    for start_s, end_s in shots:
        mid_idx = int((start_s + end_s) / 2 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()
        if not ret:
            encoded.append("")
            continue

        h, w = frame.shape[:2]
        new_w = FRAME_WIDTH
        new_h = int(h * new_w / w)
        frame = cv2.resize(frame, (new_w, new_h))
        encoded.append(encode_frame_jpeg(frame))

    cap.release()
    return encoded


def vlm_same_scene(b64_a: str, b64_b: str) -> bool:
    """VLM 判断两帧是否同一语义场景，带指数退避重试"""
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "These are two keyframes from an aerial drone video. "
                    "Are they showing the SAME geographic location/landmark? "
                    "Consider camera angle changes (zoom, pan) as still the same scene. "
                    "Only say 'no' if the location has clearly changed "
                    "(e.g. from a volcano to a waterfall, from ocean to city). "
                    "Answer ONLY 'yes' or 'no'."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_a}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_b}"}},
            ]
        }],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    last_err = None
    with _vlm_lock:
        _vlm_stats["total"] += 1
    for attempt in range(VLM_MAX_RETRIES):
        try:
            resp = requests.post(API, json=payload, timeout=60)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
            with _vlm_lock:
                _vlm_stats["success"] += 1
            return "yes" in answer
        except (requests.RequestException, KeyError, IndexError) as e:
            last_err = e
            with _vlm_lock:
                _vlm_stats["retry"] += 1
            if attempt < VLM_MAX_RETRIES - 1:
                wait = VLM_RETRY_DELAY * (2 ** attempt)
                print(f"    VLM 重试 {attempt+2}/{VLM_MAX_RETRIES} (等{wait:.0f}s, {e})")
                time.sleep(wait)

    with _vlm_lock:
        _vlm_stats["fail"] += 1
    raise RuntimeError(f"VLM 连续 {VLM_MAX_RETRIES} 次失败: {last_err}")


def cluster_shots(video_path: str, shots: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    一次性内存抽帧 + 预编码 + 并发 VLM 比较 + 合并相邻同场景镜头。
    """
    # 一次性预编码
    print("  抽帧编码中 ...")
    encoded = extract_mid_frames_encoded(video_path, shots)
    print(f"  {len(encoded)} 帧已编码")

    # 并发逐对比较
    pair_count = len(shots) - 1
    results = {}

    def compare_pair(i):
        return i, vlm_same_scene(encoded[i], encoded[i + 1])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(compare_pair, i): i for i in range(pair_count)}
        for future in as_completed(futures):
            i, same = future.result()
            results[i] = same
            tag = "同场景" if same else "不同场景"
            print(f"  L2: 镜头{i+1} vs 镜头{i+2} → {tag}")

    # 按顺序合并
    groups = [[0]]
    for i in range(1, len(shots)):
        if results.get(i - 1, False):
            groups[-1].append(i)
        else:
            groups.append([i])

    scenes = []
    for group in groups:
        start = shots[group[0]][0]
        end = shots[group[-1]][1]
        scenes.append((start, end))

    print(f"L2 合并后 {len(scenes)} 个语义场景")
    stats = _vlm_stats
    print(f"  VLM 统计: 总调用={stats['total']} 成功={stats['success']} 重试={stats['retry']} 失败={stats['fail']}")
    return scenes


# ////// L3: 输出切割

def trim_leading_black(video_path: str, threshold_ratio: float = 0.9) -> float:
    """检测视频头部黑屏/渐入，返回应裁剪的秒数"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    brightnesses = []
    for _ in range(total):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses.append(gray.mean())
    cap.release()

    if not brightnesses:
        return 0.0

    mid = float(np.median(brightnesses))
    if mid < 5:
        return 0.0

    threshold = mid * threshold_ratio
    above = 0
    for i, b in enumerate(brightnesses):
        if b >= threshold:
            above += 1
            if above >= 3:
                return max(0, (i - 2)) / fps
        else:
            above = 0
    return 0.0


def split_scenes(video_path: str, scenes: list[tuple[float, float]], output_dir: str):
    """按场景切分视频，自动去除头部黑屏渐入"""
    Path(output_dir).mkdir(exist_ok=True)
    name = Path(video_path).stem

    for i, (start, end) in enumerate(scenes):
        duration = end - start
        out_path = os.path.join(output_dir, f"{name}_scene_{i+1:03d}.mp4")

        if SPLIT_MODE == "copy":
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", video_path,
                "-t", str(duration),
                "-c", "copy", out_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-i", video_path,
                "-t", str(duration),
                "-c:v", "h264_videotoolbox", "-b:v", "8000k",
                "-c:a", "aac", "-b:a", "192k",
                out_path
            ]

        subprocess.run(cmd, capture_output=True)

        # 后处理：去除头部黑屏渐入
        black_trim = trim_leading_black(out_path)
        if black_trim > 0.05:  # 超过50ms才值得处理
            trimmed_path = out_path + ".tmp.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(black_trim), "-i", out_path,
                "-c", "copy", trimmed_path
            ], capture_output=True)
            os.replace(trimmed_path, out_path)
            print(f"  场景{i+1}: {start:.2f}s - {end:.2f}s ({duration:.2f}s) [去黑屏{black_trim:.2f}s] -> {out_path}")
        else:
            print(f"  场景{i+1}: {start:.2f}s - {end:.2f}s ({duration:.2f}s) -> {out_path}")


# ////// 入口

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video> [--split]")
        print(f"  --split  按场景拆分视频到 scenes/")
        sys.exit(1)

    video = sys.argv[1]
    do_split = "--split" in sys.argv

    meta = get_video_meta(video)
    duration = meta["duration"]
    print(f"视频: {video} ({duration:.1f}s, {meta['width']}x{meta['height']} @ {meta['fps']:.1f}fps)\n")

    # L1
    print("=== L1 物理镜头检测 ===")
    shots = detect_shots(video)

    # L2
    if len(shots) <= 1:
        print("\n=== L2 跳过（仅1个镜头）===")
        scenes = shots
    else:
        print(f"\n=== L2 语义聚类 ({len(shots)} 个镜头) ===")
        scenes = cluster_shots(video, shots)

    # 结果
    print(f"\n=== 结果 ===")
    print(f"共 {len(scenes)} 个语义场景:")
    for i, (s, e) in enumerate(scenes):
        print(f"  场景{i+1}: {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")

    # L3
    if do_split:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(video)), "scenes")
        print(f"\n=== L3 拆分输出 ===")
        split_scenes(video, scenes, output_dir)

    # 保存
    result_file = Path(video).with_suffix(".scenes.json")
    result_file.write_text(json.dumps({
        "video": str(video),
        "duration": duration,
        "scenes": [{"index": i+1, "start": round(s, 3), "end": round(e, 3), "duration": round(e-s, 3)}
                    for i, (s, e) in enumerate(scenes)],
    }, indent=2, ensure_ascii=False))
    print(f"\n场景信息已保存: {result_file}")


if __name__ == "__main__":
    main()
