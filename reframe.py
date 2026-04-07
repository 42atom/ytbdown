#!/usr/bin/env python3
"""
Auto Reframe — 视频智能重构图

将宽屏视频（如 16:9 4K航拍）智能裁切为目标比例（如 9:16 竖屏），
裁切框自动追踪画面显著区域，模拟真实运镜手感。

管线:
  分析: 显著性检测 → 重心提取 → 卡尔曼滤波 → 稀疏采样点
  渲染: 采样点插值到全帧 → numpy 裁切 → pipe 流式推送 ffmpeg → 单次输出
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

# 目标比例预设
ASPECT_PRESETS = {
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:5": (4, 5),
    "16:9": (16, 9),
}

# 输出目标宽度（高度按比例自动计算）
OUTPUT_WIDTH = 1080

# 卡尔曼参数
KALMAN_PROCESS_NOISE = 5e-3  # 过程噪声，小→更稳更慢
KALMAN_MEASURE_NOISE = 5e-1  # 测量噪声，大→更信任预测，画面更稳

# 每测量步（0.2s@5fps）最大跳变像素（物理相机速限，防止眩晕）
MAX_DELTA_PER_SAMPLE = 50

# 裁切锚定策略：中心偏向（0=完全跟热点, 0.9=90%锚定画面中心）
# 竖屏观众习惯镜头稳定，降低此值让裁切跟随内容但减少反向推拉感
CENTER_BIAS = 0.7  # 70%权重在画面中心，30%在热点

# 跟随速度：每次追上目标位置的多少（值越小→越迟钝→越稳）
FOLLOW_SPEED = 0.08

# 插帧减速：<1=慢动作加长，>1=快进（默认0.5=视频拉长一倍）
PLAYBACK_SPEED = 0.5


class CropTracker:
    """卡尔曼滤波裁切框追踪器 (状态: [x, y, vx, vy])，测量端物理速限"""

    def __init__(self, start_x: float, start_y: float):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASURE_NOISE
        self.kf.statePre = np.array([start_x, start_y, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([start_x, start_y, 0, 0], dtype=np.float32)
        self._last_x, self._last_y = start_x, start_y  # 上一次（已限速的）测量值

    def update(self, x: float, y: float) -> tuple[float, float]:
        # 测量端物理速限：相邻样本间最大跳变（5fps→每0.2s一次）
        dx = x - self._last_x
        dy = y - self._last_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > MAX_DELTA_PER_SAMPLE:
            scale = MAX_DELTA_PER_SAMPLE / dist
            x = self._last_x + dx * scale
            y = self._last_y + dy * scale
        self._last_x, self._last_y = x, y

        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kf.predict()
        corrected = self.kf.correct(measurement)
        return float(corrected[0]), float(corrected[1])


# ////// 分析阶段

def saliency_center(frame: np.ndarray) -> tuple[float, float]:
    """计算一帧的显著性重心坐标"""
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, mask = saliency.computeSaliency(frame)
    if not success:
        h, w = frame.shape[:2]
        return w / 2, h / 2

    _, mask = cv2.threshold(mask, 0.3, 1.0, cv2.THRESH_BINARY)
    mask = (mask * 255).astype(np.uint8)

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        h, w = frame.shape[:2]
        return w / 2, h / 2

    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def clamp_crop(center_x: float, center_y: float,
               frame_w: int, frame_h: int,
               crop_w: int, crop_h: int) -> tuple[float, float]:
    """根据重心计算裁切框左上角，浮点精度，边界钳制"""
    x = center_x - crop_w / 2
    y = center_y - crop_h / 2
    x = max(0, min(x, frame_w - crop_w))
    y = max(0, min(y, frame_h - crop_h))
    return x, y


def analyze_crop_path(
    video_path: str, target_ratio: tuple[int, int],
    sample_fps: float = 5.0,
) -> tuple[list[dict], dict]:
    """
    稀疏采样分析，返回 (采样点列表, 视频信息)。
    采样点包含卡尔曼平滑后的裁切框中心坐标（浮点）。
    """
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"width": w, "height": h, "fps": fps, "frames": total}

    print(f"原始分辨率: {w}x{h} @ {fps:.1f}fps ({total} 帧)")

    # 裁切尺寸（保持全高，按9:16比例裁宽）
    tw, th = target_ratio
    crop_h = h
    crop_w = int(h * tw / th)
    if crop_w > w:
        crop_w = w
        crop_h = int(w * th / tw)
    info["crop_w"] = crop_w
    info["crop_h"] = crop_h
    display_frac = crop_w / w
    print(f"裁切框: {crop_w}x{crop_h} (占原始 {display_frac*100:.0f}%)")

    frame_interval = max(1, int(fps / sample_fps))
    tracker = CropTracker(w / 2, h / 2)
    samples = []
    frame_idx = 0
    frame_center_x, frame_center_y = w / 2, h / 2
    _prev_damped = [frame_center_x, frame_center_y]  # 初始锚定画面中心

    print(f"分析中 (每{frame_interval}帧采样, 中心锚定={CENTER_BIAS}) ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            sx, sy = saliency_center(frame)
            # 目标：热点权重 CENTER_BIAS，中心权重 1-CENTER_BIAS
            # → 裁切主要锚定中心，热点偏移时才轻推
            target_x = CENTER_BIAS * frame_center_x + (1 - CENTER_BIAS) * sx
            target_y = CENTER_BIAS * frame_center_y + (1 - CENTER_BIAS) * sy
            # FOLLOW_SPEED 控制追踪速度
            cx = _prev_damped[0] + (target_x - _prev_damped[0]) * FOLLOW_SPEED
            cy = _prev_damped[1] + (target_y - _prev_damped[1]) * FOLLOW_SPEED
            _prev_damped[0], _prev_damped[1] = cx, cy
            smooth_x, smooth_y = tracker.update(cx, cy)
            # 存裁切框中心（浮点），不做取整
            samples.append({
                "time": round(frame_idx / fps, 4),
                "frame": frame_idx,
                "cx": smooth_x,
                "cy": smooth_y,
            })

            if len(samples) % 20 == 0:
                print(f"  {frame_idx/ fps:.1f}s 热点=({sx:.0f},{sy:.0f}) → 裁切=({cx:.0f},{cy:.0f})")

        frame_idx += 1

    cap.release()
    print(f"分析完成: {len(samples)} 个采样点")
    return samples, info


# ////// 插值：稀疏采样 → 全帧坐标

def interpolate_path(
    samples: list[dict], total_frames: int, fps: float,
    frame_w: int, frame_h: int, crop_w: int, crop_h: int,
) -> list[tuple[int, int]]:
    """
    三次样条插值：稀疏采样点 → 全帧坐标。
    相比 np.interp 的线性插值，CubicSpline 保证二阶导数连续，
    裁切框运动带有物理缓起缓停（Ease In/Out），无折角突变。
    """
    if not samples:
        return [(0, 0)] * total_frames

    if len(samples) < 2:
        # 单点，居中不动
        cx, cy = samples[0]["cx"], samples[0]["cy"]
        x, y = clamp_crop(cx, cy, frame_w, frame_h, crop_w, crop_h)
        return [(int(round(x)), int(round(y)))] * total_frames

    sample_frames = np.array([s["frame"] for s in samples], dtype=np.float64)
    sample_cx = np.array([s["cx"] for s in samples], dtype=np.float64)
    sample_cy = np.array([s["cy"] for s in samples], dtype=np.float64)

    # 三次样条：C² 连续，加速度平滑，运镜无折角
    all_frames = np.arange(total_frames, dtype=np.float64)
    cs_x = CubicSpline(sample_frames, sample_cx)
    cs_y = CubicSpline(sample_frames, sample_cy)
    interp_cx = cs_x(all_frames)
    interp_cy = cs_y(all_frames)

    # 转为裁切框左上角整数坐标
    result = []
    for cx, cy in zip(interp_cx, interp_cy):
        x, y = clamp_crop(cx, cy, frame_w, frame_h, crop_w, crop_h)
        result.append((int(round(x)), int(round(y))))

    return result


# ////// 渲染阶段：管道流式输出

def render_reframed(
    video_path: str, crop_coords: list[tuple[int, int]],
    info: dict, output_path: str, speed: float = 1.0,
):
    """
    单趟流式渲染：OpenCV 读帧 → numpy 裁切 → pipe 推 ffmpeg。
    零临时文件，亚像素级平滑，保留原始音轨。
    speed < 1：慢动作（setpts 插帧 + atempo 慢音）
    """
    crop_w, crop_h = info["crop_w"], info["crop_h"]
    fps = info["fps"]
    total = info["frames"]

    # 输出高度 = 按裁切宽等比（9:16）
    output_h = int(OUTPUT_WIDTH * crop_h / crop_w)

    # 视频滤镜链
    vf_parts = [f"scale={OUTPUT_WIDTH}:{output_h}"]
    if speed < 1.0:
        # 慢动作：setpts 拉长 + minterpolate 运动补偿插帧（消除闪烁）
        vf_parts.append(f"setpts={1/speed:.3f}*PTS")
        vf_parts.append(f"minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={fps:.0f}")
        output_fps = fps  # 插帧回原帧率，时长自动拉长
    elif speed > 1.0:
        vf_parts.append(f"setpts={1/speed:.3f}*PTS")
        output_fps = fps
    else:
        output_fps = fps

    # ffmpeg 参数构建
    ffmpeg_args = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{crop_w}x{crop_h}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", video_path,
        "-map", "0:v",
    ]
    if speed == 1.0:
        ffmpeg_args += ["-map", "1:a?", "-c:a", "aac", "-b:a", "192k"]
    else:
        ffmpeg_args += ["-an"]
    ffmpeg_args += [
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-vf", ",".join(vf_parts),
        "-r", str(round(output_fps, 3)),
        "-shortest",
        output_path
    ]
    process = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # 守护线程实时排空 stderr，防止管道缓冲区满→死锁
    stderr_chunks = []
    def drain_stderr():
        while True:
            chunk = process.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)

    drain_thread = threading.Thread(target=drain_stderr, daemon=True)
    drain_thread.start()

    cap = cv2.VideoCapture(video_path)
    prev_xy = None
    max_delta = 0

    print(f"渲染中 ({total} 帧, {crop_w}x{crop_h}) ...")

    for frame_idx in range(min(total, len(crop_coords))):
        ret, frame = cap.read()
        if not ret:
            break

        x, y = crop_coords[frame_idx]

        # 跟踪最大帧间跳变（验证平滑度）
        if prev_xy:
            delta = abs(x - prev_xy[0]) + abs(y - prev_xy[1])
            max_delta = max(max_delta, delta)
        prev_xy = (x, y)

        # numpy 切片裁切，直接推入管道
        cropped = frame[y:y+crop_h, x:x+crop_w]

        # 安全检查：尺寸匹配
        if cropped.shape[1] != crop_w or cropped.shape[0] != crop_h:
            cropped = cv2.resize(cropped, (crop_w, crop_h))

        process.stdin.write(cropped.tobytes())

        if frame_idx % 500 == 0:
            pct = frame_idx / total * 100
            print(f"  {pct:.0f}% ({frame_idx}/{total}) 裁切=({x},{y})")

    process.stdin.close()
    drain_thread.join()
    process.wait()

    cap.release()

    if process.returncode != 0:
        stderr_text = b"".join(stderr_chunks).decode(errors="replace")
        print(f"ffmpeg 错误: {stderr_text[-500:]}")
    else:
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"渲染完成: {output_path} ({size_mb:.1f}MB)")
        print(f"最大帧间跳变: {max_delta}px")


# ////// 入口

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video> [ratio] [--apply]")
        print(f"  ratio: {' '.join(ASPECT_PRESETS)} (default: 9:16)")
        print(f"  --apply  执行裁切输出（否则只分析）")
        sys.exit(1)

    video = sys.argv[1]
    ratio_str = "9:16"
    apply = False
    speed = PLAYBACK_SPEED

    for arg in sys.argv[2:]:
        if arg == "--apply":
            apply = True
        elif arg.startswith("--speed="):
            speed = float(arg.split("=", 1)[1])
        elif arg in ASPECT_PRESETS:
            ratio_str = arg

    ratio = ASPECT_PRESETS[ratio_str]
    print(f"目标比例: {ratio_str} ({ratio[0]}:{ratio[1]})")
    if speed != 1.0:
        print(f"播放速度: {speed}x ({'慢动作' if speed < 1 else '快进'})")

    # 分析
    samples, info = analyze_crop_path(video, ratio)

    if not samples:
        print("分析失败")
        sys.exit(1)

    # 保存分析结果
    result_file = Path(video).with_suffix(f".reframe_{ratio_str.replace(':', 'x')}.json")
    result_file.write_text(json.dumps({
        "video": str(video),
        "ratio": ratio_str,
        "crop_size": [info["crop_w"], info["crop_h"]],
        "samples": samples,
    }, indent=2))
    print(f"裁切路径已保存: {result_file}")

    # 渲染
    if apply:
        print(f"\n=== 插值 + 流式渲染 ===")
        crop_coords = interpolate_path(
            samples, info["frames"], info["fps"],
            info["width"], info["height"],
            info["crop_w"], info["crop_h"],
        )
        print(f"插值完成: {len(crop_coords)} 帧坐标")

        name = Path(video).stem
        output = os.path.join(
            os.path.dirname(os.path.abspath(video)),
            f"{name}_{ratio_str.replace(':', 'x')}.mp4"
        )
        render_reframed(video, crop_coords, info, output, speed)


if __name__ == "__main__":
    main()
