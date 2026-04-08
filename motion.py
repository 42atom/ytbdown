#!/usr/bin/env python3
"""
图片动态化模块 — OpenCV 显著性驱动 Ken Burns 运镜

7种运镜方式，根据图片显著性分布自动选择：
  push        全景 → 推向兴趣点
  pull        兴趣点特写 → 拉出全景
  pan_lr      左→右 水平摇
  pan_rl      右→左 水平摇
  tilt        上下摇
  drift       对角线漂移 + 微推（最自然）
  hold_push   定点缓推（最稳，适合文字/图表）

用法：
  from motion import gen_motion
  gen_motion("scene.png", duration=8.0, output="out.mp4", w=1920, h=1080)
"""

import os
import subprocess
import threading

import cv2
import numpy as np


# ////// 显著性分析

def analyze_saliency(image: np.ndarray) -> dict:
    """分析图片显著性分布，返回重心、分散度、方向等特征"""
    h, w = image.shape[:2]
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, mask = saliency.computeSaliency(image)

    result = {"cx": w / 2, "cy": h / 2, "spread": "center",
              "direction": "none", "std_x": 0, "std_y": 0,
              "off_x": 0, "off_y": 0}

    if not success:
        return result

    _, mask = cv2.threshold(mask, 0.3, 1.0, cv2.THRESH_BINARY)
    mask = (mask * 255).astype(np.uint8)
    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return result

    # 重心
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    result["cx"] = cx
    result["cy"] = cy

    # 二阶矩 → 分散度
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    std_x = (mu20 ** 0.5) / w
    std_y = (mu02 ** 0.5) / h
    off_x = abs(cx - w / 2) / (w / 2)
    off_y = abs(cy - h / 2) / (h / 2)

    result["std_x"] = std_x
    result["std_y"] = std_y
    result["off_x"] = off_x
    result["off_y"] = off_y

    # 判断分布特征
    if std_x > 0.2 and std_y < 0.15:
        result["spread"] = "horizontal"
        result["direction"] = "left" if cx < w / 2 else "right"
    elif std_y > 0.2 and std_x < 0.15:
        result["spread"] = "vertical"
    elif std_x > 0.2 and std_y > 0.2:
        result["spread"] = "scattered"
    elif off_x > 0.25 or off_y > 0.25:
        result["spread"] = "off_center"
        if cx > w / 2 and cy > h / 2:
            result["direction"] = "bottom_right"
        elif cx < w / 2 and cy > h / 2:
            result["direction"] = "bottom_left"
        elif cx > w / 2:
            result["direction"] = "top_right"
        else:
            result["direction"] = "top_left"
    else:
        result["spread"] = "center"

    return result


# ////// 运镜规划

def plan_move(img_w: int, img_h: int, out_w: int, out_h: int,
              sal: dict) -> dict:
    """根据显著性分析结果规划运镜 — 7种方式"""
    cx, cy = img_w / 2, img_h / 2
    sal_x, sal_y = sal["cx"], sal["cy"]

    max_zoom = min(1.8, min(img_w / out_w, img_h / out_h))
    mid_zoom = (1.0 + max_zoom) / 2
    min_zoom = 1.0
    pan_margin = img_w * 0.25

    spread = sal["spread"]

    if spread == "horizontal":
        if sal_x > cx:
            return {"type": "pan_lr",
                    "start": {"cx": pan_margin, "cy": sal_y, "zoom": mid_zoom},
                    "end": {"cx": img_w - pan_margin, "cy": sal_y, "zoom": mid_zoom}}
        else:
            return {"type": "pan_rl",
                    "start": {"cx": img_w - pan_margin, "cy": sal_y, "zoom": mid_zoom},
                    "end": {"cx": pan_margin, "cy": sal_y, "zoom": mid_zoom}}

    elif spread == "off_center":
        start_x = cx + (cx - sal_x) * 0.5
        start_y = cy + (cy - sal_y) * 0.5
        return {"type": "push",
                "start": {"cx": start_x, "cy": start_y, "zoom": min_zoom},
                "end": {"cx": sal_x, "cy": sal_y, "zoom": max_zoom * 0.85}}

    elif spread == "center":
        # 兴趣居中 → hold_push 定点缓推（最稳）
        return {"type": "hold_push",
                "start": {"cx": cx, "cy": cy, "zoom": min_zoom},
                "end": {"cx": cx, "cy": cy, "zoom": mid_zoom * 0.9}}

    elif spread == "scattered":
        # 兴趣分散 → drift 对角漂移（最自然）
        start_x = cx - img_w * 0.08
        start_y = cy - img_h * 0.08
        end_x = cx + img_w * 0.08
        end_y = cy + img_h * 0.08
        return {"type": "drift",
                "start": {"cx": start_x, "cy": start_y, "zoom": mid_zoom * 0.95},
                "end": {"cx": end_x, "cy": end_y, "zoom": min_zoom}}

    elif spread == "vertical":
        v_margin = img_h * 0.25
        if sal_y > cy:
            return {"type": "tilt",
                    "start": {"cx": sal_x, "cy": v_margin, "zoom": mid_zoom},
                    "end": {"cx": sal_x, "cy": img_h - v_margin, "zoom": mid_zoom}}
        else:
            return {"type": "tilt",
                    "start": {"cx": sal_x, "cy": img_h - v_margin, "zoom": mid_zoom},
                    "end": {"cx": sal_x, "cy": v_margin, "zoom": mid_zoom}}

    # 兜底: pull
    return {"type": "pull",
            "start": {"cx": sal_x, "cy": sal_y, "zoom": max_zoom * 0.85},
            "end": {"cx": cx, "cy": cy, "zoom": min_zoom}}


# ////// 渲染

def _ease(t: float) -> float:
    """缓入缓出"""
    return t * t * (3 - 2 * t)


# 单段运镜的理想时长范围（秒）
SEGMENT_MIN = 6.0
SEGMENT_MAX = 15.0


def _plan_segments(duration: float, img_w: int, img_h: int,
                   out_w: int, out_h: int, sal: dict) -> list[dict]:
    """
    根据总时长拆分运镜段落。

    短场景（<15s）: 单段运镜
    中场景（15-30s）: 两段（运镜 + 反向/漂移）
    长场景（>30s）: 多段交替，每段 8-12 秒
    """
    if duration <= SEGMENT_MAX:
        # 短场景：单段
        move = plan_move(img_w, img_h, out_w, out_h, sal)
        return [{"move": move, "duration": duration}]

    # 计算段数：每段 8-12 秒
    n_segs = max(2, round(duration / 10.0))
    seg_dur = duration / n_segs

    cx, cy = img_w / 2, img_h / 2
    sal_x, sal_y = sal["cx"], sal["cy"]
    max_zoom = min(1.8, min(img_w / out_w, img_h / out_h))
    mid_zoom = (1.0 + max_zoom) / 2
    min_zoom = 1.0

    # 生成多段交替运镜：push → hold → drift → pull → hold → ...
    patterns = [
        lambda: plan_move(img_w, img_h, out_w, out_h, sal),  # 显著性驱动
        lambda: {"type": "hold_push",  # 定点缓推
                 "start": {"cx": cx, "cy": cy, "zoom": min_zoom},
                 "end": {"cx": cx, "cy": cy, "zoom": mid_zoom * 0.85}},
        lambda: {"type": "drift",  # 对角漂移
                 "start": {"cx": cx - img_w * 0.06, "cy": cy - img_h * 0.06,
                            "zoom": mid_zoom * 0.9},
                 "end": {"cx": cx + img_w * 0.06, "cy": cy + img_h * 0.06,
                          "zoom": min_zoom}},
        lambda: {"type": "pull",  # 拉出
                 "start": {"cx": sal_x, "cy": sal_y, "zoom": max_zoom * 0.8},
                 "end": {"cx": cx, "cy": cy, "zoom": min_zoom}},
        lambda: {"type": "hold_drift",  # 缓慢横漂
                 "start": {"cx": cx - img_w * 0.08, "cy": cy, "zoom": mid_zoom * 0.95},
                 "end": {"cx": cx + img_w * 0.08, "cy": cy, "zoom": mid_zoom * 0.95}},
    ]

    segments = []
    for i in range(n_segs):
        pattern_fn = patterns[i % len(patterns)]
        move = pattern_fn()
        # 相邻段反向，避免跳变：上一段的终点接近下一段的起点
        if i > 0 and segments:
            prev_end = segments[-1]["move"]["end"]
            move["start"]["cx"] = prev_end["cx"]
            move["start"]["cy"] = prev_end["cy"]
            move["start"]["zoom"] = prev_end["zoom"]
        segments.append({"move": move, "duration": seg_dur})

    return segments


def gen_motion(image_path: str, duration: float, output: str,
               w: int = 1920, h: int = 1080, fps: int = 30,
               gravity: str = "center") -> str:
    """
    图片 → 有运镜的视频片段。
    短场景单段运镜，长场景自动拆成多段交替运镜。

    参数:
      image_path  输入图片路径
      duration    视频时长（秒）
      output      输出 mp4 路径
      w, h        输出分辨率
      fps         帧率
      gravity     裁切重心（供静态回退用）

    返回运镜类型字符串。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    img_h, img_w = image.shape[:2]

    # 图片太小时上采样，确保运镜空间
    min_w = w * 3
    min_h = h * 2
    if img_w < min_w or img_h < min_h:
        scale = max(min_w / img_w, min_h / img_h)
        image = cv2.resize(image, (int(img_w * scale), int(img_h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
        img_h, img_w = image.shape[:2]

    # 显著性分析
    sal = analyze_saliency(image)

    # 拆分运镜段落
    segments = _plan_segments(duration, img_w, img_h, w, h, sal)

    # ffmpeg pipe 渲染
    ffmpeg_args = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        output,
    ]
    process = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    # 排空 stderr 防死锁
    stderr_chunks = []
    def drain():
        while True:
            chunk = process.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)
    threading.Thread(target=drain, daemon=True).start()

    # 逐段渲染
    types = []
    for seg in segments:
        move = seg["move"]
        seg_frames = int(seg["duration"] * fps)
        start = move["start"]
        end = move["end"]
        types.append(move["type"])

        for i in range(seg_frames):
            t = _ease(i / max(seg_frames - 1, 1))

            cx = start["cx"] + (end["cx"] - start["cx"]) * t
            cy = start["cy"] + (end["cy"] - start["cy"]) * t
            zoom = start["zoom"] + (end["zoom"] - start["zoom"]) * t

            crop_h = int(img_h / zoom)
            crop_w = int(crop_h * w / h)

            x1 = int(max(0, min(cx - crop_w / 2, img_w - crop_w)))
            y1 = int(max(0, min(cy - crop_h / 2, img_h - crop_h)))

            cropped = image[y1:y1+crop_h, x1:x1+crop_w]
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
            process.stdin.write(resized.tobytes())

    process.stdin.close()
    process.wait()

    if process.returncode != 0:
        err = b"".join(stderr_chunks).decode(errors="replace")
        raise RuntimeError(f"ffmpeg 错误: {err[-300:]}")

    return "+".join(types)
