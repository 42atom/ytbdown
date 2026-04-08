#!/usr/bin/env python3
"""
Ken Burns 智能运镜 — 静态图片 → 动态竖屏视频

用 OpenCV 显著性检测找画面兴趣点，自动规划运镜路径：
  - 起幅：全景或局部
  - 运动：推/拉/摇 向兴趣点
  - 落幅：兴趣点特写或全景收尾

支持多张图片拼接为完整视频。
"""

import math
import os
import subprocess
import sys
import threading

import cv2
import numpy as np
from scipy.interpolate import CubicSpline


# 输出参数（运行时可通过 --ratio 修改）
OUTPUT_W = 1080
OUTPUT_H = 1920  # 9:16
FPS = 30

RATIO_PRESETS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}

# 每张图片的运镜时长（秒）
CLIP_DURATION = 5.0


def analyze_saliency(image: np.ndarray) -> dict:
    """
    分析图片显著性分布，返回重心、分散度、方向等特征。
    用这些特征决定运镜策略。
    """
    h, w = image.shape[:2]
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, mask = saliency.computeSaliency(image)

    result = {"cx": w / 2, "cy": h / 2, "spread": "center", "direction": "none"}

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

    # 二阶矩 → 分散度和方向
    # mu20: x方向方差, mu02: y方向方差, mu11: 协方差
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]

    # 标准差（归一化到图片尺寸）
    std_x = (mu20 ** 0.5) / w
    std_y = (mu02 ** 0.5) / h

    # 偏离中心程度
    off_x = abs(cx - w / 2) / (w / 2)  # 0=中心, 1=边缘
    off_y = abs(cy - h / 2) / (h / 2)

    # 判断分布特征
    if std_x > 0.2 and std_y < 0.15:
        # 水平分散 → 适合横摇
        result["spread"] = "horizontal"
        result["direction"] = "left" if cx < w / 2 else "right"
    elif std_y > 0.2 and std_x < 0.15:
        # 垂直分散 → 适合纵摇
        result["spread"] = "vertical"
    elif std_x > 0.2 and std_y > 0.2:
        # 四散 → 适合全景缓拉
        result["spread"] = "scattered"
    elif off_x > 0.25 or off_y > 0.25:
        # 集中在偏心位置 → 推向兴趣点
        result["spread"] = "off_center"
        # 判断象限
        if cx > w / 2 and cy > h / 2:
            result["direction"] = "bottom_right"
        elif cx < w / 2 and cy > h / 2:
            result["direction"] = "bottom_left"
        elif cx > w / 2:
            result["direction"] = "top_right"
        else:
            result["direction"] = "top_left"
    else:
        # 集中在中心 → 推近
        result["spread"] = "center"

    result["std_x"] = std_x
    result["std_y"] = std_y
    result["off_x"] = off_x
    result["off_y"] = off_y

    return result


def plan_camera_move(
    img_w: int, img_h: int, sal: dict,
) -> dict:
    """
    根据显著性分析结果规划运镜 — 图片内容决定运镜方式。

    spread=horizontal  → 横摇扫过兴趣带
    spread=vertical    → 纵摇
    spread=off_center  → 推向兴趣点
    spread=center      → 缓推近看
    spread=scattered   → 全景缓拉，展示全貌
    """
    cx, cy = img_w / 2, img_h / 2
    sal_x, sal_y = sal["cx"], sal["cy"]
    max_zoom = min(1.8, min(img_w / OUTPUT_W, img_h / OUTPUT_H))
    mid_zoom = (1.0 + max_zoom) / 2
    min_zoom = 1.0
    pan_margin = img_w * 0.25

    spread = sal["spread"]

    if spread == "horizontal":
        # 兴趣水平分散 → 横摇，方向跟着兴趣点重心
        if sal_x > cx:
            # 重心偏右 → 从左摇到右
            return {
                "type": "pan_lr（兴趣水平分散）",
                "start": {"cx": pan_margin, "cy": sal_y, "zoom": mid_zoom},
                "end": {"cx": img_w - pan_margin, "cy": sal_y, "zoom": mid_zoom},
            }
        else:
            return {
                "type": "pan_rl（兴趣水平分散）",
                "start": {"cx": img_w - pan_margin, "cy": sal_y, "zoom": mid_zoom},
                "end": {"cx": pan_margin, "cy": sal_y, "zoom": mid_zoom},
            }

    elif spread == "off_center":
        # 兴趣偏心 → 从对侧推向兴趣点
        # 起幅在兴趣点的对侧
        start_x = cx + (cx - sal_x) * 0.5
        start_y = cy + (cy - sal_y) * 0.5
        return {
            "type": f"push（兴趣偏{sal['direction']}）",
            "start": {"cx": start_x, "cy": start_y, "zoom": min_zoom},
            "end": {"cx": sal_x, "cy": sal_y, "zoom": max_zoom * 0.85},
        }

    elif spread == "center":
        # 兴趣集中在中心 → 缓推特写
        return {
            "type": "push_center（兴趣居中）",
            "start": {"cx": cx, "cy": cy, "zoom": min_zoom},
            "end": {"cx": sal_x, "cy": sal_y, "zoom": max_zoom * 0.85},
        }

    elif spread == "scattered":
        # 兴趣四散 → 从兴趣点特写拉出全景
        return {
            "type": "pull（兴趣分散）",
            "start": {"cx": sal_x, "cy": sal_y, "zoom": max_zoom * 0.7},
            "end": {"cx": cx, "cy": cy, "zoom": min_zoom},
        }

    elif spread == "vertical":
        # 兴趣垂直分散 → 纵摇（竖屏天然适合）
        v_margin = img_h * 0.25
        if sal_y > cy:
            return {
                "type": "tilt_down（兴趣垂直分散）",
                "start": {"cx": sal_x, "cy": v_margin, "zoom": mid_zoom},
                "end": {"cx": sal_x, "cy": img_h - v_margin, "zoom": mid_zoom},
            }
        else:
            return {
                "type": "tilt_up（兴趣垂直分散）",
                "start": {"cx": sal_x, "cy": img_h - v_margin, "zoom": mid_zoom},
                "end": {"cx": sal_x, "cy": v_margin, "zoom": mid_zoom},
            }

    # 兜底：对角线漂移
    return {
        "type": "drift",
        "start": {"cx": cx - img_w * 0.1, "cy": cy - img_h * 0.1, "zoom": min_zoom},
        "end": {"cx": sal_x, "cy": sal_y, "zoom": mid_zoom},
    }


def ease_in_out(t: float) -> float:
    """缓入缓出曲线"""
    return t * t * (3 - 2 * t)


def render_clip(
    image: np.ndarray, move: dict, duration: float, output_path: str,
):
    """渲染单张图片的运镜视频"""
    img_h, img_w = image.shape[:2]
    total_frames = int(duration * FPS)

    start = move["start"]
    end = move["end"]

    # ffmpeg pipe
    ffmpeg_args = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{OUTPUT_W}x{OUTPUT_H}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    process = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # 排空 stderr
    stderr_chunks = []
    def drain():
        while True:
            chunk = process.stderr.read(4096)
            if not chunk: break
            stderr_chunks.append(chunk)
    threading.Thread(target=drain, daemon=True).start()

    for i in range(total_frames):
        t = ease_in_out(i / max(total_frames - 1, 1))

        # 插值当前帧参数
        cx = start["cx"] + (end["cx"] - start["cx"]) * t
        cy = start["cy"] + (end["cy"] - start["cy"]) * t
        zoom = start["zoom"] + (end["zoom"] - start["zoom"]) * t

        # 裁切框尺寸（zoom=1 时显示最大范围）
        crop_h = int(img_h / zoom)
        crop_w = int(crop_h * OUTPUT_W / OUTPUT_H)

        # 边界钳制
        x1 = int(max(0, min(cx - crop_w / 2, img_w - crop_w)))
        y1 = int(max(0, min(cy - crop_h / 2, img_h - crop_h)))

        # 裁切 + 缩放到输出尺寸
        cropped = image[y1:y1+crop_h, x1:x1+crop_w]
        resized = cv2.resize(cropped, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LANCZOS4)
        process.stdin.write(resized.tobytes())

    process.stdin.close()
    process.wait()

    if process.returncode != 0:
        print(f"ffmpeg 错误: {b''.join(stderr_chunks).decode(errors='replace')[-300:]}")


def concat_clips(clip_paths: list[str], output_path: str):
    """拼接多个视频片段"""
    list_file = output_path + ".list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path,
    ], capture_output=True)
    os.remove(list_file)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image1> [image2 ...] [--duration=N] [--ratio=16:9|9:16|1:1|4:5]")
        print(f"  --duration=N  每张图运镜时长，默认 {CLIP_DURATION}s")
        print(f"  --ratio=      输出比例，默认 9:16")
        sys.exit(1)

    images = []
    duration = CLIP_DURATION

    global OUTPUT_W, OUTPUT_H
    for arg in sys.argv[1:]:
        if arg.startswith("--duration="):
            duration = float(arg.split("=", 1)[1])
        elif arg.startswith("--ratio="):
            ratio_str = arg.split("=", 1)[1]
            if ratio_str in RATIO_PRESETS:
                OUTPUT_W, OUTPUT_H = RATIO_PRESETS[ratio_str]
            else:
                print(f"不支持的比例: {ratio_str}，可选: {' '.join(RATIO_PRESETS)}")
                sys.exit(1)
        else:
            images.append(arg)

    if not images:
        print("请指定至少一张图片")
        sys.exit(1)

    clip_paths = []

    for idx, img_path in enumerate(images):
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取: {img_path}")
            continue

        h, w = image.shape[:2]

        # 图片太小时上采样，确保有足够的运镜空间
        # 至少需要宽度 = OUTPUT_W * 3 才能有舒服的平移范围
        min_w = OUTPUT_W * 3
        min_h = OUTPUT_H * 2
        if w < min_w or h < min_h:
            scale = max(min_w / w, min_h / h)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
            h, w = image.shape[:2]
            print(f"  上采样至 {w}x{h}")
        sal = analyze_saliency(image)
        move = plan_camera_move(w, h, sal)

        print(f"[{idx+1}/{len(images)}] {os.path.basename(img_path)} ({w}x{h})")
        print(f"  兴趣点: ({sal['cx']:.0f}, {sal['cy']:.0f}), 分布: {sal['spread']}, std=({sal.get('std_x',0):.2f}, {sal.get('std_y',0):.2f})")
        print(f"  运镜: {move['type']} | 缩放 {move['start']['zoom']:.1f}x → {move['end']['zoom']:.1f}x")

        clip_path = img_path.rsplit(".", 1)[0] + "_kenburns.mp4"
        render_clip(image, move, duration, clip_path)
        clip_paths.append(clip_path)
        print(f"  输出: {clip_path}")

    # 多张图片时拼接
    if len(clip_paths) > 1:
        output = os.path.join(
            os.path.dirname(os.path.abspath(images[0])),
            "kenburns_combined.mp4"
        )
        concat_clips(clip_paths, output)
        size_mb = os.path.getsize(output) / 1024 / 1024
        print(f"\n合并输出: {output} ({size_mb:.1f}MB)")
    elif clip_paths:
        size_mb = os.path.getsize(clip_paths[0]) / 1024 / 1024
        print(f"\n完成: {clip_paths[0]} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
