#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO26n 装甲板模型 (best.pt) 视频推理脚本
=========================================

功能对应作业要求：
    1. 加载模型权重，读取自己拍摄的视频；
    2. 逐帧对视频执行推理；
    3. 检测到目标时，绘制"检测框 + 类别 + 置信度 + 该目标的 4 个关键点"；
    4. 保存与原视频逐帧对应、可正常播放的推理结果视频。

说明：
    * 关键点全部来自模型预测结果 results[0].keypoints，**没有**用检测框的四个角代替；
    * 模型路径 / 视频路径 / 输出路径既可以用命令行参数修改，也可以直接改下面的默认配置；
    * 支持 CPU 与 GPU（--device auto / cpu / 0）；
    * 不要求实时，CPU 也能跑完。

用法示例（PyCharm 中直接 Shift+F10 使用默认配置；命令行可覆盖任意参数）：
    python infer.py
    python infer.py --source videos/eb3f6c6e19e16dca3b5c634497e6af2c.mp4 --output outputs/result.mp4
    python infer.py --device cpu --conf 0.35 --show
    python infer.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ============================================================================
# 一、默认配置（不想敲命令行时，直接改这里这三行路径即可）
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = str(PROJECT_DIR / "best.pt")                        # 模型权重
DEFAULT_SOURCE = str(PROJECT_DIR / "videos" / "eb3f6c6e19e16dca3b5c634497e6af2c.mp4")  # 自己拍摄的视频
DEFAULT_OUTPUT = str(PROJECT_DIR / "outputs" / "eb3f6c6e19e16dca3b5c634497e6af2c.mp4")  # 结果视频

DEFAULT_CONF = 0.25      # 置信度阈值
DEFAULT_IOU = 0.70       # NMS 的 IoU 阈值
DEFAULT_IMGSZ = 640      # 推理输入尺寸
DEFAULT_DEVICE = "auto"  # auto: 有 GPU 用 GPU，否则 CPU；也可写 cpu / 0 / 1
DEFAULT_KPT_CONF = 0.25  # 关键点置信度阈值（仅当模型输出带关键点置信度时生效）

# 每个类别的颜色 (BGR)：B 组偏蓝、R 组偏红，同一目标框与关键点使用同一颜色，便于看出归属
CLASS_COLORS = {
    "B1": (255, 128, 0),
    "B2": (255, 64, 0),
    "B3": (255, 190, 40),
    "R1": (0, 0, 230),
    "R2": (0, 64, 255),
    "R3": (0, 140, 255),
}
FALLBACK_COLORS = [
    (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 255, 0), (255, 128, 255), (128, 128, 255), (0, 200, 200),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================================
# 二、命令行参数
# ============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26n 装甲板模型视频推理（检测框 + 类别 + 置信度 + 4 个关键点）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python infer.py\n"
            "  python infer.py --source videos/eb3f6c6e19e16dca3b5c634497e6af2c.mp4 --output outputs/result.mp4\n"
            "  python infer.py --device cpu --conf 0.35 --imgsz 960 --show\n"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型权重路径，默认 {DEFAULT_MODEL}")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help=f"输入视频路径（写 0/1 则用摄像头），默认 {DEFAULT_SOURCE}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"输出视频路径，默认 {DEFAULT_OUTPUT}")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"置信度阈值，默认 {DEFAULT_CONF}")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help=f"NMS IoU 阈值，默认 {DEFAULT_IOU}")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help=f"推理输入尺寸，默认 {DEFAULT_IMGSZ}")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help="推理设备：auto / cpu / 0 / 1 …，默认 auto")
    parser.add_argument("--max-det", type=int, default=50, help="单帧最大检测数，默认 50")
    parser.add_argument("--kpt-conf", type=float, default=DEFAULT_KPT_CONF,
                        help="关键点置信度阈值（模型带关键点置信度时生效）")
    parser.add_argument("--codec", default="auto",
                        help="输出编码器：auto / mp4v / avc1 / XVID / MJPG，默认 auto")
    parser.add_argument("--show", action="store_true", help="推理时弹窗实时预览（按 q 退出）")
    parser.add_argument("--save-txt", action="store_true", help="另存检测结果文本（帧号/类别/置信度/框/关键点）")
    parser.add_argument("--max-frames", type=int, default=0, help="仅处理前 N 帧（0 表示全部，便于快速调试）")
    return parser.parse_args(argv)


# ============================================================================
# 三、工具函数
# ============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def color_of(class_id: int, class_name: str) -> tuple:
    """按类别返回绘制颜色（B 组蓝、R 组红，同一目标的所有元素共用一个颜色）。"""
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    return FALLBACK_COLORS[class_id % len(FALLBACK_COLORS)]


def resolve_device(device: str) -> str:
    """把 auto 解析成实际设备，并打印说明。"""
    if str(device).lower() == "auto":
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return str(device)


def resolve_source(source: str) -> tuple:
    """返回 (传给 OpenCV 的 source, 是否摄像头)。"""
    src = str(source)
    if src.isdigit():  # 摄像头编号
        return int(src), True
    path = Path(src)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_DIR / path
    return str(path), False


def default_output_path(source: str) -> str:
    stem = "camera" if str(source).isdigit() else Path(source).stem
    return str(PROJECT_DIR / "outputs" / f"{stem}_result.mp4")


def ensure_even_size(width: int, height: int) -> tuple:
    """H.264/MPEG-4 要求宽高为偶数，否则编码可能失败或花屏，这里补齐 1 像素。"""
    even_w = width + (width % 2)
    even_h = height + (height % 2)
    return even_w, even_h


def open_video_writer(path: str, fps: float, size: tuple, codec: str = "auto"):
    """
    打开结果视频写入器，自动选择本机可用的编码器：
        .mp4 → mp4v（兼容性最好，VLC/PotPlayer/剪辑软件均可播放）
        .avi → MJPG
    返回 (writer, 实际使用的编码器, 实际使用的路径)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".avi", ".mkv"):
        ext = ".mp4"

    if codec == "auto":
        candidates = ["mp4v", "avc1"] if ext != ".avi" else ["MJPG", "XVID"]
    else:
        candidates = [codec]

    for cc in candidates:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            return writer, cc, str(path)
        writer.release()

    # 所有候选都失败时，退回到 .avi + MJPG，保证一定能得到可播放的文件
    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if writer.isOpened():
        log(f"[WARN] 目标格式无法编码，已改用 AVI(MJPG)：{fallback}")
        return writer, "MJPG", str(fallback)
    raise RuntimeError(f"无法创建输出视频：{path}（请检查路径与磁盘空间）")


def put_text(img, text: str, org, scale: float, color=(255, 255, 255), thickness: int = 1) -> None:
    """带黑色描边的文字，保证在亮/暗背景上都清晰可见。"""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_interface(frame, lines) -> None:
    """左下角半透明信息条：模型名、分辨率/设备、进度与推理速度。"""
    h, w = frame.shape[:2]
    base = min(w, h)
    scale = max(0.4, min(1.0, base / 1400.0))
    th = max(1, int(round(base / 600.0)))
    line_h = max(16, int(round(30 * scale)))
    pad = max(6, int(round(10 * scale)))
    bar_h = min(h, line_h * len(lines) + pad)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = h - bar_h + line_h - max(2, int(4 * scale))
    for text in lines:
        put_text(frame, text, (pad, y), scale, (230, 255, 230), th)
        y += line_h


# ============================================================================
# 四、绘制：检测框 + 类别/置信度 + 4 个预测关键点
# ============================================================================

def draw_one_target(frame, box, kpts, kpt_confs, class_id, class_name, score,
                    target_index: int, kpt_conf_thres: float) -> None:
    """
    绘制单个目标：
        * 检测框（类别颜色）
        * 类别名 + 置信度 + 目标编号（如 "T1 R2 0.93"）
        * 4 个关键点：实心圆点 + 白色描边 + 序号 1~4 + 连线四边形
    关键点坐标 kpts 来自 model.predict 的 results.keypoints，不是从 box 推出来的。
    """
    h, w = frame.shape[:2]
    base = min(w, h)
    color = color_of(class_id, class_name)
    thickness = max(2, int(round(base / 320.0)))
    radius = max(5, int(round(base / 130.0)))         # 关键点半径，分辨率自适应
    kpt_font = max(0.45, base / 1000.0)

    # ---------- 1. 检测框 ----------
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # ---------- 2. 类别 + 置信度标签 ----------
    label = f"T{target_index} {class_name} {score:.2f}"
    fs = max(0.45, base / 900.0)
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, max(1, thickness - 1))
    ty = y1 - th - 8
    if ty < 0:                     # 框贴近顶部时，标签画在框内侧
        ty = y1 + 2
    lx = min(x1, max(0, w - (tw + 8)))   # 框贴近右侧时标签左移，避免超出画面
    cv2.rectangle(frame, (lx, ty), (lx + tw + 8, ty + th + 8), color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (lx + 4, ty + th + 3), FONT, fs, (0, 0, 0),
                max(1, thickness - 1), cv2.LINE_AA)

    # ---------- 3. 关键点 ----------
    valid_pts = []
    for i in range(len(kpts)):
        kx, ky = float(kpts[i][0]), float(kpts[i][1])
        # 过滤掉模型未给出有效坐标的点（(0,0) 一般表示该关键点不可见）
        if not np.isfinite(kx) or not np.isfinite(ky) or (kx <= 0.5 and ky <= 0.5):
            continue
        visible = True
        if kpt_confs is not None and i < len(kpt_confs):
            visible = float(kpt_confs[i]) >= kpt_conf_thres
        px, py = int(round(kx)), int(round(ky))
        if not (0 <= px < w and 0 <= py < h):
            continue
        if visible:
            valid_pts.append((px, py))

            # 实心圆 + 白色外环 + 黑色细边：任何背景上都醒目
            cv2.circle(frame, (px, py), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), max(2, radius // 2), (255, 255, 255), -1, cv2.LINE_AA)

            # 关键点序号（1~4），白色字 + 黑描边
            put_text(frame, str(i + 1),
                     (px + radius + 3, py - radius - 1),
                     kpt_font, (255, 255, 255), max(1, thickness - 1))
        else:
            # 置信度太低的关键点用灰色空心圈表示"预测到了但不可见"
            cv2.circle(frame, (px, py), radius, (140, 140, 140), 1, cv2.LINE_AA)

    # 把 4 个关键点按顺序连成四边形，视觉上明确"这几个点属于这个目标"
    if len(valid_pts) >= 3:
        pts = np.array(valid_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, color, max(1, thickness - 1), cv2.LINE_AA)


def render_frame(frame, result, names: dict, kpt_conf_thres: float) -> int:
    """把一帧的推理结果画到图上，返回本帧检测到的目标数量。"""
    count = 0
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return 0

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    # === 关键点直接取自模型预测输出（不是检测框的四个角）===
    kpts_all = None
    kconfs_all = None
    if result.keypoints is not None:
        kpts_all = result.keypoints.xy.cpu().numpy()          # (N, K, 2)
        if result.keypoints.conf is not None:                 # 模型若带关键点置信度
            kconfs_all = result.keypoints.conf.cpu().numpy()  # (N, K)

    for i in range(len(boxes)):
        name = names.get(int(classes[i]), str(classes[i])) if isinstance(names, dict) \
            else (names[int(classes[i])] if int(classes[i]) < len(names) else str(classes[i]))
        kpts = kpts_all[i] if (kpts_all is not None and i < len(kpts_all)) else np.zeros((0, 2))
        kconfs = kconfs_all[i] if (kconfs_all is not None and i < len(kconfs_all)) else None
        draw_one_target(frame, boxes[i], kpts, kconfs, int(classes[i]), name,
                        float(scores[i]), i + 1, kpt_conf_thres)
        count += 1
    return count


# ============================================================================
# 五、主流程：加载模型 → 读取视频 → 逐帧推理 → 写结果视频 → 校验
# ============================================================================

def main(argv=None) -> int:
    args = parse_args(argv)

    # ---------- 参数与文件检查 ----------
    model_path = Path(args.model)
    if not model_path.is_absolute() and not model_path.exists():
        model_path = PROJECT_DIR / model_path
    if not model_path.exists():
        log(f"[ERROR] 找不到模型权重：{model_path}")
        log("        请用 --model 指定 best.pt 的路径。")
        return 1

    src, is_camera = resolve_source(args.source)
    if not is_camera and not Path(src).exists():
        log(f"[ERROR] 找不到输入视频：{src}")
        log("        请把自己拍的视频放到该路径，或用 --source 指定视频文件。")
        log("        例如： python infer.py --source videos/eb3f6c6e19e16dca3b5c634497e6af2c.mp4")
        return 1

    output = args.output
    if not output:
        output = default_output_path(args.source)
    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = PROJECT_DIR / out_path

    # 输出路径不能和输入相同，避免读的同时被覆盖
    if not is_camera and out_path.resolve() == Path(src).resolve():
        out_path = out_path.with_name(out_path.stem + "_result" + (out_path.suffix or ".mp4"))
        log(f"[WARN] 输出路径与输入相同，已改为：{out_path}")

    device = resolve_device(args.device)

    # ---------- 1. 加载模型权重 ----------
    log("=" * 78)
    log("YOLO26n 装甲板视频推理")
    log("=" * 78)
    log(f"[1/5] 加载模型：{model_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
    except Exception as exc:
        log(f"[ERROR] 模型加载失败：{exc}")
        return 1

    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    try:
        kpt_shape = model.model.kpt_shape
    except Exception:
        kpt_shape = None
    kpt_count = int(kpt_shape[0]) if kpt_shape else 0
    log(f"      任务类型 : {model.task}")
    log(f"      类别 ({len(names)}) : {', '.join(str(v) for v in names.values())}")
    log(f"      关键点   : {kpt_count if kpt_count else '?'} 个"
        f"（含置信度: {'是' if kpt_shape and kpt_shape[1] == 3 else '否'}）")
    log(f"      推理设备 : {device}")

    # ---------- 2. 读取视频 ----------
    log(f"[2/5] 打开视频：{src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log(f"[ERROR] 无法打开视频：{src}")
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 30.0                      # 个别视频读不到帧率时用 30 兜底
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:            # 摄像头可能暂时拿不到尺寸，先读一帧探测
        ok, probe = cap.read()
        if not ok:
            log("[ERROR] 读取视频帧失败。")
            return 1
        height, width = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    orientation = 0
    if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
        try:
            orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        except Exception:
            orientation = 0
    if orientation:
        log(f"[WARN] 视频带有旋转元数据 ({orientation}°)，本脚本按原始像素保存，"
            f"若播放方向不对请先转正原视频。")

    out_w, out_h = ensure_even_size(width, height)
    log(f"      分辨率 : {width}x{height}"
        + ("" if (out_w, out_h) == (width, height) else f"（输出补齐为 {out_w}x{out_h}）"))
    log(f"      帧率   : {src_fps:.3f} fps")
    log(f"      帧数   : {'未知（摄像头）' if is_camera else total_frames}")
    if args.max_frames:
        log(f"      限制   : 只处理前 {args.max_frames} 帧")

    # ---------- 3. 准备结果视频（帧率/分辨率与原视频保持一致）----------
    log(f"[3/5] 创建结果视频：{out_path}")
    writer, used_codec, out_path_str = open_video_writer(str(out_path), src_fps, (out_w, out_h), args.codec)
    log(f"      编码器 : {used_codec}")

    # 预热一次，避免第一帧特别慢
    try:
        model.predict(np.zeros((out_h, out_w, 3), dtype=np.uint8), imgsz=args.imgsz,
                      device=device, verbose=False)
    except Exception:
        pass

    # ---------- 4. 逐帧推理 ----------
    log("[4/5] 开始逐帧推理 ...")
    txt_lines = []
    class_counter = Counter()
    conf_sum, conf_cnt = 0.0, 0
    frames_read = frames_written = frames_with_det = det_total = 0
    t_start = time.perf_counter()
    last_report = t_start

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_read += 1

        if (out_w, out_h) != (width, height):     # 奇数尺寸补 1 像素
            frame = cv2.copyMakeBorder(frame, 0, out_h - height, 0, out_w - width,
                                       cv2.BORDER_REPLICATE)

        # ---- 单帧推理 ----
        results = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                device=device, max_det=args.max_det, verbose=False)
        result = results[0]

        # ---- 统计 ----
        n = 0 if result.boxes is None else len(result.boxes)
        if n:
            frames_with_det += 1
            det_total += n
            for c, s in zip(result.boxes.cls.cpu().numpy().astype(int),
                            result.boxes.conf.cpu().numpy()):
                class_counter[names.get(int(c), str(c)) if isinstance(names, dict) else str(c)] += 1
                conf_sum += float(s)
                conf_cnt += 1
            if args.save_txt:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                kxy = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
                for i in range(n):
                    row = [str(frames_read), names.get(int(clss[i]), str(clss[i])),
                           f"{float(confs[i]):.4f}"] + [f"{v:.1f}" for v in xyxy[i]]
                    if kxy is not None and i < len(kxy):
                        row += [f"{v:.1f}" for v in kxy[i].reshape(-1)]
                    txt_lines.append(",".join(row))

        # ---- 绘制 ----
        render_frame(frame, result, names, args.kpt_conf)

        # ---- 画面信息条 ----
        elapsed = time.perf_counter() - t_start
        cur_fps = frames_read / elapsed if elapsed > 0 else 0.0
        progress = f" {frames_read}/{total_frames}" if (not is_camera and total_frames > 0) else f" {frames_read}"
        draw_interface(frame, [
            f"YOLO26n armor pose | {model_path.name} | device={device}",
            f"frame{progress} | det={n} | {cur_fps:.1f} FPS",
        ])

        writer.write(frame)
        frames_written += 1

        # ---- 预览 ----
        if args.show:
            cv2.imshow("infer.py - YOLO26n armor pose", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                log("用户按 q，提前结束。")
                break

        # ---- 进度 ----
        now = time.perf_counter()
        if now - last_report >= 2.0:
            last_report = now
            speed = frames_read / (now - t_start)
            eta = ""
            if not is_camera and total_frames > 0:
                eta = f"，预计剩余 {max(0.0, (total_frames - frames_read) / max(speed, 1e-6)):.0f}s"
            log(f"      进度 {frames_read}/{total_frames if total_frames > 0 else '?'} 帧 | "
                f"本帧检出 {n} | 平均 {speed:.1f} FPS{eta}")

        if args.max_frames and frames_read >= args.max_frames:
            log(f"      已达到 --max-frames={args.max_frames}，停止。")
            break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    elapsed_all = time.perf_counter() - t_start

    if frames_written == 0:
        log("[ERROR] 没有写入任何帧，请检查输入视频是否可以正常解码。")
        return 1

    # ---------- 5. 结果校验：帧数、帧率、分辨率是否与原视频对得上 ----------
    log("[5/5] 校验输出视频 ...")
    verified = False
    check = cv2.VideoCapture(out_path_str)
    if check.isOpened():
        v_fps = check.get(cv2.CAP_PROP_FPS)
        v_w = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v_n = 0
        while True:
            ok, _ = check.read()
            if not ok:
                break
            v_n += 1
        check.release()
        verified = v_n == frames_written
        log(f"      文件 : {out_path_str}")
        log(f"      帧数 : {v_n}（写入 {frames_written}）  "
            f"帧率: {v_fps:.3f}（原 {src_fps:.3f}）  分辨率: {v_w}x{v_h}")
        log(f"      结论 : {'可正常解码，与原视频一致' if verified else '帧数与写入不一致，请检查编码器'}")

    if args.save_txt:
        txt_path = Path(out_path_str).with_suffix(".txt")
        kpt_cols = []
        for k in range(kpt_count):
            kpt_cols += [f"kpt{k + 1}x", f"kpt{k + 1}y"]
        header = ",".join(["frame", "class", "conf", "x1", "y1", "x2", "y2"] + kpt_cols)
        txt_path.write_text("\n".join([header] + txt_lines) + "\n", encoding="utf-8")
        log(f"      结果文本: {txt_path}")

    # ---------- 汇总 ----------
    log("-" * 78)
    log(f"处理帧数 : {frames_read}    写入帧数 : {frames_written}    "
        f"有目标帧 : {frames_with_det}    检出目标 : {det_total}")
    if class_counter:
        log("类别统计 : " + ", ".join(f"{k}={v}" for k, v in sorted(class_counter.items())))
        log(f"平均置信度 : {conf_sum / max(conf_cnt, 1):.3f}")
    log(f"耗时     : {elapsed_all:.1f}s（平均 {frames_read / max(elapsed_all, 1e-6):.1f} FPS）")
    log(f"结果视频 : {Path(out_path_str).resolve()}")
    log("=" * 78)
    log("[OK] 推理完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
