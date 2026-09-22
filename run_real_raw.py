# -*- coding: utf-8 -*-
"""
真实 RAW 文件直通 demo —— sensor dump -> ISP -> 输出图
======================================================

main.py 走的是"场景图 -> 合成 RAW"路径; 本脚本补上另一条完整路径:

    RAW 文件(裸 dump/MIPI 打包) -> 参数解析 -> 黑电平估计 -> pipeline -> sRGB 输出

真实裸 RAW 没有 header, 宽/高/位深/Bayer 排列必须外部指定(真实平台
由 sensor 驱动提供); 黑电平的三种来源在本脚本全部可演示:
  1. 标定值(OTP):     --black-level 64
  2. 黑帧统计:        --dark-frame dark.raw (盖镜头盖拍的帧)
  3. 光学黑区(OB)行:  --ob-rows 4 (sensor 预留的被遮蔽行)

用法:
    python run_real_raw.py                          # 默认处理 data/sim_raw10.raw
    python run_real_raw.py --input your.raw --width 4096 --height 3072
    python run_real_raw.py --mipi                   # MIPI RAW10 打包格式
    python run_real_raw.py --ob-rows 4              # 从 OB 行估计黑电平

注意: 本仓库的 AWB/CCM 参数是按仿真 sensor 配的, 拿真实 sensor 的
RAW 跑会偏色(LSC/CCM 需按真机重新标定) —— 这本身就是面试要点:
ISP 参数是"每台模组标定一套", 算法框架才是通用的。
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from isp.pipeline import run_pipeline
from isp.utils import read_raw, mipi_raw10_unpack, save_rgb, save_gray
from isp.lsc import calibrate_lsc_grid
from isp.blc import estimate_black_level, estimate_black_level_from_ob
from tools.simulate_raw import simulate_raw, make_synthetic_scene


def parse_args():
    p = argparse.ArgumentParser(description="RAW file -> ISP -> sRGB")
    p.add_argument("--input", default=os.path.join(ROOT, "data", "sim_raw10.raw"),
                   help="裸 RAW 文件路径(未打包平面格式)")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--bits", type=int, default=10, choices=[8, 10, 12, 16])
    p.add_argument("--bayer", default="RGGB",
                   choices=["RGGB", "BGGR", "GRBG", "GBRG"])
    p.add_argument("--mipi", action="store_true",
                   help="输入是 MIPI RAW10 打包格式(默认未打包)")
    p.add_argument("--black-level", type=float, default=None,
                   help="标定黑电平(OTP); 不给则走估计流程")
    p.add_argument("--dark-frame", default=None,
                   help="黑帧 RAW 文件(盖镜头盖拍), 用它分通道估计黑电平")
    p.add_argument("--ob-rows", type=int, default=None,
                   help="光学黑区行数, 从输入帧顶部 OB 行估计黑电平")
    return p.parse_args()


def load_black_level(args, raw, max_val):
    """按优先级确定黑电平: 显式标定值 > 黑帧 > OB 行 > 默认值。"""
    if args.black_level is not None:
        print(f"[BLC] 标定值(OTP): BL={args.black_level:.1f}")
        return args.black_level
    if args.dark_frame:
        dark = read_raw(args.dark_frame, args.width, args.height, args.bits)
        per = estimate_black_level(dark, args.bayer)
        print(f"[BLC] 黑帧估计(分通道): "
              + " ".join(f"{k}={v:.1f}" for k, v in per.items()))
        return float(np.mean(list(per.values())))
    if args.ob_rows:
        per = estimate_black_level_from_ob(raw, args.ob_rows, args.bayer)
        print(f"[BLC] OB 行估计(分通道, 顶 {args.ob_rows} 行): "
              + " ".join(f"{k}={v:.1f}" for k, v in per.items()))
        return float(np.mean(list(per.values())))

    # 都没给: 演示黑帧法 —— 用仿真器造一张黑帧(真实系统就是盖盖子拍一张),
    # 估计值应接近真值 64, 展示估计精度
    dark = simulate_raw(np.zeros((args.height, args.width, 3), np.uint8),
                        bits=args.bits, black_level=64.0, dead_ratio=0,
                        hot_ratio=0, seed=99)
    per = estimate_black_level(dark, args.bayer)
    print(f"[BLC] 黑帧估计(仿真黑帧, 分通道): "
          + " ".join(f"{k}={v:.1f}" for k, v in per.items())
          + "  (真值 64)")
    return float(np.mean(list(per.values())))


def main():
    args = parse_args()
    max_val = float((1 << args.bits) - 1)
    # 0) 默认输入缺失时自动生成(clone 后无需先跑 main.py)
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    default_raw = os.path.join(ROOT, "data", "sim_raw10.raw")
    if not os.path.exists(args.input):
        if args.input != default_raw:
            sys.exit(f"input not found: {args.input}")
        print("[GEN] 默认 RAW 不存在, 先生成 data/sim_raw10.raw ...")
        import cv2
        from isp.utils import imwrite_unicode
        os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
        scene_g = make_synthetic_scene(512, 512)
        scene_u8 = (np.clip(scene_g, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        imwrite_unicode(os.path.join(ROOT, "data", "scene.png"),
                        cv2.cvtColor(scene_u8, cv2.COLOR_BGR2RGB))
        r = simulate_raw(scene_u8, bits=args.bits, black_level=64.0,
                         analog_gain=0.72, seed=42)
        r.astype(np.uint16).tofile(args.input)

    print(f"RAW: {args.input}  {args.width}x{args.height} "
          f"{args.bits}bit {args.bayer}")

    # 1) 读文件: 裸 dump 或 MIPI 打包
    if args.mipi:
        with open(args.input, "rb") as f:
            packed = f.read()
        raw = mipi_raw10_unpack(np.frombuffer(packed, np.uint8),
                                args.width, args.height).astype(np.float32)
        print(f"[LOAD] MIPI RAW10 解包: {len(packed)} bytes -> "
              f"{raw.shape}")
    else:
        raw = read_raw(args.input, args.width, args.height, args.bits)
        print(f"[LOAD] 裸 RAW 读取: 值域 [{raw.min():.0f}, {raw.max():.0f}]")

    # 2) 黑电平: 三种来源演示
    bl = load_black_level(args, raw, max_val)

    # 3) LSC 增益表: 白场标定(产线流程, 拍均匀白场)
    white_raw = simulate_raw(np.full((512, 512, 3), 205, np.uint8),
                             bits=args.bits, black_level=bl,
                             analog_gain=0.72, dead_ratio=0, hot_ratio=0,
                             sigma_read=0.5, shot=0.0, seed=7)
    c = 256
    lsc_target = float(np.median(white_raw[c - 32:c + 32, c - 32:c + 32]))
    lsc_gains = calibrate_lsc_grid(white_raw, lsc_target, grid=(33, 33),
                                   max_gain=16.0)
    print(f"[LSC] 白场标定: target={lsc_target:.0f}")

    # 4) 完整 pipeline
    out, stages = run_pipeline(raw.copy(), bayer_format=args.bayer,
                               max_val=max_val, black_level=bl,
                               lsc_gains=lsc_gains, capture_stages=True)
    gains = stages.get("05_awb_gains")
    if gains:
        print(f"[AWB] R={gains[0]:.3f} B={gains[1]:.3f}")
    save_gray(os.path.join(ROOT, "output", "real_raw_input.png"),
              raw, args.bits)
    save_rgb(os.path.join(ROOT, "output", "real_raw_output.png"), out)
    for name in ("02_blc", "06_demosaic", "12_final"):
        if name in stages:
            img = stages[name]
            if img.ndim == 2:
                save_gray(os.path.join(ROOT, "output",
                          f"real_raw_stage_{name}.png"), img, args.bits)
            else:
                save_rgb(os.path.join(ROOT, "output",
                         f"real_raw_stage_{name}.png"), img)
    print("[OUT] output/real_raw_output.png (+ 中间 stage 图)")


if __name__ == "__main__":
    main()
