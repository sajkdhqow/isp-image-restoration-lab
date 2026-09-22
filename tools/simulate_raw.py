# -*- coding: utf-8 -*-
"""
模拟 RAW 生成器 —— 从 sRGB 图像生成带物理缺陷的 Bayer RAW
==========================================================

没有真实 sensor RAW 也能完整演示 ISP pipeline。模拟的缺陷与真实
成像一一对应, 方便逐模块验证:

1. 色偏: sensor 光谱响应与 sRGB 不同 -> 线性域乘 SENSOR_COLOR_DISTORTION
2. 渐晕(luma shading): 边缘按 cos^4 类规律变暗
3. 色渐晕(color shading): 边缘 R/B 相对 G 额外衰减 -> 边缘偏绿
4. 暗电流/黑电平: 整体抬高 BL(默认 64, 10bit)
5. 散粒+读出噪声: sigma = sqrt(sigma_read^2 + shot * I)
6. 坏点: 随机 dead(暗点) / hot(亮点) 像素
7. 可选 MIPI RAW10 打包输出
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isp.pipeline import SENSOR_COLOR_DISTORTION  # noqa: E402
from isp.utils import merge_bayer  # noqa: E402


def make_synthetic_scene(h=512, w=512, seed=0):
    """生成一张测试场景: 色卡 + 渐变背景 + 几何图形 + 高对比边缘。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 背景: 对角渐变
    bg = 0.25 + 0.35 * (xx / w) + 0.2 * (yy / h)
    img = np.stack([bg * 0.9, bg, bg * 1.1], axis=-1)

    # 色块区(上左): 8 个饱和色块
    colors = [(0.85, 0.2, 0.15), (0.2, 0.75, 0.25), (0.2, 0.3, 0.9),
              (0.95, 0.85, 0.2), (0.9, 0.4, 0.85), (0.3, 0.9, 0.9),
              (0.9, 0.55, 0.2), (0.6, 0.3, 0.6)]
    for i, c in enumerate(colors):
        r0 = 20 + (i % 4) * 90
        c0 = 20 + (i // 4) * 90
        img[r0:r0 + 80, c0:c0 + 80] = c

    # 圆形(右下) 与棋盘格(中下) 检验边缘
    cy, cx, rad = h - 120, w - 120, 80
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2
    img[mask] = (0.9, 0.9, 0.9)
    chess_size = 16
    for i in range(0, 160, chess_size):
        for j in range(0, 160, chess_size):
            if (i // chess_size + j // chess_size) % 2 == 0:
                img[250 + i:250 + i + chess_size,
                    280 + j:280 + j + chess_size] = 0.05
    # 细线条
    img[60:64, 200:460] = 0.95
    img[64:68, 200:460] = 0.05

    img += rng.normal(0, 0.005, img.shape)  # 轻微纹理
    return np.clip(img, 0, 1)


def rgb_to_bayer(rgb_linear, bayer_format="RGGB"):
    """sRGB 图(线性域) -> Bayer 马赛克。"""
    h, w, _ = rgb_linear.shape
    bayer = np.zeros((h, w), dtype=np.float32)
    if bayer_format == "RGGB":
        bayer[0::2, 0::2] = rgb_linear[0::2, 0::2, 0]  # R
        bayer[0::2, 1::2] = rgb_linear[0::2, 1::2, 1]  # Gr
        bayer[1::2, 0::2] = rgb_linear[1::2, 0::2, 1]  # Gb
        bayer[1::2, 1::2] = rgb_linear[1::2, 1::2, 2]  # B
    elif bayer_format == "GBRG":
        bayer[0::2, 0::2] = rgb_linear[0::2, 0::2, 1]
        bayer[0::2, 1::2] = rgb_linear[0::2, 1::2, 2]
        bayer[1::2, 0::2] = rgb_linear[1::2, 0::2, 0]
        bayer[1::2, 1::2] = rgb_linear[1::2, 1::2, 1]
    elif bayer_format == "GRBG":
        bayer[0::2, 0::2] = rgb_linear[0::2, 0::2, 1]
        bayer[0::2, 1::2] = rgb_linear[0::2, 1::2, 0]
        bayer[1::2, 0::2] = rgb_linear[1::2, 0::2, 2]
        bayer[1::2, 1::2] = rgb_linear[1::2, 1::2, 1]
    elif bayer_format == "BGGR":
        bayer[0::2, 0::2] = rgb_linear[0::2, 0::2, 2]
        bayer[0::2, 1::2] = rgb_linear[0::2, 1::2, 1]
        bayer[1::2, 0::2] = rgb_linear[1::2, 0::2, 1]
        bayer[1::2, 1::2] = rgb_linear[1::2, 1::2, 0]
    return bayer


def simulate_raw(rgb_srgb, bits=10, black_level=64.0, analog_gain=1.0,
                 exposure_scale=1.0,
                 vignette=0.45, color_shading=0.25,
                 dead_ratio=3e-4, hot_ratio=2e-4,
                 sigma_read=2.5, shot=0.03,
                 apply_color_distortion=True,
                 bayer_format="RGGB", seed=0):
    """从 sRGB 图像(0~255 uint8)生成带缺陷的 RAW。

    Parameters
    ----------
    exposure_scale : float
        模拟曝光时间: 场景亮度线性于曝光量(t * 增益), 真实 AE 闭环
        就是通过调整它和 analog_gain 控制画面亮度的。>1 时高光可能
        过饱和 clip, 这正是 AE 高光保护要处理的情况。

    Returns
    -------
    raw : (H, W) float array, 范围 [0, 2^bits - 1]
    """
    rng = np.random.default_rng(seed)
    h, w, _ = rgb_srgb.shape
    max_val = float((1 << bits) - 1)

    lin = (rgb_srgb.astype(np.float32) / 255.0) ** 2.2  # 解 gamma 回线性
    lin = lin * exposure_scale  # 模拟曝光时间(亮度线性于曝光量)

    # 1) sensor 色偏
    if apply_color_distortion:
        lin = np.einsum("ij,hwj->hwi", SENSOR_COLOR_DISTORTION, lin)

    # 2) luma shading: cos^4 类渐晕
    yy = (np.arange(h) - (h - 1) / 2) / ((h - 1) / 2)
    xx = (np.arange(w) - (w - 1) / 2) / ((w - 1) / 2)
    r2 = yy[:, None] ** 2 + xx[None, :] ** 2
    falloff = 1.0 - vignette * r2
    lin = lin * falloff[..., None]

    # 3) color shading: 边缘 R/B 相对 G 衰减 -> 偏绿
    cb = 1.0 - color_shading * r2
    lin[..., 0] *= cb
    lin[..., 2] *= cb

    # 4) CFA 马赛克
    raw = rgb_to_bayer(lin, bayer_format)

    # 5) 增益 + 黑电平
    raw = raw * analog_gain * max_val + black_level

    # 6) 噪声: 读出(常数 sigma) + 散粒(随信号)
    sigma = np.sqrt(sigma_read ** 2 + shot * np.maximum(raw - black_level, 0))
    raw = raw + rng.normal(0, 1, raw.shape) * sigma

    # 7) 坏点
    n_dead = int(h * w * dead_ratio)
    n_hot = int(h * w * hot_ratio)
    for _ in range(n_dead):
        raw[rng.integers(0, h), rng.integers(0, w)] = \
            rng.uniform(0, black_level * 0.5)
    for _ in range(n_hot):
        raw[rng.integers(0, h), rng.integers(0, w)] = \
            rng.uniform(max_val * 0.9, max_val)

    return np.clip(raw, 0, max_val)
