# -*- coding: utf-8 -*-
"""
BNR —— Bayer 域(RAW域)降噪 (Bayer Noise Reduction)
====================================================

对应视频: 《ISP算法精讲——BNR》及其中三篇论文的讲解:
1. PCA-Based Spatially Adaptive Denoising of CFA Images for
   Single-Sensor Digital Cameras (Zhang & Gunturk)
2. PSEUDO FOUR-CHANNEL IMAGE DENOISING FOR NOISY CFA RAW DATA (Hirakawa)
3. Noise Reduction for CFA Image Sensors Exploiting HVS Behaviour

【为什么要在 RAW 域降噪】
1. RAW 上噪声模型最简单: 近似 Poisson-Gaussian, sigma^2 = a*I + b,
   亮部散粒噪声大, 暗部读出噪声占主导;
2. pipeline 后级(demosaic 插值、CCM、gamma、锐化)会改变噪声模型:
   插值让噪声空间相关, CCM 放大色通道噪声, gamma 提亮暗部,
   越晚降噪越难且代价越大;
3. RGB 域降噪容易在边缘产生彩色伪影, RAW 域单通道没有这个问题。

【实现】
1. 伪四通道法 (Hirakawa): 把 Bayer 拆成 R/Gr/Gb/B 四个半分辨率平面,
   视作"四通道图像"逐通道降噪, 破坏了 Bayer 结构带来的空间欠采样问题,
   降噪后重组。本实现配合自适应 Wiener 滤波(局部均值/方差 + 噪声方差):
       out = mean + max(var - sigma^2, 0) / (var + eps) * (x - mean)
2. HVS 行为加权: 人眼对暗部噪声更敏感(暗部中人眼对比灵敏度低但噪声
   相对可见度高), 对平坦区噪声也远比纹理区敏感。用局部活动度(方差)
   生成权重图, 平坦区强降噪、纹理/边缘区弱降噪。
3. 噪声水平估计: 用 1/1 差分(MAD) 估计 sigma, 或从标定的噪声模型
   sigma^2 = a*I + b 查表。

【面试要点】
1. 噪声来源: 光子散粒噪声(泊松)、读出噪声(高斯)、暗电流噪声、
  复位噪声; 增益越大散粒噪声占比越高。
2. "为什么不在 RGB 域一次降噪到位" —— 噪声被 pipeline 前级改变,
   空间相关+彩色化, 保边降噪难度陡增。
3. 双边/NLM 等保边滤波器在 RAW 上跑四通道比在 RGB 上便宜, 硬件友好。
"""

import numpy as np


def estimate_noise_sigma(img):
    """Immerkær 快速噪声估计: 用拉普拉斯核差分 + MAD。

    L = [[1,-2,1],[-2,4,-2],[1,-2,1]], sigma = sqrt(pi/2) / 6 * MAD(|L*I|)
    对 Bayer 图直接使用会受颜色差异影响, demo 中按通道拆分后估计。
    """
    lap = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    from scipy.ndimage import convolve
    d = np.abs(convolve(img, lap, mode="nearest"))
    mad = np.median(d)
    return float(np.sqrt(np.pi / 2) / 6.0 * mad)


def _adaptive_wiener(ch, noise_sigma, win=5):
    """局部自适应 Wiener 滤波: 平坦区平滑, 纹理区保留。"""
    from scipy.ndimage import uniform_filter

    eps = 1e-8
    mean = uniform_filter(ch, size=win, mode="nearest")
    var = uniform_filter((ch - mean) ** 2, size=win, mode="nearest")
    # 局部活动度权重: 方差越接近噪声水平越平坦 -> 越强平滑
    factor = np.clip(1.0 - noise_sigma ** 2 / (var + eps), 0.0, 1.0)
    return mean + factor * (ch - mean)


def bnr_pseudo4ch(raw, noise_sigma=None, bayer_format="RGGB",
                  max_val=1023.0):
    """伪四通道 RAW 降噪。

    拆 R/Gr/Gb/B 四通道 -> 各自自适应 Wiener -> 重组。
    noise_sigma 为 None 时自动估计。
    """
    from .utils import split_bayer, merge_bayer

    if noise_sigma is None:
        # G 通道像素最多, 用它估计最具代表性
        _, Gr, _, _ = split_bayer(raw, bayer_format)
        noise_sigma = max(estimate_noise_sigma(Gr), 1e-3)

    R, Gr, Gb, B = split_bayer(raw, bayer_format)
    # 不同通道噪声水平随增益不同, 这里用同一 sigma 近似
    R = _adaptive_wiener(R, noise_sigma)
    Gr = _adaptive_wiener(Gr, noise_sigma)
    Gb = _adaptive_wiener(Gb, noise_sigma)
    B = _adaptive_wiener(B, noise_sigma)
    out = merge_bayer(R, Gr, Gb, B, bayer_format)
    return np.clip(out, 0, max_val)


def _activity_weight(ch, win=9):
    """局部活动度(归一化方差), 用于 HVS 加权: 越平坦权重越接近 1。"""
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(ch, size=win, mode="nearest")
    var = uniform_filter((ch - mean) ** 2, size=win, mode="nearest")
    act = np.sqrt(var)
    # 归一化: 活动度低于 2 个像素值波动认为平坦
    return np.clip(1.0 - act / (act.max() + 1e-6), 0.0, 1.0)


def bnr_hvs(raw, noise_sigma=None, strength=1.0, bayer_format="RGGB",
            max_val=1023.0):
    """HVS 加权 RAW 降噪: 平坦区强降噪 + 纹理区弱降噪。

    在伪四通道 Wiener 结果与原图之间按活动度权重做凸组合,
    strength 控制整体降噪力度。
    """
    from .utils import split_bayer, merge_bayer

    if noise_sigma is None:
        _, Gr, _, _ = split_bayer(raw, bayer_format)
        noise_sigma = max(estimate_noise_sigma(Gr), 1e-3)

    den = bnr_pseudo4ch(raw, noise_sigma, bayer_format, max_val)
    _, Gr, _, _ = split_bayer(raw, bayer_format)
    w_flat = _activity_weight(Gr)
    # 半分辨率权重放大到全图(bayer 中 G 占一半, 简单用 repeat 融合)
    w_full = np.repeat(np.repeat(w_flat, 2, axis=0), 2, axis=1)
    w_full = w_full[:raw.shape[0], :raw.shape[1]]
    out = raw + strength * w_full * (den - raw)
    return np.clip(out, 0, max_val)
