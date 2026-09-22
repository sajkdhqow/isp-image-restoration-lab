# -*- coding: utf-8 -*-
"""
EE —— 边缘增强 / 锐化 (Edge Enhance / Sharpen)
================================================

对应视频: 《ISP算法精讲——EE》

【为什么需要】
1. 镜头本质是低通滤波器(衍射极限 + 像差), 边缘过渡被拉缓;
2. pipeline 中的 demosaic、降噪也是低通特性, 进一步损失高频;
   -> 成像锐度下降, 需要在 YUV 域对亮度通道做补偿。

【原理】
unsharp masking(反锐化掩模):
    Y_out = Y + λ · (Y - G(Y))
    G(Y): 低通(高斯)后的模糊图, (Y - G(Y)) 即高频/边缘分量。
视频里强调的两个优化点:
1. 边缘判决器: 不是所有像素都要锐化 —— 用梯度阈值区分强边缘、
   弱纹理、平坦区, 平坦区锐化只会放大噪声(颗粒感);
2. λ 自适应: 强边缘给大 λ, 噪声/平坦区给小 λ;
   另外要 clamp 过冲(overshoot), 防止边缘两侧出现"白边/黑边"
   (振铃 ringing), 有的方案限制输出不超过原图一定百分比。

【面试要点】
- 为什么在 YUV 域的 Y 上做: 锐化 UV 会把彩噪也锐出来, 产生彩色镶边;
- 锐化过强的画质问题: 白边(halo)、锯齿感、噪点被放大 —— 调参时
  "细节增益 vs 噪声放大"的平衡是 IQ 调优高频话题;
- 与降噪的先后: UVNR -> EE, 先把噪声压住再锐化, 否则锐化放大噪声;
- 3D-LUT/directional sharpening(按方向自适应核)是进阶方案。
"""

import numpy as np

from .nr import gaussian_filter


def ee_sharpen(yuv, lam=0.5, sigma=1.0, edge_thr=8.0, edge_gain=2.0,
               max_over_ratio=0.25):
    """自适应反锐化掩模。

    Parameters
    ----------
    yuv : (H, W, 3) YUV 图(0~255 或任意线性范围)
    lam : float
        基础锐化强度。
    sigma : float
        高斯低通核宽, 越小锐化的细节尺度越细。
    edge_thr : float
        边缘判决梯度阈值, 低于它的区域视为平坦/噪声区。
    edge_gain : float
        强边缘区 λ 的放大倍数。
    max_over_ratio : float
        输出相对原图的最大过冲比例, 抑制白边。
    """
    yuv = np.asarray(yuv, np.float32)
    y = yuv[..., 0]
    blur = gaussian_filter(y, ksize=5, sigma=sigma)
    detail = y - blur

    # 边缘判决: 梯度幅值(简化用 detail 绝对值 + sobel)
    gx = np.zeros_like(y); gy = np.zeros_like(y)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    from scipy.ndimage import convolve
    gx = convolve(y, kx, mode="nearest")
    gy = convolve(y, ky, mode="nearest")
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # λ 自适应: 平坦区 -> 0, 强边缘 -> lam * edge_gain
    t = np.clip((grad - edge_thr) / (edge_thr * 2), 0, 1)
    lam_map = lam * (1.0 + (edge_gain - 1.0) * t)

    sharp = y + lam_map * detail

    # 过冲 clamp: 限制相对原图的偏移
    lim = max_over_ratio * np.maximum(np.abs(y), 1.0)
    delta = np.clip(sharp - y, -lim, lim)
    y_out = y + delta

    return np.stack([y_out, yuv[..., 1], yuv[..., 2]], axis=-1)
