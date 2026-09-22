# -*- coding: utf-8 -*-
"""
UVNR —— YUV 域色度降噪 (UV Noise Reduction / CNR)
===================================================

对应视频: 《ISP算法——UVNR》

【背景】
经过 CSC 进入 YUV 域后, 剩下的彩噪(color noise / chroma noise)
集中在 U/V 分量上, 表现为平坦区五颜六色的小色斑。
人眼对色度分辨率远低于亮度 -> 对 U/V 可以"下狠手":
用更强的低通/中值, 甚至配合 4:2:0 亚采样。

【算法】
1. 简单方案: 对 U/V 各做一次小核高斯/中值滤波;
2. 边缘保持方案(本实现): 亮度引导的双边滤波 ——
   值域权重按 U/V 差异计算, 并叠加 Y 的梯度权重,
  色度边缘跟着亮度边缘走, 避免彩色"溢出"到边缘另一侧;
3. 工程上还有 UV 联合滤波(把 U,V 拼成 2D 向量, 用向量中值),
   保证滤波后色度不偏移。

【面试要点】
- 彩噪来源: 小像素高增益 + CCM 对角>1 的放大 + demosaic 插值扩散;
- 为什么不把彩噪提前在 RAW 域全解决: RAW 域对 G 降噪的同时也会把
 亮度噪声压掉, 亮度噪声影响观感更大, 分域处理各取所长;
- UVNR 与 YNR(亮度降噪)的力度配平是画质调优中彩色vs颗粒感的核心
 trade-off, 调参时通常先压彩噪再看亮度噪声容忍度。
"""

import numpy as np
from scipy.ndimage import convolve


def uvnr_bilateral(yuv, ksize=5, sigma_spatial=1.5, sigma_range=12.0,
                   y_edge_weight=0.5):
    """亮度引导的双边 UVNR。

    Parameters
    ----------
    yuv : (H, W, 3), 通道顺序 Y, U, V
    sigma_range : float
        U/V 值域核宽度, 越小保色边越强、降噪越弱。
    y_edge_weight : float
        Y 参与值域权重的比重(0~1), 让色度滤波避开亮度边缘。
    """
    yuv = np.asarray(yuv, np.float32)
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    r = ksize // 2
    h, w = u.shape

    x = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    gs = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_spatial ** 2))

    pad_u = np.pad(u, r, mode="edge")
    pad_v = np.pad(v, r, mode="edge")
    pad_y = np.pad(y, r, mode="edge")

    u_out = np.zeros_like(u)
    v_out = np.zeros_like(v)
    den = np.zeros_like(u)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            nu = pad_u[r + dy: r + dy + h, r + dx: r + dx + w]
            nv = pad_v[r + dy: r + dy + h, r + dx: r + dx + w]
            ny = pad_y[r + dy: r + dy + h, r + dx: r + dx + w]
            du, dv = nu - u, nv - v
            dy_ = ny - y
            wr = np.exp(-(du ** 2 + dv ** 2) / (2 * sigma_range ** 2))
            wy = np.exp(-(dy_ ** 2) / (2 * (sigma_range * 2) ** 2))
            weight = gs[dy + r, dx + r] * wr * (1 + y_edge_weight * (wy - 1))
            u_out += weight * nu
            v_out += weight * nv
            den += weight
    u_out /= np.maximum(den, 1e-8)
    v_out /= np.maximum(den, 1e-8)
    return np.stack([y, u_out, v_out], axis=-1)


def uvnr_median(yuv, ksize=3):
    """简单方案: U/V 分量各做一次中值滤波(向量意义下独立)。"""
    from .nr import median_filter
    yuv = np.asarray(yuv, np.float32)
    yuv[..., 1] = median_filter(yuv[..., 1], ksize)
    yuv[..., 2] = median_filter(yuv[..., 2], ksize)
    return yuv
