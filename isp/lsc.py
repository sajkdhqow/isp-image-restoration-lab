# -*- coding: utf-8 -*-
"""
LSC —— 镜头阴影校正 (Lens Shading Correction)
================================================

对应视频: 《ISP算法精讲——LSC》

【现象】
拍摄纯灰色卡, 图像中心亮、四角暗(luma shading / 暗角);
有的镜头还会四角偏绿或偏品红(color shading), 即亮度与颜色双重不均。

【成因】
1. Luma shading: 镜头光学特性决定 —— 边缘光线斜射造成的光通量下降、
   cos^4 定律(余弦四次方定律)、镜筒遮挡(vignetting 渐晕)。
2. Color shading: 镜头对不同波长光的折射率不同(CRA 与 sensor 微透镜
   不匹配), 不同色光在边缘处的落点偏移不同, 造成边缘偏色。

【算法】
核心思想: 逐像素乘一个位置相关的增益图 Gain(x, y), 中心≈1, 边缘>1。

1. 网格增益表法(工程主流, 本实现):
   - 标定: 拍摄均匀白场(D65 光源 + 积分球/均匀光源), 对 R/Gr/Gb/B 四个
     通道分别在每个网格 bin 内求中值, gain = 中心目标值 / bin 中值;
   - 应用: 对增益网格做双线性插值放大到全分辨率, 逐像素相乘。
   - 存储与带宽友好, 硬件只需一个小 RAM 存网格。
2. 多项式拟合法: gain(r) = 1 + k1*r^2 + k2*r^4 + ...,
   r 为归一化径向距离; 拟合参数少, 但贴合度不如网格法。

【工程细节 / 面试要点】
1. LSC 放在 BLC 之后: 先归零再算增益, 否则黑电平会污染标定。
2. 边缘增益大 -> 边缘噪声同样被放大, SNR 变差; 所以 BNR 降噪强度要
   和 LSC 联动, 或把 LSC 放在降噪后以避免放大噪声(不同平台顺序不同)。
3. 增益要有上限(如 4x), 否则角落噪声爆炸; 实际还能配合 AE 策略。
4. 四通道分开校正, 顺便把 color shading 一起修掉;
   GR/Gb 修不好会带来"绿平衡"问题。
5. 更换镜头模组/贴装偏移都会影响 shading, 产线要做二次标定。
"""

import numpy as np


def _bin_centers(h, w, grid_h, grid_w):
    """网格 bin 的中心坐标(像素坐标)。"""
    ys = (np.arange(grid_h) + 0.5) * (h / grid_h)
    xs = (np.arange(grid_w) + 0.5) * (w / grid_w)
    return ys, xs


def calibrate_lsc_grid(gray_ref, target_value, grid=(33, 33),
                       bayer_format="RGGB", max_gain=8.0):
    """从均匀白场 RAW 标定四通道网格增益表。

    Parameters
    ----------
    gray_ref : (H, W) float array
        均匀光源下拍摄的 RAW(已做 BLC)。
    target_value : float
        期望的中心亮度。
    grid : (grid_h, grid_w)
        增益网格分辨率。

    Returns
    -------
    gains : dict {"R": (gh, gw), "Gr": ..., "Gb": ..., "B": ...}
    """
    from .utils import split_bayer

    h, w = gray_ref.shape
    gh, gw = grid
    ys, xs = _bin_centers(h, w, gh, gw)
    gains = {}
    for name, ch in zip(("R", "Gr", "Gb", "B"),
                        split_bayer(gray_ref, bayer_format)):
        ch_h, ch_w = ch.shape
        # 半分辨率下的 bin 边界
        bin_y = np.linspace(0, ch_h, gh + 1).astype(int)
        bin_x = np.linspace(0, ch_w, gw + 1).astype(int)
        g = np.zeros((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                blk = ch[bin_y[i]:bin_y[i + 1], bin_x[j]:bin_x[j + 1]]
                med = np.median(blk) if blk.size else 0.0
                g[i, j] = np.clip(target_value / max(med, 1.0), 1.0, max_gain)
        gains[name] = g
    return gains


def _interp_grid(gain_grid, shape):
    """把 (gh, gw) 增益网格双线性插值到全分辨率 shape。"""
    gh, gw = gain_grid.shape
    h, w = shape
    ys, xs = _bin_centers(h, w, gh, gw)
    gy = (np.arange(h)[:, None] - 0.0)
    # 采样位置映射到网格坐标系
    fy = np.interp(np.arange(h), ys, np.arange(gh))
    fx = np.interp(np.arange(w), xs, np.arange(gw))
    y0 = np.clip(np.floor(fy).astype(int), 0, gh - 1)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, gw - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    wy = (fy - y0)[:, None]
    wx = (fx - x0)[None, :]
    g = (gain_grid[np.ix_(y0, x0)] * (1 - wy) * (1 - wx)
         + gain_grid[np.ix_(y1, x0)] * wy * (1 - wx)
         + gain_grid[np.ix_(y0, x1)] * (1 - wy) * wx
         + gain_grid[np.ix_(y1, x1)] * wy * wx)
    return g


def lsc_apply(raw, gains, bayer_format="RGGB", max_val=1023.0):
    """按四通道增益表对 RAW 做 LSC。gains 来自 calibrate_lsc_grid。"""
    from .utils import split_bayer, merge_bayer

    R, Gr, Gb, B = split_bayer(raw, bayer_format)
    R *= _interp_grid(gains["R"], R.shape)
    Gr *= _interp_grid(gains["Gr"], Gr.shape)
    Gb *= _interp_grid(gains["Gb"], Gb.shape)
    B *= _interp_grid(gains["B"], B.shape)
    out = merge_bayer(R, Gr, Gb, B, bayer_format)
    return np.clip(out, 0, max_val)


def fit_lsc_polynomial(gray_ref, target_value, order=4, bayer_format="RGGB"):
    """多项式阴影模型: gain(r) = 1 + k1*r^2 + k2*r^4 + ...
    用最小二乘拟合各通道系数, 返回 {channel: coeffs} 与图像尺寸。
    """
    from .utils import split_bayer

    h, w = gray_ref.shape
    yy = (np.arange(h) - (h - 1) / 2) / ((h - 1) / 2)
    xx = (np.arange(w) - (w - 1) / 2) / ((w - 1) / 2)
    r2 = (yy[:, None] ** 2 + xx[None, :] ** 2)
    A = np.stack([(r2 ** k).ravel() for k in range(1, order // 2 + 1)], axis=1)
    coeffs = {}
    for name, ch in zip(("R", "Gr", "Gb", "B"),
                        split_bayer(gray_ref, bayer_format)):
        gain_img = target_value / np.maximum(ch, 1.0)
        coef, *_ = np.linalg.lstsq(A, gain_img.ravel() - 1.0, rcond=None)
        coeffs[name] = np.concatenate([[1.0], coef])
    return coeffs, (h, w)


def lsc(raw, gains=None, target_value=None, grid=(33, 33),
        bayer_format="RGGB", max_val=1023.0):
    """一键 LSC: 未给 gains 时, 用当前图自身当作"白场"做简易标定后应用。

    真实产线流程是拍均匀白场单独标定, 这里为 demo 提供自标定捷径。
    """
    if gains is None:
        if target_value is None:
            target_value = np.percentile(raw, 95)
        gains = calibrate_lsc_grid(raw, target_value, grid, bayer_format)
    return lsc_apply(raw, gains, bayer_format, max_val)
