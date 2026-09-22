# -*- coding: utf-8 -*-
"""
BLC —— 黑电平校正 (Black Level Correction)
=============================================

对应视频: 《ISP算法精讲——BLC》

【现象】
拍摄纯黑(盖镜头盖)时, RAW 图的像素值不为 0, 而是整体抬高到一个固定值。

【成因】
1. 暗电流 (dark current): CMOS 光电二极管在无光照时反向截止区仍有微小电流,
   载流子扩散与器件内部缺陷都会造成无照电流, 使 0 照度对应非 0 输出。
2. ADC 偏置: 为了给数据链路预留动态范围、避免负值削波, sensor 通常把
   黑电平抬到一个正值(如 10bit 的 64)再输出。

【算法】
对每个像素减去黑电平(可按 R/Gr/Gb/B 四通道分别减, 因为各通道增益/暗电流
不同), 并截断到 0:

    out = clip(raw - BL_ch, 0, max)

进阶做法是减完后再按 (WL-BL) 归一化到满量程, 充分利用位深。

【工程细节 / 面试要点】
1. 黑电平哪里来: ① OTP/标定值, ② sensor 特殊帧(黑帧)实时统计,
   ③ 实际很多 sensor 会留 dummy rows/ob(光学黑)像素, 用它们实时估计。
2. 为什么按通道减: 四个通道的模拟增益不同, 暗电流折算到数字域的
   offset 也不同, 不分通道减会导致绿平衡(GR/Gb)和色偏问题。
3. BLC 必须放在 pipeline 最前面: 后面所有统计(AWB/AE)、增益计算都假设
   黑色对应 0, 基准不准则全链路误差放大。
4. 减 BL 会让画面整体变暗、损失一点动态范围, 所以有的平台会同时
   做数字增益补偿。
"""

import numpy as np


def estimate_black_level(dark_frame, bayer_format="RGGB", method="median"):
    """从黑帧(盖镜头盖拍)估计黑电平, 分通道返回。

    真实产线两种做法之一(另一见 estimate_black_level_from_ob):
    黑帧统计。取中位数而非均值——坏点/热像素是长尾离群值,
    均值会被拉偏, 中位数稳健。

    Returns
    -------
    dict: {"R": v, "Gr": v, "Gb": v, "B": v} 各通道黑电平
    """
    from .utils import split_bayer
    raw = np.asarray(dark_frame, np.float32)
    R, Gr, Gb, B = split_bayer(raw, bayer_format)
    fn = np.median if method == "median" else np.mean
    return {"R": float(fn(R)), "Gr": float(fn(Gr)),
            "Gb": float(fn(Gb)), "B": float(fn(B))}


def estimate_black_level_from_ob(raw, ob_rows=4, bayer_format="RGGB"):
    """从光学黑区(OB)行估计黑电平。

    多数 sensor 在有效像素区顶部/底部预留被遮蔽的 dummy 行(OB),
    无光照只收暗电流, 是黑电平的实时来源。默认取图像顶部 ob_rows 行,
    按列方向求中位数再对全行求均值, 抗坏点。

    Returns
    -------
    dict: 分通道黑电平
    """
    return estimate_black_level(raw[:ob_rows, :], bayer_format)


def blc(raw, black_level=64.0, max_val=1023.0, bayer_format="RGGB",
        per_channel=None, rescale=False):
    """黑电平校正。

    Parameters
    ----------
    raw : (H, W) float array
        Bayer RAW 图。
    black_level : float
        全局黑电平值。
    per_channel : dict or None
        可选, 形如 {"R": 64, "Gr": 64, "Gb": 64, "B": 64} 的分通道黑电平,
        优先级高于 black_level。
    rescale : bool
        True 时同时做 (WL-BL) 归一化, 输出重新铺满 [0, max_val]。
    """
    if per_channel is None:
        bl_map = np.full_like(raw, float(black_level))
    else:
        from .utils import split_bayer, merge_bayer
        R, Gr, Gb, B = split_bayer(raw, bayer_format)
        R = R - per_channel.get("R", black_level)
        Gr = Gr - per_channel.get("Gr", black_level)
        Gb = Gb - per_channel.get("Gb", black_level)
        B = B - per_channel.get("B", black_level)
        raw = merge_bayer(R, Gr, Gb, B, bayer_format)
        bl_map = 0.0

    out = raw - bl_map
    if rescale:
        wl = float(np.max(raw)) if per_channel is not None else max_val + bl_map.mean()
        out = out / max(wl - float(black_level), 1e-6) * max_val
    return np.clip(out, 0, max_val)
