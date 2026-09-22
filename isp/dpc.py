# -*- coding: utf-8 -*-
"""
DPC —— 坏点校正 (Defective Pixel Correction)
===============================================

对应视频: 《ISP算法精讲——DPC》

【现象】
画面上出现与周围明显不同的孤立亮点(hot pixel)或暗点(dead pixel)。

【成因 / 分类】
1. 感光元件工艺瑕疵、光线采集缺陷、产品个体差异。
2. hot pixel: 固定输出高值; dead pixel: 固定输出低值;
   noise pixel: 随光照的响应曲线异常(如半坏点)。
3. 静态坏点: 出厂即存在, 位置固定; 动态坏点: 温度漂移、暗电流异常导致,
   随时间/温度出现。

【算法】
1. 静态校正: sensor 厂标定坏点坐标表, ISP 直接查表用邻域同色像素插值修复。
2. 动态校正: 滑窗检测 —— 在 3x3(或 5x5)窗口内取同色通道的邻域中值,
   若 |中心 - 中值| > 阈值, 判定为坏点, 用中值替换; 常迭代 1~2 遍,
   第二遍用较大窗口, 兜底连续/成对坏点。

【工程细节 / 面试要点】
1. 为什么必须用"同色通道"邻域: Bayer 阵列中相邻像素颜色不同, 不同通道
   的信号本来就有差异, 跨色比较会误杀正常像素, 也会把色度信息抹掉。
2. 阈值取舍: 阈值太小会误杀纹理/高频细节, 太大漏检; 好的实现会做
   边缘方向判断(沿梯度方向找同色样本), 或动态阈值。
3. DPC 放在 RAW 域(插值前), 因为修复的是物理坏点, 且此时只有单通道
   信号, 同色判断最可靠; 也可以在 demosaic 后再补一道 YUV 域 DPC 处理
   残留坏点。
4. 误杀带来的"过校正"会让整幅图变糊, 所以 ISP 一般给 DPC 配 strength
   可调, 实拍星光/噪点多的场景要调低。
"""

import numpy as np


def dpc_dynamic(raw, threshold=20.0, bayer_format="RGGB",
                iterations=2, max_val=1023.0, flat_range=None):
    """动态坏点校正: 同色邻域中值检测 + 替换, 可迭代。

    Parameters
    ----------
    raw : (H, W) float array
    threshold : float
        与同色邻域中值的差超过该值即判为坏点。
    iterations : int
        迭代次数, 第 2 遍可修复相邻成对的坏点(第一遍替换后信息更干净)。
    flat_range : float or None
        邻域一致性(平坦度)门限: 同色邻居的 max-min 超过该值时认为是
        边缘而非坏点, 不做校正。None 时取 4*threshold。没有这一层保护,
        强边缘(如黑白棋盘格)处的同色邻居一半黑一半白, 中值落在中间,
        边缘像素会被当坏点"吃掉", 小方块特征会被逐渐腐蚀。

    Notes
    -----
    【关键实现细节 / 易错点】"同色邻域"必须真的同色: 直接在整幅 Bayer 上
    取 3x3 窗口是错的 —— RGGB 中 R 像素的对角邻居是 B、G 像素的对角邻居
    才是同色 G。若把 R/B 配对角、G 配十字, 大片正常像素会被误判成坏点,
    迭代后甚至出现 R/B 值互换的诡异现象。正确做法: 先拆成 R/Gr/Gb/B
    四张半分辨率子图, 每张子图内 3x3 邻域天然同色。
    """
    from .utils import split_bayer, merge_bayer

    if flat_range is None:
        flat_range = 4.0 * threshold

    out = raw.copy()
    for _ in range(iterations):
        chans = split_bayer(out, bayer_format)
        fixed = []
        for ch in chans:
            h2, w2 = ch.shape
            pad = np.pad(ch, 1, mode="edge")
            nb = np.stack([pad[dy:dy + h2, dx:dx + w2]
                           for dy in range(3) for dx in range(3)
                           if not (dy == 1 and dx == 1)], axis=0)
            med = np.median(nb, axis=0)
            spread = nb.max(axis=0) - nb.min(axis=0)
            bad = (np.abs(ch - med) > threshold) & (spread < flat_range)
            fixed.append(np.where(bad, med, ch))
        out = merge_bayer(*fixed, bayer_format=bayer_format)
    return np.clip(out, 0, max_val)


def dpc_static(raw, bad_pixel_list, bayer_format="RGGB", max_val=1023.0):
    """静态坏点校正: 按标定坐标表逐点用同色邻域均值替换。

    bad_pixel_list : list[(row, col)]

    同色邻居: G 像素取 3x3 对角 4 点; R/B 像素同色点在 5x5 窗口的
    偶偏移位置(±2,0)/(0,±2)/(±2,±2)。
    """
    out = raw.copy()
    h, w = raw.shape
    from .utils import _channel_masks
    r_mask, g_mask, b_mask = _channel_masks(out.shape, bayer_format)
    for (r, c) in bad_pixel_list:
        if r <= 1 or r >= h - 2 or c <= 1 or c >= w - 2:
            continue
        if g_mask[r, c]:
            offs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            offs = [(-2, 0), (2, 0), (0, -2), (0, 2),
                    (-2, -2), (-2, 2), (2, -2), (2, 2)]
        nb = [out[r + dy, c + dx] for dy, dx in offs]
        out[r, c] = float(np.mean(nb))
    return np.clip(out, 0, max_val)


def dpc(raw, threshold=20.0, bayer_format="RGGB", iterations=2,
        bad_pixel_list=None, max_val=1023.0):
    """完整 DPC: 静态查表 + 动态检测。"""
    if bad_pixel_list:
        raw = dpc_static(raw, bad_pixel_list, bayer_format, max_val)
    return dpc_dynamic(raw, threshold, bayer_format, iterations, max_val)
