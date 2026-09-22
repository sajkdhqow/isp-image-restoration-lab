# -*- coding: utf-8 -*-
"""
CCM —— 颜色校正矩阵 (Color Correction Matrix)
================================================

对应视频: 《ISP算法精讲——CCM》

【为什么需要】
sensor 的光谱响应(CMF)与人眼不同, 还混有 CFA 串扰, 导致拍出的颜色
"饱和度低、偏色"。CCM 把 sensor RGB 线性映射到标准色彩空间(sRGB):

    [R']     [c00 c01 c02]   [R]
    [G']  =  [c10 c11 c12] @ [G]
    [B']     [c20 c21 c22]   [B]

【标定】
1. 拍摄标准 24 色卡(ColorChecker), 每块取中心区域均值,
   得到 sensor RGB 响应 S (N x 3);
2. 已知色卡的标准 sRGB 参考值 T (N x 3);
3. 最小二乘求解 M = (S^T S)^-1 S^T T;
4. 约束处理: 常把每行归一化到和为 1(白点保持, 即白色经 CCM 后仍白),
   并可对亮度列做能量保持修正。
5. 实际标定在 AWB 之后、线性光域(demosaic 后, gamma 前)进行。

【应用时注意】
- CCM 会放大噪声: 对角系数通常 >1(如 1.7), 副系数为负,
  相当于通道间做差分, 噪声方差被放大 —— 所以 CCM 之后 gamma 之前
  常再做一次降噪或用 3D LUT 代替;
- 饱和度补偿: 纯 CCM 后画面常偏灰,有的方案再乘一个饱和度因子
  blend: rgb' = gray + sat * (rgb - gray);
- 色温相关: 不同光源下 sensor 响应不同, 严格方案要按色温查多组
  CCM 做插值(multi-CCM), 与 AWB 模块联动。

【面试要点】
- 为什么 CCM 在 gamma 前: gamma 是非线性操作, 线性色彩变换必须
  在线性域做, 顺序反了颜色必错;
- 为什么不用 3x3 而用 4x3 或加常数列: 处理不完全依赖原色的偏移项;
- CCM 与 3D LUT 的取舍: 3x3 只能矫正线性色偏, 3D LUT 可以矫正
  非线性/局部色偏, 但存储与标定成本高。
"""

import numpy as np


def calibrate_ccm(sensor_rgb, target_rgb, row_normalize=True):
    """最小二乘标定 3x3 CCM。

    Parameters
    ----------
    sensor_rgb : (N, 3) 色卡在 sensor 线性 RGB 空间的响应(白平衡后)
    target_rgb : (N, 3) 色卡的标准 sRGB 值(线性光域)
    row_normalize : bool
        每行归一化到和为 1, 保证白点不偏移。
    """
    sensor_rgb = np.asarray(sensor_rgb, np.float64)
    target_rgb = np.asarray(target_rgb, np.float64)
    # T = S @ M^T  ->  M = lstsq(S, T)
    M, *_ = np.linalg.lstsq(sensor_rgb, target_rgb, rcond=None)  # (3,3)
    M = M.T
    if row_normalize:
        M = M / M.sum(axis=1, keepdims=True)
    return M


def apply_ccm(rgb, ccm):
    """对 (H, W, 3) RGB 图应用 3x3 CCM。"""
    rgb = np.asarray(rgb, np.float64)
    return np.einsum("ij,hwj->hwi", np.asarray(ccm, np.float64), rgb)


def apply_ccm_saturation(rgb, ccm, saturation=1.0):
    """CCM + 饱和度补偿: 在每像素上做
        gray = mean(rgb'), rgb'' = gray + sat*(rgb' - gray)
    """
    out = apply_ccm(rgb, ccm)
    if saturation != 1.0:
        gray = out.mean(axis=-1, keepdims=True)
        out = gray + saturation * (out - gray)
    return out


# ---------------------------------------------------------------------------
# demo 用: 内置 ColorChecker 24 色的 sRGB 参考值(线性光域)
# ---------------------------------------------------------------------------
# 常用的 ColorChecker 标准 D50 线性 sRGB 均值(与 BabelColor 平均数据一致,
# 工程上建议以自己的测量值/官方数据表为准)
COLORCHECKER_SRGB_LINEAR = np.array([
    [0.1182, 0.0732, 0.0494],  # dark skin
    [0.2290, 0.1276, 0.0916],  # light skin
    [0.1136, 0.1490, 0.2234],  # blue sky
    [0.1008, 0.1161, 0.0440],  # foliage
    [0.1863, 0.1690, 0.3197],  # blue flower
    [0.1450, 0.2577, 0.3133],  # bluish green
    [0.2317, 0.1629, 0.0422],  # orange
    [0.1021, 0.0979, 0.3215],  # purplish blue
    [0.2826, 0.1060, 0.1234],  # moderate red
    [0.1374, 0.2385, 0.0745],  # purple
    [0.3460, 0.3863, 0.0479],  # yellow green
    [0.3744, 0.2312, 0.0307],  # orange yellow
    [0.0734, 0.0445, 0.1861],  # blue
    [0.0646, 0.2309, 0.1116],  # green
    [0.3499, 0.0605, 0.0765],  # red
    [0.3300, 0.4753, 0.5043],  # yellow
    [0.4153, 0.3004, 0.0272],  # magenta
    [0.0769, 0.1425, 0.3348],  # cyan
    [0.8157, 0.8135, 0.8065],  # white 9.5
    [0.5772, 0.5772, 0.5772],  # neutral 8
    [0.3548, 0.3548, 0.3548],  # neutral 6.5
    [0.1945, 0.1945, 0.1945],  # neutral 5
    [0.0908, 0.0908, 0.0908],  # neutral 3.5
    [0.0311, 0.0311, 0.0311],  # black 2
], dtype=np.float64)
