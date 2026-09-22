# -*- coding: utf-8 -*-
"""
CSC —— 色彩空间转换 (Color Space Conversion / CSM)
====================================================

对应视频: 《ISP算法精讲——CSC》

【为什么转 YUV】
1. 人眼对亮度(Y)敏感、对色度(UV)不敏感:
   - 可以对 UV 降采样(4:2:0/4:2:2)节省带宽;
   - 降噪可以对 UV 下更重的手(UVNR), 锐化只对 Y 做(EE),
     避免锐化放大彩噪 —— 这就是为什么 EE/UVNR 都在 YUV 域;
2. 编码压缩(H.264/HEVC)在 YUV 域进行。

【常用矩阵(全范围 full range)】
BT.601 (SDTV, CVBS 时代, 也是 ISP 默认):
    Y  =  0.299 R + 0.587 G + 0.114 B
    U  =  0.564 (B - Y)      V = 0.713 (R - Y)
BT.709 (HDTV):
    Y  =  0.2126 R + 0.7152 G + 0.0722 B
    U  =  0.539 (B - Y)      V = 0.635 (R - Y)

【工程细节 / 面试要点】
1. full range (0~255) vs limited range (Y:16~235, UV:16~240):
   HDMI/TV 常用 limited, 文件存储(JPEG)多用 full, 搞错会出现
   "灰蒙蒙"或"黑不黑"的画面;
2. 硬件实现: 系数左移 2^10 定点化 + 取整, 加舍入偏移 512;
3. YUV 各分量概念: Y' 是经过 gamma 的亮度(luma), 不是物理亮度
   (luminance), 面试能区分 luma/luminance 加分。
"""

import numpy as np

# 定点化: 系数 << 10 后取整, 硬件行为一致
_BT601 = np.array([
    [0.299, 0.587, 0.114],
    [-0.169, -0.331, 0.5],
    [0.5, -0.419, -0.081],
], dtype=np.float64)

_BT601_INV = np.array([
    [1.0, 0.0, 1.772],
    [1.0, -0.344, -0.714],
    [1.0, 2.017, 0.0],
], dtype=np.float64)

_BT709 = np.array([
    [0.2126, 0.7152, 0.0722],
    [-0.1146, -0.3854, 0.5],
    [0.5, -0.4542, -0.0458],
], dtype=np.float64)


def rgb_to_yuv(rgb, standard="bt601"):
    """RGB(各通道同范围) -> YUV, 返回同尺度 (H,W,3)。"""
    M = _BT601 if standard == "bt601" else _BT709
    return np.einsum("ij,hwj->hwi", M, np.asarray(rgb, np.float64))


def yuv_to_rgb(yuv, standard="bt601"):
    """YUV -> RGB, 与 rgb_to_yuv 互逆(全范围)。"""
    if standard == "bt601":
        M = _BT601_INV
    else:
        # BT.709 逆矩阵
        M = np.linalg.inv(_BT709)
    return np.einsum("ij,hwj->hwi", M, np.asarray(yuv, np.float64))
