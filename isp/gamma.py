# -*- coding: utf-8 -*-
"""
Gamma 与 HVS —— 伽马校正 / 人眼视觉特性
==========================================

对应视频: 《ISP算法精讲——Gamma Correction》《ISP算法精讲——HVS》

【为什么需要 Gamma】
1. 历史原因: CRT 显示器的电子枪输出亮度与输入电压呈幂函数关系
   (L ≈ V^2.2), 为了正确显示, 图像存储前要预补偿 (V = L^(1/2.2)),
   即编码 gamma;
2. 感知原因(更本质): 人眼对亮度的响应是非线性的 —— Weber 定律:
   刚可察觉差 JND 与背景亮度成正比, 暗区需要更大的绝对变化才能
   察觉, 中高亮区很小的变化就能察觉。因此把编码带宽更多地分配给
   暗部(每级码值代表的物理亮度在暗区更细), 与 HVS 特性匹配,
   8bit 量化噪声才不可见。

【结论要分清(面试常挖坑)】
- sensor 输出线性光 -> ISP 中做 gamma 编码(I = L^(1/2.2), 曲线上凸,
  提亮暗部) -> 显示端天然有解码特性(V^2.2);
- 视频里那张"对比度变化"的图是反 gamma 的直观感受, 别混淆;
- sRGB 的编码是分段函数(线性段 + 幂函数段), 线性段避免 0 点斜率
  无穷大导致暗部量化炸裂。

【工程细节】
- ISP 中用 LUT(查找表)实现, 硬件是 RAM; 常把 gamma 与对比度/色调
  曲线合成一张 LUT;
- gamma 之前是线性域: CCM、白平衡都在线性域; gamma 之后是感知域:
  锐化、直方图统计、AE 权重都在感知域更符合人眼评价。
"""

import numpy as np


def gamma_lut(gamma=2.2, bits=8):
    """生成 gamma 编码查找表: out = (in/max)^(1/gamma) * max。"""
    max_val = (1 << bits) - 1
    x = np.arange(max_val + 1, dtype=np.float64) / max_val
    lut = np.power(x, 1.0 / gamma)
    return (lut * max_val + 0.5).astype(np.uint8)


def gamma_correct(img, gamma=2.2, max_val=1023.0):
    """对归一化图像做 gamma 编码(提亮暗部)。"""
    x = np.clip(np.asarray(img, np.float64) / max_val, 0, 1)
    return np.power(x, 1.0 / gamma)  # 输出 [0,1]


def srgb_encode(img_linear, max_val=1023.0):
    """sRGB 编码(分段): 感知均匀性优于纯 gamma 2.2。"""
    x = np.clip(np.asarray(img_linear, np.float64) / max_val, 0, 1)
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(img_srgb):
    """sRGB 解码回线性光域。"""
    x = np.clip(np.asarray(img_srgb, np.float64), 0, 1)
    return np.where(x <= 0.04045, x / 12.92,
                    np.power((x + 0.055) / 1.055, 2.4))


def weber_jnd(background_lum):
    """Weber 定律演示: JND ≈ k * I (k≈0.02)。
    说明暗区人眼需要更大绝对变化才可察觉, 支持 gamma 编码的思想。
    """
    return 0.02 * np.asarray(background_lum)
