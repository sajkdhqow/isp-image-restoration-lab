# -*- coding: utf-8 -*-
"""
ISP 算法精讲 —— Python 实现
================================

参考 B 站 UP 主「食鱼者」的《ISP算法精讲》系列视频(共24集)的算法讲解,
将其中提到的所有 ISP 算法用 Python + NumPy 从零实现,并按经典 ISP Pipeline
串联成完整可运行的图像处理链路。

Pipeline 顺序(与视频 1.0《ISPPipeline》中给出的基本流程一致):

    RAW(Bayer) -> DPC -> BLC -> Denoise(BNR) -> LSC -> AWB
    -> Demosaic -> CCM -> Gamma -> CSC(RGB->YUV) -> UVNR -> EE
    -> CSC(YUV->RGB) -> 输出

模块清单(对应视频):
    - mipi_parser   : MIPI 打包 RAW 数据与 RAW 图互转
    - dpc           : 坏点校正 (静态查表 + 动态检测)
    - blc           : 黑电平校正
    - nr            : NR 基础滤波器族(均值/高斯/双边/NLM/中值系列/中值有理)
    - bnr           : RAW 域降噪 (伪四通道 / HVS 加权)
    - lsc           : 镜头阴影校正 (网格增益表 + 多项式拟合)
    - awb           : 自动白平衡 (灰世界/完美反射/动态阈值)
    - demosaic      : 去马赛克 (双线性/Malvar/边缘方向色差法)
    - interpolation : 图像插值 (最近邻/双线性/双三次)
    - ccm           : 颜色校正矩阵 (标定 + 应用)
    - gamma         : 伽马校正 (普通 gamma / sRGB)
    - csc           : 色彩空间转换 (RGB <-> YUV, BT.601/BT.709)
    - uvnr          : YUV 域彩噪去除
    - ee            : 边缘增强 / 锐化
    - pipeline      : 全流程串联

每个模块的 docstring 中都包含: 现象 -> 成因 -> 算法原理 -> 面试要点,
便于求职面试时快速组织语言。
"""

__version__ = "1.0.0"
__author__ = "ISP-Learning"
