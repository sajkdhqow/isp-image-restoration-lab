# -*- coding: utf-8 -*-
"""
ISP Pipeline —— 全流程串联
============================

对应视频: 《1.0 ISP算法精讲——ISPPipeline》

流程(与视频基本 pipeline 一致, 括号内为常见平台差异):

    RAW(Bayer)
      -> DPC        坏点校正(静态+动态)
      -> BLC        黑电平校正
      -> BNR        RAW 域降噪(可选)
      -> LSC        镜头阴影校正
      -> AWB        自动白平衡增益(作用在 RAW 四通道)
      -> Demosaic   去马赛克, 得到 RGB
      -> CCM        颜色校正(sensor RGB -> sRGB)
      -> Gamma      伽马编码(线性 -> 感知域)
      -> CSC        RGB -> YUV
      -> UVNR       色度降噪
      -> EE         亮度锐化
      -> CSC        YUV -> RGB
      -> 输出 sRGB 图像

注意: 不同平台的模块顺序会调整, 面试要能说清"为什么":
- DPC 在 BLC 前: 黑电平不影响坏点判断(阈值同向平移), 且早修坏点
  避免其污染后面的统计;
- 也有平台把 LSC 放降噪前(先放大再一起压噪), 或 AWB 塞进 LSC 增益表;
- CCM 必须在 gamma 前的线性域; 锐化/直方图在 gamma 后的感知域。
"""

import numpy as np

from .blc import blc
from .dpc import dpc
from .lsc import lsc
from .bnr import bnr_hvs
from .awb import awb
from .utils import apply_channel_gains
from .demosaic import demosaic
from .ccm import apply_ccm
from .gamma import gamma_correct
from .csc import rgb_to_yuv, yuv_to_rgb
from .uvnr import uvnr_bilateral
from .ee import ee_sharpen


# 模拟"sensor 光谱响应偏差"用的矩阵与它的逆(标定得到的 CCM)
SENSOR_COLOR_DISTORTION = np.array([
    [1.45, -0.25, -0.20],
    [-0.15, 1.35, -0.20],
    [-0.10, -0.30, 1.40],
])
DEFAULT_CCM = np.linalg.inv(SENSOR_COLOR_DISTORTION)
# 按行归一化(白点保持), 与真实标定流程一致
DEFAULT_CCM = DEFAULT_CCM / DEFAULT_CCM.sum(axis=1, keepdims=True)


def run_pipeline(raw, bayer_format="RGGB", max_val=1023.0,
                 black_level=64.0,
                 enable_dpc=True, dpc_threshold=20.0,
                 enable_bnr=True,
                 enable_lsc=True, lsc_target=None, lsc_gains=None,
                 awb_method="dynamic_threshold",
                 ccm=None,
                 gamma=2.2,
                 enable_uvnr=True,
                 enable_ee=True, ee_lam=0.6,
                 capture_stages=False):
    """执行完整 ISP 流水线。

    Parameters
    ----------
    raw : (H, W) float array, Bayer RAW(未做任何校正)
    max_val : float
        位深满量程(10bit -> 1023)。
    lsc_gains : dict or None
        白场标定得到的四通道增益表(calibrate_lsc_grid 的输出)。
        产线标准流程; None 时退化为用当前图自标定(demo 捷径, 彩色
        物体会被错误拉灰, 仅作功能演示)。
    ccm : 3x3 or None
        None 时使用默认标定矩阵(与 tools.simulate_raw 的色偏配对)。

    Returns
    -------
    img : (H, W, 3) float [0, 1], 最终 sRGB 图像
    stages : dict (capture_stages=True 时), 各中间结果便于可视化
    """
    stages = {}

    # 1. DPC 坏点校正
    if enable_dpc:
        raw = dpc(raw, threshold=dpc_threshold, bayer_format=bayer_format,
                  iterations=2, max_val=max_val)
    if capture_stages:
        stages["01_dpc"] = raw.copy()

    # 2. BLC 黑电平校正
    raw = blc(raw, black_level=black_level, max_val=max_val,
              bayer_format=bayer_format)
    if capture_stages:
        stages["02_blc"] = raw.copy()

    # 3. BNR RAW 域降噪
    if enable_bnr:
        raw = bnr_hvs(raw, strength=1.0, bayer_format=bayer_format,
                      max_val=max_val)
    if capture_stages:
        stages["03_bnr"] = raw.copy()

    # 4. LSC 镜头阴影校正
    if enable_lsc:
        raw = lsc(raw, gains=lsc_gains, target_value=lsc_target,
                  bayer_format=bayer_format, max_val=max_val)
    if capture_stages:
        stages["04_lsc"] = raw.copy()

    # 5. AWB 自动白平衡(增益作用于 RAW 通道)
    rgain, bgain, grgain, gbgain = awb(raw, method=awb_method,
                                       bayer_format=bayer_format,
                                       max_val=max_val)
    raw = apply_channel_gains(raw, rgain, bgain, grgain, gbgain,
                              bayer_format=bayer_format, max_val=max_val)
    if capture_stages:
        stages["05_awb"] = raw.copy()
        stages["05_awb_gains"] = (rgain, bgain, grgain, gbgain)

    # 6. Demosaic 去马赛克 -> RGB (线性域)
    rgb = demosaic(raw, method="malvar", bayer_format=bayer_format,
                   max_val=max_val) / max_val
    if capture_stages:
        stages["06_demosaic"] = rgb.copy()

    # 7. CCM 颜色校正(线性域)
    # 【关键工程点】CCM 必须与 AWB 联动: AWB 增益 D 作用在 sensor 域之后,
    # 色彩响应变成 D@M, CCM 需按增益后的响应标定, 即 inv(D@M)
    # (真实平台按色温在多组标定 CCM 之间插值, 本质相同)。
    # 若直接用 inv(M), AWB 的对角增益会残留在色彩里导致整体色相偏移。
    if ccm is None:
        D = np.diag([rgain, (grgain + gbgain) / 2, bgain])
        ccm = np.linalg.inv(D @ SENSOR_COLOR_DISTORTION)
        ccm = ccm / ccm.sum(axis=1, keepdims=True)
    rgb = apply_ccm(rgb, ccm)
    rgb = np.clip(rgb, 0, 1)
    if capture_stages:
        stages["07_ccm"] = rgb.copy()

    # 8. Gamma 伽马编码(线性 -> 感知域)
    rgb = gamma_correct(rgb, gamma=gamma, max_val=1.0)
    if capture_stages:
        stages["08_gamma"] = rgb.copy()

    # 9. CSC -> YUV
    yuv = rgb_to_yuv(rgb, standard="bt601") * 255.0  # 缩放到 0~255 方便调参

    # 10. UVNR 色度降噪
    if enable_uvnr:
        yuv = uvnr_bilateral(yuv, ksize=5, sigma_spatial=1.5,
                             sigma_range=14.0)
    if capture_stages:
        # YUV 数据直接存图会被当 RGB 显示成怪色, 转回 RGB 便于可视化
        stages["10_uvnr"] = yuv_to_rgb(yuv / 255.0, standard="bt601")

    # 11. EE 锐化(只作用 Y 通道)
    if enable_ee:
        yuv = ee_sharpen(yuv, lam=ee_lam, sigma=1.0, edge_thr=10.0)
    if capture_stages:
        stages["11_ee"] = yuv_to_rgb(yuv / 255.0, standard="bt601")

    # 12. CSC -> RGB
    rgb = np.clip(yuv_to_rgb(yuv / 255.0, standard="bt601"), 0, 1)

    if capture_stages:
        stages["12_final"] = rgb.copy()
        return rgb, stages
    return rgb
