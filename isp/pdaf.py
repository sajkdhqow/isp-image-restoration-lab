# -*- coding: utf-8 -*-
"""
PDAF —— 相位对焦与混合 AF
===========================

【背景】对比度 AF 要来回移镜搜索(几十次采图), 且重复纹理会假峰。
PDAF(Phase Detection AF)在 sensor 上做了一半像素的遮挡(microlens
半掩蔽), 左右两组"半像素"各看到镜头出瞳的一半, 相当于两只眼睛:
离焦时两幅半图有反向视差, 由视差可直接解出离焦方向与量级,
一次跳到对焦点附近, 再用对比度 AF 精修 —— 混合 AF, 手机主流方案。

【视差 -> 离焦的几何】
离焦量为 e 时, 左右半图的像水平错开 2*e*k(k 与光圈/像素尺寸相关),
方向指示"太近"还是"太远"。demo 用水平平移 + 离焦模糊合成 L/R:

    L = shift(scene, -d) 再模糊,  R = shift(scene, +d) 再模糊
    d = parallax_gain * (lens_pos - best_pos) / pos_range  (像素)
    视差(parallax) = 2d, 由相位相关(phase correlation)测出。

真实 sensor 的 PDAF 像素是稀疏排布的(如每 4 行取 2), 半图要
demosaic/插值才能做相关; 高端平台用专用硬件块做相关计算。

【工程要点 / 面试点】
1. 方向标定: 视差符号与"该往哪边推镜头"的映射依赖模组装配方向,
   产线要做 PDAF 方向/增益标定 —— 本实现内置自动标定(试探一步,
   看视差是否减小), 这也是真实产线流程的简化版;
2. 弱光退化: 遮挡损失一半感光面积 + 半图噪声大, 暗光下 PDAF
   置信度低, 需回退对比度 AF —— 混合策略按置信度加权;
3. 重复纹理: PDAF 直接测几何视差, 不受重复纹理假峰影响(对比度
   AF 的痛点), 这是混合方案的核心价值;
4. 视差只有水平一维信息(线型 PDAF), 斜向纹理的相关峰会漂移,
   全像素/四合一 PDAF(如 Quad Bayer)可改善。
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from .af import af_search, defocus_sigma


def pdaf_lr_views(scene_lin, lens_pos, best_pos, pos_range=1023.0,
                  parallax_gain=6.0, k=4.0):
    """虚拟 PDAF 相机: 生成左/右半像素图(视差 + 离焦模糊)。

    d 与 (lens_pos - best_pos) 同号: 镜头在近焦点侧(<)时 d<0,
    上下文中只要求 L/R 视差符号与离焦方向单调对应即可, 算法侧
    通过自动标定确定符号约定。k 为离焦模糊增益(与 af_search 用
    同一物理模型)。
    """
    d = parallax_gain * (float(lens_pos) - float(best_pos)) / pos_range
    sigma = defocus_sigma(lens_pos, best_pos, k=k)

    def view(s):
        out = np.empty_like(scene_lin)
        for i in range(scene_lin.shape[-1]):
            out[..., i] = ndi_shift(scene_lin[..., i], (0, s),
                                    order=1, mode="nearest")
        if sigma > 1e-3:
            out = np.stack([gaussian_filter(out[..., i], sigma)
                            for i in range(out.shape[-1])], axis=-1)
        return out

    return view(-d), view(+d), d


def pdaf_parallax(L, R):
    """相位相关测 L/R 视差。返回 (dx, response)。

    response 是相位相关的峰值置信度(0~1), 真实 PDAF 用它决定
    "信 PDAF 还是回退对比度 AF"。
    """
    w = np.array([0.299, 0.587, 0.114], np.float32)
    luma = lambda x: (x @ w).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(luma(L), luma(R))
    return float(dx), float(resp)


def af_hybrid_search(pdaf_fn, contrast_fn, pos0=64, pos_range=1023.0,
                     parallax_gain=6.0, refine_win=24, refine_step=8,
                     n_frames=1):
    """混合 AF: PDAF 粗定位(2 次采样) + 对比度小邻域精修。

    Parameters
    ----------
    pdaf_fn : callable(lens_pos) -> (dx, response)
        在指定位置采 L/R 像素对并返回视差与置信度。
    contrast_fn : callable(lens_pos, frame_idx) -> metric
        对比度对焦指标(供精修)。
    pos0 : int
        搜索起点(真实相机为上一帧对焦位置/hyperfocal)。
    parallax_gain : float
        视差->位置的换算增益, 需与虚拟相机一致(真实模组由标定给出)。

    Returns
    -------
    best_pos : int
    records : list[(pos, metric)] 精修阶段的对比度搜索记录
    info : dict: pdaf_est(PDAF 估计位置), n_pdaf, n_contrast,
           response(最后一步 PDAF 置信度)
    """
    n_pdaf = 0
    n_contrast = 0

    dx, resp = pdaf_fn(pos0)
    n_pdaf += 1
    k = 2.0 * parallax_gain / pos_range   # 视差 px / 位置单位
    if abs(dx) < 0.02:
        est = int(pos0)                   # 视差≈0, 已在对焦点
    else:
        # 方向标定: 常规约定视差 dx 与 (pos - best) 同号, 按 -dx/k 推
        # 一步后视差应变小; 变大说明模组装配映射反了, 翻转符号 s ——
        # 简化版产线 PDAF 方向标定
        step = -dx / k
        cand = int(np.clip(round(pos0 + step), 0, pos_range))
        dx2, _ = pdaf_fn(cand)
        n_pdaf += 1
        s = -1.0 if abs(dx2) > abs(dx) * 0.9 else 1.0
        est = int(np.clip(round(pos0 + s * step), 0, pos_range))

        # 迭代精化: 远处离焦模糊重、视差测量偏差大, 跳到近处后重测
        # 1~2 次(轻模糊下相位相关更准)。修正量始终朝视差减小方向:
        # est -= s*dx/k, 符号由标定统一给出
        for _ in range(2):
            if not (0 < est < pos_range):
                break
            dx, resp = pdaf_fn(est)
            n_pdaf += 1
            if abs(dx) < 0.02:
                break
            est = int(np.clip(round(est - s * dx / k), 0, pos_range))

    # 对比度精修: est ±refine_win 小窗口, 复用三段式 af_search。
    # 位置缓存: af_search 的粗/精/二次精窗口有重叠, 同一位置会被
    # 重复探测; 真实系统每次探测 = 移镜+采图(几十 ms), 缓存省掉
    # 重复行程, 采图帧数进一步下降
    cache = {}

    def wrapped(pos, frame_idx):
        nonlocal n_contrast
        key = (int(pos), int(frame_idx))
        if key not in cache:
            n_contrast += 1
            cache[key] = contrast_fn(pos, frame_idx)
        return cache[key]

    best, rec = af_search(wrapped,
                          pos_min=max(0, est - refine_win),
                          pos_max=min(int(pos_range), est + refine_win),
                          coarse_step=refine_win, fine_step=refine_step,
                          fine2_step=max(2, refine_step // 2),
                          n_frames=n_frames)
    info = {"pdaf_est": est, "n_pdaf": n_pdaf, "n_contrast": n_contrast,
            "response": float(resp)}
    return best, rec, info
