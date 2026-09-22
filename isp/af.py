# -*- coding: utf-8 -*-
"""
AF —— 自动对焦 (Auto Focus)
=============================

3A 之三。手机模组用 VCM(音圈马达)推动镜头, 镜头位置由 DAC 值控制
(常见 10bit, 0~1023)。对比度 AF 的思路: 焦平面处图像高频能量最大,
沿镜头位置搜索对焦指标的峰值:

    镜头位置 -> 采图 -> 对焦指标(Tenengrad) -> 搜索峰值 -> 对焦完成

【指标】
Tenengrad: Sobel 梯度平方和的均值。对焦清晰处边缘最锐利, 指标最大。
也可用 Laplacian 能量 / 归一化方差。噪声会抬高所有位置的指标基线
(把"地板"抬起来但峰还在), 所以 AF 要在 AE 收敛后做。

【搜索策略】
两段式: 粗扫(大步长, 定位峰区) + 精搜(峰值邻域小步长爬山)。
全范围细扫太耗时(每次移镜+采图几十 ms, 用户体验差), 粗扫 + 精搜
能把搜索次数从 ~64 次压到 ~25 次。

【面试要点】
1. 为什么 AF 在 AE 之后: 对焦指标是梯度能量, 噪声淹没梯度时峰被
   埋掉; 欠曝下搜索出来的"峰值"是噪声起伏, 对焦必错。
2. 爬山法的局部极值: 重复纹理(栅栏/百叶窗)会出现多个假峰, 工程上
   用 PDAF(相位对焦)做粗定位 + 对比度 AF 精修(混合 AF); PDAF 靠
   遮挡像素对的视差直接算出离焦方向和量级, 一次到位。
3. 峰值附近指标近似抛物线, 可用三点抛物线拟合做亚步长插值, 把
   对焦精度做到比步长更细。
4. 搜索过程中画面虚焦, 显示端要冻结预览或直接回跳峰值位置,
   否则用户看到画面"呼吸"。
5. 对焦窗口一般取画面中心或用户点按的区域(ROI), 不用全图 ——
   主体在画面中央, 背景的梯度会干扰峰值判断。
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def defocus_sigma(lens_pos, best_pos, k=4.0, pos_range=1023.0):
    """演示用散焦模型: 高斯模糊 sigma 与 |位置-对焦点| 成正比。

    真实光学是离焦 PSF(弥散圆/艾里斑), demo 用高斯近似 —— 只影响
    演示的真实感, 不影响 AF 搜索逻辑本身。
    """
    return max(1e-3, k * abs(float(lens_pos) - best_pos) / pos_range)


def blur_linear_scene(scene_lin, sigma):
    """线性域高斯模糊(模拟离焦), 返回模糊后的线性场景。

    离焦发生在光学系统, 应作用在线性光域; 在 gamma 域模糊会产生
    轻微的暗边偏差, demo 数据上看不出差别, 但习惯要保持。
    """
    if sigma <= 1e-3:
        return scene_lin
    return np.stack([gaussian_filter(scene_lin[..., i], sigma)
                     for i in range(3)], axis=-1)


def focus_metric_tenengrad(luma, max_val=1023.0):
    """Tenengrad 对焦指标: Sobel 梯度能量均值(越大越清晰)。"""
    img = np.asarray(luma, np.float32) / max_val
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return float(np.mean(gx * gx + gy * gy))


def af_search(eval_fn, pos_min=0, pos_max=1023, coarse_step=128,
              fine_step=16, fine2_step=4, n_frames=1):
    """三段式对比度 AF: 粗扫 -> 精搜 -> 二次精搜 + 抛物线插值。

    Parameters
    ----------
    eval_fn : callable(lens_pos, frame_idx) -> metric
        移动镜头到指定位置, 采图并返回对焦指标。frame_idx 供多帧
        平均使用(n_frames>1 时调用方内部用不同 seed 采多帧,
        平均后指标曲线更平滑, 单帧噪声导致的定位抖动被压掉)。
    fine2_step : int
        二次精搜步长(在 ±fine_step 窗口内), 0 表示跳过。
    n_frames : int
        每个位置的采图帧数, 指标取多帧平均。

    Returns
    -------
    best_pos : int 对焦位置
    records : list[(pos, metric)] 搜索记录(每位置一条, 已含多帧平均)

    Notes
    -----
    【精度来源】最终误差由三层贡献: 精搜步长(量化误差) + 抛物线
    插值残差 + 单帧指标噪声。三层分别用 fine2_step(步长 16->4)、
    抛物线顶点、n_frames 平均逐一压掉。搜索位置数从两段式的
    ~34 增加到 ~43, 但换亚步长精度是划算的(位置探测是主要成本,
    多帧只在关键窗口用)。
    """
    records = []

    def probe(pos):
        m = float(np.mean([eval_fn(pos, i) for i in range(n_frames)]))
        records.append((int(pos), m))
        return m

    # 1) 粗扫: 大步长全范围, 定位峰区
    coarse_pos = list(range(pos_min, pos_max + 1, coarse_step))
    metrics = [probe(p) for p in coarse_pos]
    c_best = coarse_pos[int(np.argmax(metrics))]

    # 2) 精搜: 峰两侧各扩一个粗步长, 小步长爬山
    lo = max(pos_min, c_best - coarse_step)
    hi = min(pos_max, c_best + coarse_step)
    fine_pos = list(range(lo, hi + 1, fine_step))
    f_metrics = [probe(p) for p in fine_pos]
    f_best = fine_pos[int(np.argmax(f_metrics))]

    # 3) 二次精搜: 更小步长, 在 ±fine_step 窗口内收窄
    best = f_best
    if fine2_step and fine2_step < fine_step:
        lo2 = max(pos_min, f_best - fine_step)
        hi2 = min(pos_max, f_best + fine_step)
        fb_pos = list(range(lo2, hi2 + 1, fine2_step))
        fb_metrics = [probe(p) for p in fb_pos]
        i_best = int(np.argmax(fb_metrics))
        best = fb_pos[i_best]

        # 4) 抛物线亚步长插值: 峰值附近指标近似抛物线, 用峰点与两侧
        #    邻点拟合顶点, 精度做到比步长更细(面试加分点)。
        #    顶点在中间(denom<0)才有效; 噪声或平台形曲线时跳过。
        if 0 < i_best < len(fb_pos) - 1:
            m1, m2, m3 = fb_metrics[i_best - 1:i_best + 2]
            denom = m1 + m3 - 2 * m2
            if denom < 0:
                delta = fine2_step * (m1 - m3) / (2 * denom)
                best = int(round(best + float(np.clip(delta, -fine2_step,
                                                      fine2_step))))

    probe(best)
    return best, records
