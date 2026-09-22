# -*- coding: utf-8 -*-
"""
AWB —— 自动白平衡 (Auto White Balance)
========================================

对应视频: 《ISP算法精讲——AWB》

【现象与概念】
人眼有"颜色恒常性"——光照变化下我们对白纸的知觉趋于稳定;
sensor 没有, 拍出来的白色会随光源色温偏黄/偏蓝。
色温: 黑体辐射的颜色随温度变化, 低温偏暖(黄红), 高温偏冷(蓝),
单位 K(开尔文)。白平衡的目标: 让画面中的"白色"恢复为白色。

【算法】
AWB 统计 -> 计算通道增益 -> 在 RAW 域(R/Gr/Gb/B)乘增益:

1. 灰世界假设 (Gray World):
   假设场景平均反射率是灰色, 即 R/G/B 均值应相等:
       Rgain = G_mean / R_mean, Bgain = G_mean / B_mean
   简单快速, 但画面被大面积单色覆盖(草地/大海)时彻底失效。

2. 完美反射假设 (Perfect Reflector / White Patch):
   假设画面中最亮的点就是白色。取高亮度分位数的像素做统计。
   高光过曝或无白色物体时失效。

3. 动态阈值法 (经典工程实现):
   a. RGB -> YUV, 先按亮度直方图掐头去尾(排除过曝/欠曝);
   b. 在 U-V 平面上, 用均值±k·σ 的动态阈值筛选"接近白色"的像素;
   c. 用筛选出的白区像素计算 R/B 增益, 迭代更新。
   鲁棒性最好, 是多数 ISP firmware 的基础框架。

【工程细节 / 面试要点】
1. 增益作用位置: 在 RAW 域 demosaic 之前逐通道乘增益, 硬件只需 2 个
   乘法器; 有些平台也把 AWB 增益塞进 LSC 增益表一起乘, 省一遍访存。
2. 增益限制: Rgain/Bgain 要夹在 [1/K, K](如 4)内, 防止统计错误把
   画面拉爆; G 通道增益通常归一为 1。
3. 特殊场景: 夜景路灯(钠灯偏橙)会把 AWB 拉向蓝色, 需要场景检测/
   色温曲线约束; 这是面试最爱问的"AWB 什么时候翻车"。
4. 评价 AWB 的客观指标: 白点误差(ΔUV), 常用色卡 24 色块统计。
"""

import numpy as np


def awb_gray_world(raw, bayer_format="RGGB", max_val=1023.0,
                   gain_clip=(0.25, 4.0)):
    """灰世界 AWB, 返回 (rgain, bgain, grgain, gbgain)。"""
    from .utils import split_bayer

    R, Gr, Gb, B = split_bayer(raw, bayer_format)
    r_mean, b_mean = R.mean(), B.mean()
    g_mean = (Gr.mean() + Gb.mean()) / 2
    rgain = np.clip(g_mean / max(r_mean, 1e-6), *gain_clip)
    bgain = np.clip(g_mean / max(b_mean, 1e-6), *gain_clip)
    return float(rgain), float(bgain), 1.0, 1.0


def awb_perfect_reflector(raw, bayer_format="RGGB", max_val=1023.0,
                          percentile=99.0, tol=0.4, gain_clip=(0.25, 4.0)):
    """完美反射 AWB: 取高亮分位像素, 并要求三通道都足够亮(避免彩色高光)。"""
    from .utils import split_bayer

    R, Gr, Gb, B = split_bayer(raw, bayer_format)
    thr = np.percentile(np.dstack([R, Gr, Gb, B]).ravel(), percentile)
    mR, mG, mB = R[R >= thr * tol].mean() if np.any(R >= thr * tol) else R.mean(), \
        Gr[Gr >= thr * tol].mean() if np.any(Gr >= thr * tol) else Gr.mean(), \
        B[B >= thr * tol].mean() if np.any(B >= thr * tol) else B.mean()
    rgain = np.clip(mG / max(mR, 1e-6), *gain_clip)
    bgain = np.clip(mG / max(mB, 1e-6), *gain_clip)
    return float(rgain), float(bgain), 1.0, 1.0


def _rgb_to_yuv(rgb):
    """BT.601 YUV (full range), rgb: (..., 3) in [0, 1]。"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.564 * (b - y)
    v = 0.713 * (r - y)
    return y, u, v


def awb_dynamic_threshold(raw, bayer_format="RGGB", max_val=1023.0,
                          iterations=3, k=1.5, gain_clip=(0.25, 4.0)):
    """动态阈值 AWB(经典 YUV 白区筛选 + 迭代)。

    步骤:
    1) 依当前增益做简易 demosaic(demo 用双线性);
    2) 转 YUV, 掐掉亮度前后 5% 的极端像素;
    3) 白区条件: |U|<k*σU 且 |V|<k*σV, 每轮迭代收紧一次;
    4) 用白区像素的通道均值算增益。
    """
    from .utils import split_bayer, merge_bayer, apply_channel_gains
    from .demosaic import demosaic_bilinear

    rgain = bgain = 1.0
    for _ in range(iterations):
        cur = apply_channel_gains(raw, rgain, bgain,
                                  bayer_format=bayer_format,
                                  max_val=max_val)
        rgb = demosaic_bilinear(cur, bayer_format) / max_val
        y, u, v = _rgb_to_yuv(rgb)
        lum_lo, lum_hi = np.percentile(y, 5), np.percentile(y, 95)
        base = (y > lum_lo) & (y < lum_hi)
        if base.sum() < 100:
            break
        mu_u, sd_u = u[base].mean(), u[base].std()
        mu_v, sd_v = v[base].mean(), v[base].std()
        white = base & (np.abs(u - mu_u) < k * sd_u) \
                     & (np.abs(v - mu_v) < k * sd_v)
        if white.sum() < 100:
            break
        # 白区像素在半分辨率四通道上的均值
        R, Gr, Gb, B = split_bayer(cur, bayer_format)
        # 白区掩码取上半分辨率对应位置(直接取同位置 2x2 中至少一个白)
        w_half = (white[0::2, 0::2] | white[0::2, 1::2] |
                  white[1::2, 0::2] | white[1::2, 1::2])
        if w_half.sum() < 50:
            break
        r_mean, b_mean = R[w_half].mean(), B[w_half].mean()
        g_mean = (Gr[w_half].mean() + Gb[w_half].mean()) / 2
        rgain = np.clip(g_mean / max(r_mean, 1e-6), *gain_clip)
        bgain = np.clip(g_mean / max(b_mean, 1e-6), *gain_clip)
    return float(rgain), float(bgain), 1.0, 1.0


def awb(raw, method="dynamic_threshold", bayer_format="RGGB",
        max_val=1023.0):
    """统一入口: method in {gray_world, perfect_reflector, dynamic_threshold}"""
    fn = {
        "gray_world": awb_gray_world,
        "perfect_reflector": awb_perfect_reflector,
        "dynamic_threshold": awb_dynamic_threshold,
    }[method]
    return fn(raw, bayer_format, max_val)


def awb_robust(raw, bayer_format="RGGB", max_val=1023.0,
               spread_thresh=0.30, extreme_gain=2.2, gain_clip=(0.25, 4.0)):
    """多假设一致性 + 黑体轨迹约束的鲁棒 AWB(返回诊断 dict)。

    【逻辑】
    1. 多假设: 并行跑灰世界/完美反射/动态阈值三个估计器, 在 log
       增益空间比较一致性(增益是乘性的, log 空间的差才是"差几倍");
    2. 一致(最大 log 差 < spread_thresh)且增益幅度现实
       (|ln(gain)| 对应 < extreme_gain 倍) -> 场景可信, 取三者中位数
       (中位数对单个估计器翻车免疫);
    3. 不一致 或 增益极端(典型: 大面积单色物体如草地/大海 ——
       三个估计器会共享同一失效模式, 一致地错!) -> 退守物理先验:
       黑体光源下 sensor 的 R/B 增益近似落在 R×B≈1 的轨迹上
       (G 归一), 把中位数估计投影到轨迹:

           t = sqrt(r_med / b_med)  ->  (t, 1/t)

       【推导】log 空间最小二乘: min (ln r - x)^2 + (ln b + x)^2,
       d/dx = 0 -> x = (ln r - ln b)/2 -> t = sqrt(r/b)。

    "一致地错"是单看一致性的盲区: 草地场景里三个估计器都基于
    "场景均值偏绿"给出同样极端的增益, spread=0 但全错。增益幅度
    检查堵住这个漏洞 —— 物理上不可能的估计无论多一致都不可信。

    Returns
    -------
    dict: rgain/bgain/grgain/gbgain, confidence("high"/"low"),
          spread(最大 log 差), projected(是否轨迹投影),
          candidates(三个估计器的原始结果)
    """
    cands = {}
    for name in ("gray_world", "perfect_reflector", "dynamic_threshold"):
        g = awb(raw, name, bayer_format, max_val)
        cands[name] = (float(g[0]), float(g[1]))

    ls = np.log([[c[0], c[1]] for c in cands.values()])
    spread = float(max(ls[:, 0].max() - ls[:, 0].min(),
                       ls[:, 1].max() - ls[:, 1].min()))
    r_med = float(np.median([c[0] for c in cands.values()]))
    b_med = float(np.median([c[1] for c in cands.values()]))

    extreme = max(r_med, b_med, 1.0 / max(r_med, 1e-6),
                  1.0 / max(b_med, 1e-6)) > extreme_gain
    if spread <= spread_thresh and not extreme:
        return {"rgain": r_med, "bgain": b_med, "grgain": 1.0, "gbgain": 1.0,
                "confidence": "high", "spread": spread, "projected": False,
                "candidates": cands}
    t = float(np.clip(np.sqrt(r_med / max(b_med, 1e-6)),
                      gain_clip[0], gain_clip[1]))
    return {"rgain": t, "bgain": 1.0 / t, "grgain": 1.0, "gbgain": 1.0,
            "confidence": "low", "spread": spread, "projected": True,
            "candidates": cands}
