# -*- coding: utf-8 -*-
"""
NR 基础篇 —— 空域滤波器大全
=============================

对应视频: 《NR基础篇上——均值/高斯/双边/NLM》
          《NR基础篇下——中值/多级中值/混合中值/加权中值/中值有理》

【噪声分类与滤波器选型】
- 高斯噪声(连续灰度扰动): 线性滤波(均值/高斯)有效;
- 椒盐噪声(孤立脉冲): 顺序统计滤波(中值族)有效;
- 既要去噪又要保边: 双边/NLM/多级中值族。

线性滤波器(均值/高斯) = 加权求和, 对高斯噪声最优(Cramer-Rao 下界),
但边缘也被平均 -> 模糊;
非线性滤波器(中值族) = 排序统计, 能完全剔除脉冲(崩溃点 50%),
保边能力强, 但对高斯噪声抑制效率低于线性滤波。

【本模块实现清单】
线性:  mean_filter, gaussian_filter, bilateral_filter, nlm_denoise
中值族: median_filter, mlm_filter(多级中值), mhmf_filter(多级中值混合),
        weighted_median(加权中值), median_rational_filter(中值有理)

【面试要点】
1. 双边滤波两个核: 空间核(离得远近) × 值域核(长得像不像),
   值域核是保边的根源; 缺点是梯度反转(gradient reversal)伪影;
2. NLM: 以 patch 相似度定权重, exp(-d²/h²), 利用图像自相似性,
   是 BM3D 的思想前身; 搜索窗 + 模板窗两级窗口;
3. 中值滤波保边原理: 排序后脉冲排在两端, 中值不受其影响;
   多级中值(MLM)用 4 个方向线状中值组合, 方向性保边更好;
4. 工程上 RAW/RGB 域降噪常用双边系, YUV 域彩噪用小核中值/高斯。
"""

import numpy as np
from scipy.ndimage import convolve, uniform_filter


# ===========================================================================
# 线性滤波
# ===========================================================================
def mean_filter(img, ksize=3):
    """均值滤波: 邻域等权平均。核越大越模糊。"""
    return uniform_filter(np.asarray(img, np.float32), size=ksize,
                          mode="nearest")


def gaussian_kernel(ksize=5, sigma=1.0):
    """高斯核: G(x,y) = 1/(2πσ²)·exp(-(x²+y²)/2σ²), 归一化。"""
    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_filter(img, ksize=5, sigma=1.0):
    """高斯滤波(可分离: 先横后纵, 计算量 O(k) 而非 O(k²))。"""
    r = ksize // 2
    x = np.arange(-r, r + 1, dtype=np.float32)
    k1 = np.exp(-x ** 2 / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    tmp = convolve(np.asarray(img, np.float32), k1[None, :], mode="nearest")
    return convolve(tmp, k1[:, None], mode="nearest")


def bilateral_filter(img, ksize=5, sigma_spatial=1.5, sigma_range=15.0):
    """双边滤波: 空间权重 × 值域权重。

    W(i,j) ∝ G_spatial(||p-q||) * G_range(|I(p)-I(q)|)
    平坦区 -> 值域权重都接近 1 -> 退化为高斯, 平滑去噪;
    边缘处 -> 值域权重把另一侧像素权重压到 0 -> 保边。
    """
    img = np.asarray(img, np.float32)
    r = ksize // 2
    h, w = img.shape
    pad = np.pad(img, r, mode="edge")

    # 空间核(固定)
    x = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    gs = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_spatial ** 2))

    num = np.zeros_like(img)
    den = np.zeros_like(img)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            neighbor = pad[r + dy: r + dy + h, r + dx: r + dx + w]
            diff = neighbor - img
            gr = np.exp(-(diff ** 2) / (2 * sigma_range ** 2))
            weight = gs[dy + r, dx + r] * gr
            num += weight * neighbor
            den += weight
    return num / np.maximum(den, 1e-8)


def nlm_denoise(img, search_window=7, patch_size=3, h_factor=1.0,
                sigma=None):
    """非局部均值 NLM (Buades 2005)。

    out(p) = Σ_q w(p,q) I(q),  w = exp(-max(0, d²(p,q) - 2σ²) / h²)
    d² = ||P(p) - P(q)||² 为 patch 欧氏距离平方;
    减 2σ² 是噪声方差修正(噪声越大权重越保守)。

    加速: 对每个搜索偏移, patch 距离用"滑窗累加"技巧,
    等价于对 (I_shift - I)² 做 patch 大小的均匀滤波, 复杂度
    O(search² × patch) 而非暴力 O(search² × patch²)。
    """
    img = np.asarray(img, np.float32)
    r_s = search_window // 2
    r_p = patch_size // 2
    if sigma is None:
        from .bnr import estimate_noise_sigma
        sigma = estimate_noise_sigma(img)
    h2 = (h_factor * max(sigma, 1e-3)) ** 2

    h, w = img.shape
    pad = np.pad(img, r_s + r_p, mode="edge")
    # 以 r_s 为界的中心图
    center = pad[r_s + r_p: r_s + r_p + h, r_s + r_p: r_s + r_p + w]

    num = np.zeros_like(img)
    den = np.zeros_like(img)
    for dy in range(-r_s, r_s + 1):
        for dx in range(-r_s, r_s + 1):
            neighbor = pad[r_s + r_p + dy: r_s + r_p + dy + h,
                           r_s + r_p + dx: r_s + r_p + dx + w]
            sq_diff = (neighbor - center) ** 2
            # patch 内累计平方差(均匀滤波=滑窗求和)
            d2 = uniform_filter(sq_diff, size=patch_size, mode="constant")
            weight = np.exp(-np.maximum(d2 - 2 * sigma ** 2, 0.0) / h2)
            if dy == 0 and dx == 0:
                weight = np.where(weight > 0.99, 0.0, weight)
            num += weight * neighbor
            den += weight
    # 中心像素自身权重兜底
    den += 1.0
    num += center
    return num / np.maximum(den, 1e-8)


# ===========================================================================
# 中值族
# ===========================================================================
def _median_windows(img, ksize):
    """滑窗提取 + 逐窗口中值。返回与输入同尺寸的滤波结果。"""
    img = np.asarray(img, np.float32)
    r = ksize // 2
    pad = np.pad(img, r, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(pad, (ksize, ksize))  # (h, w, k, k)
    return np.median(win, axis=(-2, -1))


def median_filter(img, ksize=3):
    """中值滤波: 排序取中位。椒盐噪声克星, 保边。"""
    return _median_windows(img, ksize)


def _directional_medians(img, length=3):
    """四个方向(0°/45°/90°/135°)线状窗口的中值。

    返回 4 个与原图同尺寸的方向中值图。
    """
    img = np.asarray(img, np.float32)
    r = length // 2
    pad = np.pad(img, r, mode="edge")
    h, w = img.shape

    def line_med(offsets):
        acc = []
        for dy, dx in offsets:
            acc.append(pad[r + dy: r + dy + h, r + dx: r + dx + w])
        return np.median(np.stack(acc, 0), axis=0)

    m_v = line_med([(i, 0) for i in range(-r, r + 1)])    # 90°
    m_h = line_med([(0, i) for i in range(-r, r + 1)])    # 0°
    m_d1 = line_med([(i, i) for i in range(-r, r + 1)])   # 135°
    m_d2 = line_med([(i, -i) for i in range(-r, r + 1)])  # 45°
    return m_v, m_h, m_d1, m_d2


def mlm_filter(img, ksize=3, variant="I"):
    """多级中值滤波 MLM (Multilevel Median Filter)。

    第一级: 4 个方向线状中值 m_v, m_h, m_d1, m_d2;
    第二级(variant):
      I  : out = median{ max(m), min(m), x_center }   (经典 MLM-I)
      II : out = median{ median(m_v, m_h, m_d1, m_d2), x_center }
    方向性结构保留更好, 尤其对细线条与角点。
    """
    img = np.asarray(img, np.float32)
    mv, mh, md1, md2 = _directional_medians(img, ksize)
    m = np.stack([mv, mh, md1, md2], 0)
    mx, mn = m.max(0), m.min(0)
    med = np.median(m, 0)
    if variant == "I":
        return np.median(np.stack([mx, mn, img], 0), axis=0)
    return np.median(np.stack([med, img], 0), axis=0)


def mhmf_filter(img, ksize=3):
    """多级中值混合滤波 MHMF: median{ 3x3 中值, MLM-I, 中心 }。

    3x3 中值负责压平坦区噪声, MLM 负责保方向性结构, 中心防过校正。
    """
    img = np.asarray(img, np.float32)
    m3 = _median_windows(img, ksize)
    mlm = mlm_filter(img, ksize, variant="I")
    return np.median(np.stack([m3, mlm, img], 0), axis=0)


def weighted_median_filter(img, ksize=3, center_weight=3):
    """加权中值滤波 WMF: 权重模板 = 像素重复次数。

    权重模板(中心加权):
        [1 1 1]
        [1 c 1]    c = center_weight
        [1 1 1]
    中心权重越大输出越偏向中心像素, 保细节能力越强。
    """
    img = np.asarray(img, np.float32)
    r = ksize // 2
    pad = np.pad(img, r, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(pad, (ksize, ksize))  # (h, w, k, k)
    stack = []
    for i in range(ksize):
        for j in range(ksize):
            w = center_weight if (i == r and j == r) else 1
            stack.extend([win[..., i, j]] * w)
    return np.median(np.stack(stack, 0), axis=0)


def median_rational_filter(img, ksize=3, k=0.2, h=1.0):
    """中值有理混合滤波 MRHF (Median-Rational Hybrid Filter,
    Khriji & Gabbouj 提出的有理函数类非线性滤波, 这里取其标量形式)。

    三个子滤波器:
        Φ1 = 3x3 中值           (去噪主力)
        Φ2 = 多级中值 MLM       (保方向结构)
        Φ3 = 中心加权中值 CWM   (细节参考)

    有理函数校正项(保持直流: 三者一致时校正为零):
        out = Φ2 + [β1(Φ1-Φ2) + β3(Φ3-Φ2)] / (h + k·|Φ1 - Φ3|)

    直觉: |Φ1-Φ3| 大(边缘/脉冲)时分母变大, 校正项被压 -> 输出回退到
    保守的 Φ2, 脉冲不被放大; 三者接近(平坦)时校正退化为加权平均偏差,
    平滑而不过度。k 控制非线性强度, h 防止分母过小。
    """
    img = np.asarray(img, np.float32)
    phi1 = _median_windows(img, ksize)
    mv, mh, md1, md2 = _directional_medians(img, ksize)
    phi2 = np.median(np.stack([mv, mh, md1, md2, img], 0), axis=0)
    phi3 = weighted_median_filter(img, ksize, center_weight=3)
    beta1, beta3 = 1.0, 1.0
    corr = (beta1 * (phi1 - phi2) + beta3 * (phi3 - phi2)) \
        / (h + k * np.abs(phi1 - phi3))
    return phi2 + corr
