# -*- coding: utf-8 -*-
"""
Demosaic —— 去马赛克 (Demosaicking)
=====================================

对应视频: 《ISP算法精讲——Demosaicking》

【背景】
sensor 只能感知亮度, 通过 CFA(CFA 滤色阵列, 柯达 Bryce Bayer 发明)
让每个像素只透过 R/G/B 中的一种颜色, G 占一半(人眼对绿色最敏感,
绿色也是亮度的主要贡献者)。demosaic 就是把单通道 Bayer 图恢复成
每个像素都有完整 RGB 的全彩图, 是 ISP 中决定清晰度和伪彩的关键环节。

【算法(本仓库实现三种)】
1. 双线性插值(最基础):
   - G 缺失处: 上下左右 4 个 G 取平均;
   - R/B 缺失处: 4 个对角同色取平均;
   - 在 R 位置求 G、在 G 位置求 R 等情形用相邻同色平均。
   问题: 忽略通道间相关性, 高频边缘处出现拉链伪影(zipper)与伪彩。

2. Malvar-He-Cutler (2004) 线性插值升级版:
   核心思想: 假设色差通道 (R-G, B-G) 是带限的(局部低频),
   在双线性基础上加"拉普拉斯校正项", 用 5x5 卷积核一步完成,
   PSNR 比双线性高 5dB+。实现按论文 Table 1 的四个核:
       G at R/B:            [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],
                             [0,0,2,0,0],[0,0,-1,0,0]]/8
       R at G(R行)/B at G(B行): [[0,0,.5,0,0],[0,-1,0,-1,0],[-1,4,5,4,-1],
                             [0,-1,0,-1,0],[0,0,.5,0,0]]/8  (及其转置)
       R at B / B at R:      [[0,0,-1.5,0,0],[0,2,0,2,0],[-1.5,0,6,0,-1.5],
                             [0,2,0,2,0],[0,0,-1.5,0,0]]/8
   核直接作用在 Bayer 马赛克图上(R 项构成拉普拉斯, 平坦区自动抵消)。

3. 边缘方向 + 色差法 (HQLI 思路, 高质量线性插值):
   - G 通道: 在每个 R/B 位置比较水平/垂直梯度, 选梯度小(更平滑)
     的方向插值; 两个方向接近时取平均;
   - R/B 通道: 先得到 G, 再在色差域 (R-G, B-G) 做双线性插值
     (利用通道相关性, 显著减少伪彩)。

【面试要点】
1. 评价: 客观用 PSNR(Kodak24 集上 Malvar ≈ 34dB+), 主观看伪彩
   (false color)与拉链效应(zipper artifact)。
2. 为什么 G 先插且更讲究: G 采样率是 R/B 的两倍, 人眼对亮度(G 主导)
   的空间分辨率高, G 的质量决定整体清晰度。
3. 色差恒定假设: 小邻域内 R-G / B-G 近似常数, 这是色差法的理论根基,
   面试能画出这条推导链条很加分。
4. 工程上还有 adaptive(方向加权/梯度修正)与深度学习(JDD: 联合
   去马赛克+去噪, 如 DJDD 网络)方法, 深度学习方法同时建模噪声,
   低光下优势大 —— 视频中提到了基于学习的思路。
"""

import numpy as np
from scipy.ndimage import convolve


# ---------------------------------------------------------------------------
# 掩码工具
# ---------------------------------------------------------------------------
def _masks(shape, bayer_format="RGGB"):
    from .utils import _channel_masks
    r, g, b = _channel_masks(shape, bayer_format)
    return (r.astype(np.float32), g.astype(np.float32), b.astype(np.float32))


# ---------------------------------------------------------------------------
# 1) 双线性 demosaic
# ---------------------------------------------------------------------------
def demosaic_bilinear(raw, bayer_format="RGGB", max_val=1023.0):
    """双线性去马赛克(带掩码图的归一化均值卷积实现)。

    对每个通道: 缺失值 = 周围已知同色像素的加权平均。
    实现上用 3x3 均值核对"掩码图"和"掩码本身"同时卷积,
    num/den 的比值即是归一化的邻域均值, 等价于分位置的双线性插值。
    """
    r_m, g_m, b_m = _masks(raw.shape, bayer_format)
    k_full = np.ones((3, 3), np.float32) / 9

    def fill(chan_mask):
        masked = raw * chan_mask
        num = convolve(masked, k_full, mode="nearest")
        den = convolve(chan_mask, k_full, mode="nearest")
        return num / np.maximum(den, 1e-6)

    Rf = fill(r_m)
    Gf = fill(g_m)
    Bf = fill(b_m)
    return np.stack([Rf, Gf, Bf], axis=-1).clip(0, max_val)


# ---------------------------------------------------------------------------
# 2) Malvar-He-Cutler 2004
# ---------------------------------------------------------------------------
def demosaic_malvar(raw, bayer_format="RGGB", max_val=1023.0):
    """Malvar (2004) 线性去马赛克, 论文 Table 1 四个 5x5 核。"""
    r_m, g_m, b_m = _masks(raw.shape, bayer_format)

    GR_GB = np.array([[0, 0, -1, 0, 0],
                      [0, 0, 2, 0, 0],
                      [-1, 2, 4, 2, -1],
                      [0, 0, 2, 0, 0],
                      [0, 0, -1, 0, 0]], np.float32) / 8

    RG_RB_BG_BR = np.array([[0, 0, 0.5, 0, 0],
                            [0, -1, 0, -1, 0],
                            [-1, 4, 5, 4, -1],
                            [0, -1, 0, -1, 0],
                            [0, 0, 0.5, 0, 0]], np.float32) / 8
    RG_BR_BG_RB = RG_RB_BG_BR.T

    RB_BB_RR_RR = np.array([[0, 0, -1.5, 0, 0],
                            [0, 2, 0, 2, 0],
                            [-1.5, 0, 6, 0, -1.5],
                            [0, 2, 0, 2, 0],
                            [0, 0, -1.5, 0, 0]], np.float32) / 8

    R = raw * r_m
    G = raw * g_m
    B = raw * b_m

    # G 缺失(R/B 位置)
    G_at_RB = convolve(raw, GR_GB, mode="nearest")
    G = np.where((r_m == 1) | (b_m == 1), G_at_RB, G)

    conv_a = convolve(raw, RG_RB_BG_BR, mode="nearest")
    conv_b = convolve(raw, RG_BR_BG_RB, mode="nearest")
    conv_c = convolve(raw, RB_BB_RR_RR, mode="nearest")

    h, w = raw.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # 依据 Bayer 排列判断: 行属于 R 行 / B 行, 列属于 R 列 / B 列
    row_is_r = (yy % 2 == 0) if bayer_format in ("RGGB", "GRBG") \
        else (yy % 2 == 1)
    col_is_r = (xx % 2 == 0) if bayer_format in ("RGGB", "GBRG") \
        else (xx % 2 == 1)

    at_r = r_m == 1
    at_b = b_m == 1
    at_g = g_m == 1

    # R 在 G 位置: R 行 + B 列 -> 核 a; B 行 + R 列 -> 核 b
    R = np.where(at_g & row_is_r & ~col_is_r, conv_a, R)
    R = np.where(at_g & ~row_is_r & col_is_r, conv_b, R)
    # B 在 G 位置: B 行 + R 列 -> 核 a; R 行 + B 列 -> 核 b
    B = np.where(at_g & ~row_is_r & col_is_r, conv_a, B)
    B = np.where(at_g & row_is_r & ~col_is_r, conv_b, B)
    # R 在 B 位置 / B 在 R 位置: 核 c
    R = np.where(at_b, conv_c, R)
    B = np.where(at_r, conv_c, B)

    return np.stack([R, G, B], axis=-1).clip(0, max_val)


# ---------------------------------------------------------------------------
# 3) 边缘方向 G + 色差域 R/B (HQLI 思路)
# ---------------------------------------------------------------------------
def demosaic_edge_aware(raw, bayer_format="RGGB", max_val=1023.0,
                        eps=1e-6):
    """边缘方向 G 插值 + 色差域 R/B 插值。

    G 插值(在 R/B 位置):
        水平差 Gh = |G_left - G_right|, 垂直差 Gv = |G_up - G_down|
        Gh < Gv: 水平平均; Gv < Gh: 垂直平均; 否则 4 点平均。
    R/B 插值: 色差域 K = R - G, K' = B - G, 对 K/K' 双线性插值后加回 G。
    """
    r_m, g_m, b_m = _masks(raw.shape, bayer_format)

    h, w = raw.shape
    pad = np.pad(raw, 1, mode="edge")

    G_full = np.zeros_like(raw)
    # 已知 G
    G_full = np.where(g_m == 1, raw, G_full)

    # R/B 位置的方向 G 插值
    g_l = pad[1:-1, 0:-2]; g_r = pad[1:-1, 2:]
    g_u = pad[0:-2, 1:-1]; g_d = pad[2:, 1:-1]
    gh = np.abs(g_l - g_r)          # 水平梯度(G 均为同色对? 在 R 行, 左右是 G? RGGB 中 R 行左右是 G ✓; B 行左右是 G ✓)
    gv = np.abs(g_u - g_d)          # 垂直: R 位置的上下是 G ✓, B 位置上下是 G ✓
    g_h_avg = (g_l + g_r) / 2
    g_v_avg = (g_u + g_d) / 2
    g_avg4 = (g_l + g_r + g_u + g_d) / 4
    g_est = np.where(gh < gv, g_h_avg,
                     np.where(gv < gh, g_v_avg, g_avg4))
    G_full = np.where(g_m != 1, g_est, G_full)

    # 色差域 R / B
    Rm = raw * r_m
    Bm = raw * b_m
    Kr = (Rm - np.where(r_m == 1, G_full, 0)) * r_m  # 只在 R 位置有效
    Kb = (Bm - np.where(b_m == 1, G_full, 0)) * b_m

    k_blur = np.ones((3, 3), np.float32) / 9

    def fill(masked, m):
        num = convolve(masked, k_blur, mode="nearest")
        den = convolve(m, k_blur, mode="nearest")
        return num / np.maximum(den, eps)

    Kr_f = fill(Kr, r_m)
    Kb_f = fill(Kb, b_m)
    R_full = G_full + Kr_f
    B_full = G_full + Kb_f
    return np.stack([R_full, G_full, B_full], axis=-1).clip(0, max_val)


def demosaic(raw, method="malvar", bayer_format="RGGB", max_val=1023.0):
    """统一入口: method in {bilinear, malvar, edge_aware}。"""
    fn = {
        "bilinear": demosaic_bilinear,
        "malvar": demosaic_malvar,
        "edge_aware": demosaic_edge_aware,
    }[method]
    return fn(raw, bayer_format, max_val)
