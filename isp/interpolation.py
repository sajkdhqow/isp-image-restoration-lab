# -*- coding: utf-8 -*-
"""
图像插值算法 (Image Interpolation)
====================================

对应视频: 《ISP算法精讲——图像插值算法》

【应用场景】
图像放大、旋转、仿射变换后新坐标与原像素网格不对齐, 出现"空穴",
需要插值填补。ISP 中 zoom(electronic zoom)、畸变校正、EWB 等都依赖它。

【空间映射】
- 前向映射: 遍历原图像素, 把它"扔"到目标图, 会产生空穴与重叠;
- 反向映射(工程通用): 遍历目标图像素, 反算它在原图中的坐标再取值,
  保证目标图每个像素都有值。
坐标换算需做像素中心对齐:
    src = (dst + 0.5) * scale - 0.5

【三种插值】
1. 最近邻: 直接取四邻中最近的点。快, 但放大后马赛克/锯齿明显。
2. 双线性: 用周围 2x2 邻域按距离加权平均, 权重 = (1-du)(1-v) 等。
   连续但一阶导不连续, 放大有轻微模糊。
3. 双三次: 用周围 4x4 邻域, 卷积核取三次多项式(参数 A 常取 -0.5):
   对 |x|: W(x) = (A+2)|x|^3 - (A+3)|x|^2 + 1        (|x|<1)
           W(x) = A|x|^3 - 5A|x|^2 + 8A|x| - 4A      (1<=|x|<2)
   保留更多高频细节, 但可能过冲(overshoot), 计算量更大。

【面试要点】
- 为什么反向映射 + 中心对齐, 不做会半像素偏移, 放大缩小时尤为明显;
- 双三次的 A 参数: A=-0.5 与 Photoshop 一致, A=-0.75 边缘更锐但过冲多;
- 面试常考手推双线性公式, 一定要练熟。
"""

import numpy as np


def _undo_coords(dst_shape, src_shape):
    """反向映射: 目标坐标 -> 源浮点坐标(含中心对齐)。"""
    dh, dw = dst_shape
    sh, sw = src_shape
    sy = (np.arange(dh) + 0.5) * (sh / dh) - 0.5
    sx = (np.arange(dw) + 0.5) * (sw / dw) - 0.5
    return sy, sx


def nearest_interpolate(img, scale=None, dst_shape=None):
    """最近邻插值(支持 2D/3D)。"""
    src_shape = img.shape[:2]
    if dst_shape is None:
        dst_shape = (int(round(src_shape[0] * scale)),
                     int(round(src_shape[1] * scale)))
    sy, sx = _undo_coords(dst_shape, src_shape)
    yi = np.clip(np.rint(sy).astype(int), 0, src_shape[0] - 1)
    xi = np.clip(np.rint(sx).astype(int), 0, src_shape[1] - 1)
    return img[yi[:, None], xi[None, :]]


def bilinear_interpolate(img, scale=None, dst_shape=None):
    """双线性插值。

    P = (1-du)(1-dv)*P00 + du(1-dv)*P01 + (1-du)dv*P10 + du*dv*P11
    其中 du/dv 是浮点坐标相对左上角整数坐标的偏移。
    """
    src_shape = img.shape[:2]
    if dst_shape is None:
        dst_shape = (int(round(src_shape[0] * scale)),
                     int(round(src_shape[1] * scale)))
    sy, sx = _undo_coords(dst_shape, src_shape)
    y0 = np.clip(np.floor(sy).astype(int), 0, src_shape[0] - 2)
    x0 = np.clip(np.floor(sx).astype(int), 0, src_shape[1] - 2)
    y1, x1 = y0 + 1, x0 + 1
    dy = (sy - y0)[:, None]          # (dh, 1)
    dx = (sx - x0)[None, :]          # (1, dw)
    y0c, y1c = y0[:, None], y1[:, None]
    x0c, x1c = x0[None, :], x1[None, :]
    # 权重统一成 (dh, dw, 1) 以兼容彩色图广播
    if img.ndim == 2:
        p00 = img[y0c, x0c]; p01 = img[y0c, x1c]
        p10 = img[y1c, x0c]; p11 = img[y1c, x1c]
        w00, w10 = (1 - dy) * (1 - dx), dy * (1 - dx)
        w01, w11 = (1 - dy) * dx, dy * dx
    else:
        p00 = img[y0c, x0c, :]; p01 = img[y0c, x1c, :]
        p10 = img[y1c, x0c, :]; p11 = img[y1c, x1c, :]
        dy = dy[..., None]          # (dh, 1, 1)
        dx = dx[..., None]          # (1, dw, 1)
        w00, w10 = (1 - dy) * (1 - dx), dy * (1 - dx)
        w01, w11 = (1 - dy) * dx, dy * dx
    out = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11
    return out.astype(img.dtype)


def _cubic_kernel(x, A=-0.5):
    """双三次卷积核 Keys' family。"""
    ax = np.abs(x)
    w = np.zeros_like(ax)
    m1 = ax < 1
    m2 = (ax >= 1) & (ax < 2)
    w[m1] = ((A + 2) * ax[m1] ** 3 - (A + 3) * ax[m1] ** 2 + 1)
    w[m2] = (A * ax[m2] ** 3 - 5 * A * ax[m2] ** 2
             + 8 * A * ax[m2] - 4 * A)
    return w


def bicubic_interpolate(img, scale=None, dst_shape=None, A=-0.5):
    """双三次插值(可分离实现: 先横向后纵向, 各做 4 点加权和)。"""
    src_shape = img.shape[:2]
    if dst_shape is None:
        dst_shape = (int(round(src_shape[0] * scale)),
                     int(round(src_shape[1] * scale)))
    sy, sx = _undo_coords(dst_shape, src_shape)

    # --- 水平方向 ---
    x0 = np.floor(sx).astype(int)
    tx = sx - x0
    # 4 个横向邻居相对权重
    wx = np.stack([_cubic_kernel(tx + 1, A), _cubic_kernel(tx, A),
                   _cubic_kernel(1 - tx, A), _cubic_kernel(2 - tx, A)],
                  axis=0)  # (4, dw)
    xi = np.clip(np.stack([x0 - 1, x0, x0 + 1, x0 + 2]), 0, src_shape[1] - 1)
    # --- 垂直方向 ---
    y0 = np.floor(sy).astype(int)
    ty = sy - y0
    wy = np.stack([_cubic_kernel(ty + 1, A), _cubic_kernel(ty, A),
                   _cubic_kernel(1 - ty, A), _cubic_kernel(2 - ty, A)],
                  axis=0)  # (4, dh)
    yi = np.clip(np.stack([y0 - 1, y0, y0 + 1, y0 + 2]), 0, src_shape[0] - 1)

    # 可分离两步: 先横向 4 点加权和 (sh, dw), 再纵向 4 点加权和 (dh, dw)
    tail = (1,) * (img.ndim - 2)  # 彩色图时给权重补通道维
    horiz = np.zeros((src_shape[0], len(sx)) + img.shape[2:])
    for k in range(4):
        # 注意: 写成 img[:, xi[k]] 会让 advanced 索引维度跑到最前面
        # (numpy 经典陷阱), 用 take(axis=1)才能保持 (sh, dw) 顺序
        horiz += img.take(xi[k], axis=1) * wx[k].reshape((1, len(sx)) + tail)
    out = np.zeros((len(sy), len(sx)) + img.shape[2:])
    for k in range(4):
        out += horiz.take(yi[k], axis=0) * wy[k].reshape((len(sy), 1) + tail)
    return out.astype(img.dtype)
