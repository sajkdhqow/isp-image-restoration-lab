# -*- coding: utf-8 -*-
"""
工具函数: RAW 图读写 / Bayer 通道拆分合并 / MIPI 打包解包 / 可视化辅助
======================================================================

对应视频: 《ISP--MIPI数据与RAW数据的转换》《mipiToRaw-大端小端》

【面试要点】
1. Sensor 输出的 RAW 图是单通道 Bayer 马赛克图, 常见位深 8/10/12/14/16bit。
2. MIPI CSI-2 传输时为了节省带宽会对 RAW 做打包(RAW10: 4像素塞进5字节),
   ISP 收到后第一件事就是解包还原成标准位深对齐的 RAW。
3. 大小端: MIPI 打包内部按小端语义组装字节, 解包时必须先约定字节序,
   面试官常拿"10bit 图为什么和 16bit 图一样大"来考察对打包的理解。
"""

import os

import cv2
import numpy as np

BAYER_FORMATS = ("RGGB", "BGGR", "GRBG", "GBRG")


# ---------------------------------------------------------------------------
# 中文路径安全的图像读写
# ---------------------------------------------------------------------------
# 【工程坑点/面试素材】Windows 下 OpenCV 的 imread/imwrite 内部用 ANSI 编码
# fopen, 遇到中文路径(如 D:\...\求职\)会静默失败(返回 False/None 不报错)。
# 通用解法: imencode 到内存 buffer, 再用 numpy/Python 原生 IO 落盘。
def imwrite_unicode(path, img):
    """支持中文路径的 cv2.imwrite 替代。"""
    ext = os.path.splitext(str(path))[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cv2.imencode failed: {path}")
    buf.tofile(str(path))
    return True


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的 cv2.imread 替代。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"cv2.imdecode failed: {path}")
    return img


# ---------------------------------------------------------------------------
# RAW 读写
# ---------------------------------------------------------------------------
def read_raw(filepath, width, height, bits=10, bayer_format="RGGB"):
    """读取未打包的平面 RAW 文件(每像素直接按 bits 对齐存储)。

    bits <= 8 时按 uint8 读取; 其余按 uint16 读取(高位对齐, 低位补零)。
    """
    dt = np.uint8 if bits <= 8 else np.uint16
    raw = np.fromfile(filepath, dtype=dt)
    if raw.size != width * height:
        raise ValueError(
            f"RAW size mismatch: file has {raw.size} pixels, "
            f"expect {width * height} ({width}x{height})"
        )
    raw = raw.reshape(height, width).astype(np.float32)
    if bits > 8:
        # 多 bit 数据常以 16bit 容器存放, 若实际数据在低位需左移对齐,
        #这里默认已左对齐(海思等平台常见), 使用者可按平台调整。
        pass
    return raw


def write_raw(img, filepath, bits=10):
    """将单通道 RAW 图保存为平面二进制文件。"""
    max_val = (1 << bits) - 1
    dt = np.uint8 if bits <= 8 else np.uint16
    np.clip(img, 0, max_val).astype(dt).tofile(filepath)


# ---------------------------------------------------------------------------
# MIPI 打包/解包 (RAW10)
# ---------------------------------------------------------------------------
def mipi_raw10_unpack(data, width, height):
    """MIPI RAW10 解包: 每 4 个像素打包进 5 个字节。

    字节布局(标准 MIPI CSI-2):
        b0 = p0[7:0], b1 = p1[7:0], b2 = p2[7:0], b3 = p3[7:0]
        b4 = [p3[9:8], p2[9:8], p1[9:8], p0[9:8]]  (高 2bit 从低到高排布)

    Parameters
    ----------
    data : np.ndarray (uint8, 1D)
        打包数据流, 长度 = height * ceil(width/4) * 5
    """
    data = np.asarray(data, dtype=np.uint8).ravel()
    pix_grp = width // 4
    rem = width % 4
    grp_bytes = data.reshape(-1, 5) if rem == 0 else None
    if rem != 0:
        # 行尾有残余像素时按整行打包规则逐行解, 简化处理: 先按整组解,
        # 行尾残余字节单独解析(工程上多数 sensor 宽度是 4 的倍数)。
        raise ValueError("本实现要求 width 为 4 的倍数 (demo 场景足够)")

    total_grp = grp_bytes.shape[0]
    h_groups = height * pix_grp
    if total_grp < h_groups:
        raise ValueError("packed data too small")

    b0 = grp_bytes[:h_groups, 0].astype(np.uint16)
    b1 = grp_bytes[:h_groups, 1].astype(np.uint16)
    b2 = grp_bytes[:h_groups, 2].astype(np.uint16)
    b3 = grp_bytes[:h_groups, 3].astype(np.uint16)
    b4 = grp_bytes[:h_groups, 4].astype(np.uint16)

    p0 = b0 | ((b4 & 0x03) << 8)
    p1 = b1 | ((b4 & 0x0C) << 6)
    p2 = b2 | ((b4 & 0x30) << 4)
    p3 = b3 | ((b4 & 0xC0) << 2)

    # (h_groups, 4) -> (h, w)
    pix = np.stack([p0, p1, p2, p3], axis=1).reshape(height, pix_grp * 4)
    return pix.astype(np.float32)


def mipi_raw10_pack(raw):
    """RAW10 打包: 与 mipi_raw10_unpack 互逆, 4 像素 -> 5 字节。"""
    raw = np.asarray(raw)
    h, w = raw.shape
    if w % 4 != 0:
        raise ValueError("width 必须是 4 的倍数")
    p = raw.astype(np.uint16).reshape(h, w // 4, 4)
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    high = ((p3 >> 8) & 0x03) << 6 | ((p2 >> 8) & 0x03) << 4 \
        | ((p1 >> 8) & 0x03) << 2 | ((p0 >> 8) & 0x03)
    out = np.empty((h, w // 4, 5), dtype=np.uint8)
    out[..., 0], out[..., 1] = p0.astype(np.uint8), p1.astype(np.uint8)
    out[..., 2], out[..., 3] = p2.astype(np.uint8), p3.astype(np.uint8)
    out[..., 4] = high.astype(np.uint8)
    return out.reshape(h, -1).ravel()


# ---------------------------------------------------------------------------
# Bayer 通道拆分 / 合并
# ---------------------------------------------------------------------------
def _channel_masks(shape, bayer_format):
    """生成 R / G / B 的布尔掩码。

    RGGB: (0,0)=R (0,1)=Gr (1,0)=Gb (1,1)=B, 其余格式由 2x2 平移得到。
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r0, c0 = 0, 0
    if bayer_format == "BGGR":
        r0, c0 = 1, 1
    elif bayer_format == "GRBG":
        r0, c0 = 0, 1
    elif bayer_format == "GBRG":
        r0, c0 = 1, 0
    r_mask = ((yy % 2 == r0 % 2) & (xx % 2 == c0 % 2))
    b_mask = ((yy % 2 == (r0 + 1) % 2) & (xx % 2 == (c0 + 1) % 2))
    g_mask = ~(r_mask | b_mask)
    return r_mask, g_mask, b_mask


def split_bayer(raw, bayer_format="RGGB"):
    """把 Bayer RAW 拆成 R, Gr, Gb, B 四个半分辨率通道(下采样)。"""
    r_mask, g_mask, b_mask = _channel_masks(raw.shape, bayer_format)
    yy, xx = np.mgrid[0:raw.shape[0], 0:raw.shape[1]]
    gr_mask = g_mask & (yy % 2 == 0)
    gb_mask = g_mask & (yy % 2 == 1)
    R = raw[r_mask]
    Gr = raw[gr_mask]
    Gb = raw[gb_mask]
    B = raw[b_mask]
    return (R.reshape(raw.shape[0] // 2, raw.shape[1] // 2),
            Gr.reshape(raw.shape[0] // 2, raw.shape[1] // 2),
            Gb.reshape(raw.shape[0] // 2, raw.shape[1] // 2),
            B.reshape(raw.shape[0] // 2, raw.shape[1] // 2))


def merge_bayer(R, Gr, Gb, B, bayer_format="RGGB"):
    """把 R, Gr, Gb, B 四个半分辨率通道合成一整幅 Bayer RAW。"""
    h2, w2 = R.shape
    full = np.zeros((h2 * 2, w2 * 2), dtype=np.float32)
    pos = {
        "RGGB": {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 0), "B": (1, 1)},
        "BGGR": {"B": (0, 0), "Gb": (0, 1), "Gr": (1, 0), "R": (1, 1)},
        "GRBG": {"Gr": (0, 0), "R": (0, 1), "B": (1, 0), "Gb": (1, 1)},
        "GBRG": {"Gb": (0, 0), "B": (0, 1), "R": (1, 0), "Gr": (1, 1)},
    }[bayer_format]
    full[pos["R"][0]::2, pos["R"][1]::2] = R
    full[pos["Gr"][0]::2, pos["Gr"][1]::2] = Gr
    full[pos["Gb"][0]::2, pos["Gb"][1]::2] = Gb
    full[pos["B"][0]::2, pos["B"][1]::2] = B
    return full


def apply_channel_gains(raw, rgain, bgain, grgain=1.0, gbgain=1.0,
                        bayer_format="RGGB", max_val=1023.0):
    """在 RAW 域按通道施加白平衡增益(AWB 的输出就干这个)。"""
    R, Gr, Gb, B = split_bayer(raw, bayer_format)
    R = np.clip(R * rgain, 0, max_val)
    B = np.clip(B * bgain, 0, max_val)
    Gr = np.clip(Gr * grgain, 0, max_val)
    Gb = np.clip(Gb * gbgain, 0, max_val)
    return merge_bayer(R, Gr, Gb, B, bayer_format)


# ---------------------------------------------------------------------------
# 可视化辅助
# ---------------------------------------------------------------------------
def to_display(img, bits=10, gamma_on=True):
    """把 RAW/浮点图转成 8bit 可显示图(内部先按位深归一化)。"""
    img = img.astype(np.float32)
    img = np.clip(img / ((1 << bits) - 1), 0, 1)
    if gamma_on:
        img = np.power(img, 1 / 2.2)
    return (img * 255).astype(np.uint8)


def save_gray(path, img, bits=10):
    imwrite_unicode(str(path), to_display(img, bits, gamma_on=True))


def save_rgb(path, img):
    """保存 float RGB [0,1] 图像为 8bit png。"""
    imwrite_unicode(str(path), cv2.cvtColor(
        (np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def psnr(ref, test, peak=1.0):
    """计算 PSNR(dB)。ref/test 为同尺寸 float 数组, peak 为满量程。"""
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    mse = np.mean((ref - test) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(peak ** 2 / mse)
