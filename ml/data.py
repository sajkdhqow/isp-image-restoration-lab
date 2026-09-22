# -*- coding: utf-8 -*-
"""
配对训练数据合成 —— 低照度 RAW -> 正常照度 sRGB
================================================

SID 用 Sony/Fuji 真机短曝(暗)-长曝(亮)图像对(18GB); 我们用
tools.simulate_raw 合成等效数据(求职 repo 不背负大数据集, 原理等价):

    输入: simulate_raw(scene, exposure_scale=低) -> BLC -> pack 4ch
          + 曝光增益压缩(见 model.py 说明)
    GT:   场景 sRGB 真值(合成数据的红利 —— 我们手里有真值)

【关键教训: 单场景训练 = 背答案】
最初版本用单一场景的 24 个翻转变体训练, val_l1 很好看, 但 held-out
场景上色度推断崩溃(彩虹色块) —— 模型背下了该场景的色彩布局。
修正: 每个样本在线生成随机场景(随机配色/布局/渐变/曝光), 等效
无限训练集, 迫使网络学"成像物理的逆"而不是记忆场景。

真实差距(面试要主动讲): 真实 RAW 的噪声模型(PRN/DSNU/行噪声)、
镜头特性、动态范围都比仿真复杂, 合成数据训出的模型只演示方法,
不迁移到真机。
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from tools.simulate_raw import simulate_raw

BLACK_LEVEL = 64.0
MAX_VAL = 1023.0
SCENE_SIZE = 256   # 在线场景尺寸(全分辨率), crop 前的画布


def pack_bayer(blc_raw):
    """(H, W) Bayer -> (H/2, W/2, 4) packed, 通道序 R/Gr/Gb/B。

    每个 2x2 Bayer 单元无损重排为一个 4ch 位置, 没有信息损失 ——
    这是网络吃 RAW 的标准方式(SID 同款)。
    """
    h, w = blc_raw.shape
    R = blc_raw[0::2, 0::2]
    Gr = blc_raw[0::2, 1::2]
    Gb = blc_raw[1::2, 0::2]
    B = blc_raw[1::2, 1::2]
    return np.stack([R, Gr, Gb, B], axis=-1)


def make_lowlight_input(scene_u8, exposure_scale, seed, gain_ref=0.35,
                        gain_max=48.0, sigma_read=3.0, shot=0.08):
    """低照度 RAW -> 归一化 packed 输入 (H/2, W/2, 4) float [0,1]。

    噪声默认值取 SID 风格极端低照度(读出噪声 3DN + 强散粒噪声):
    温和噪声下标定完善的传统 pipeline 接近最优(它全知前向模型),
    ML 的价值要在传统方法崩溃的噪声水平才显现。
    """
    raw = simulate_raw(scene_u8, bits=10, black_level=BLACK_LEVEL,
                       analog_gain=1.0, exposure_scale=exposure_scale,
                       dead_ratio=0, hot_ratio=0, sigma_read=sigma_read,
                       shot=shot, seed=seed)
    packed = (pack_bayer(raw) - BLACK_LEVEL) / MAX_VAL
    # 曝光增益压缩(SID 技巧): in = lin * gain_ref, 所有曝光的输入
    # 亮度归一到同一尺度, 网络面对的唯一变量是噪声大小 —— 它不需要
    # 从数据里猜"曝光是多少"。gain_max 必须 >= gain_ref/最低曝光,
    # 否则截断会让最暗档输入变暗, 引入"曝光歧义"(实测色度恢复变差)
    g = np.clip(gain_ref / max(exposure_scale, 1e-6), 1.0, gain_max)
    x = np.clip(packed * g, 0.0, 1.0)
    # 第 5 通道: 归一化增益(log 域), 曝光元数据条件化(ISO-aware)
    g_ch = np.full(x.shape[:2], np.log2(g) / max(np.log2(gain_max), 1e-6),
                   np.float32)
    return np.concatenate([x, g_ch[..., None]], axis=-1), g


def random_scene(rng, h=SCENE_SIZE, w=SCENE_SIZE):
    """生成随机合成场景(sRGB uint8) —— 每次调用布局配色都不同。

    元素池: 随机方向渐变背景 / 随机色块(位置大小颜色随机) /
    随机亮暗圆盘 / 随机细条纹。多样性是泛化的根本。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)

    # 随机方向/强度渐变背景
    ang = rng.uniform(0, np.pi)
    base = rng.uniform(0.10, 0.35)
    amp = rng.uniform(0.15, 0.45)
    g = base + amp * (np.cos(ang) * xx + np.sin(ang) * yy) / max(h, w)
    tint = rng.uniform(0.9, 1.1, 3)
    img[..., 0] = g * tint[0]
    img[..., 1] = g * tint[1]
    img[..., 2] = g * tint[2]

    # 随机色块 4~8 个
    for _ in range(int(rng.integers(4, 9))):
        c = rng.uniform(0.05, 0.95, 3)
        ch = int(rng.integers(30, 110))
        cw = int(rng.integers(30, 110))
        r0 = int(rng.integers(0, h - ch))
        c0 = int(rng.integers(0, w - cw))
        img[r0:r0 + ch, c0:c0 + cw] = c

    # 随机圆盘 1~2 个(随机亮/暗)
    for _ in range(int(rng.integers(1, 3))):
        cy, cx = rng.uniform(40, h - 40, 2)
        rad = rng.uniform(25, 70)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2
        img[mask] = rng.uniform(0.05, 0.95)

    # 随机细条纹(检验去马赛克/锯齿)
    if rng.random() < 0.7:
        r0 = int(rng.integers(20, h - 40))
        c0 = int(rng.integers(0, w - 160))
        img[r0:r0 + 6, c0:c0 + 160] = 0.05
        img[r0 + 6:r0 + 12, c0:c0 + 160] = 0.90

    img += rng.normal(0, 0.004, img.shape)
    lin = np.clip(img, 0, 1)
    return (lin ** (1 / 2.2) * 255).astype(np.uint8)


class LowLightDataset(Dataset):
    """在线随机场景 + 在线 crop 的配对数据集(等效无限数据)。

    每个样本: 随机场景 -> 随机曝光仿真正常照度 GT 与低照度 RAW ->
    packed 4ch crop。同一样本反复取到时场景/曝光/噪声/位置全不同。
    """

    def __init__(self, crop_packed=64, exposure_range=(0.01, 0.08),
                 seed=0, samples=4096):
        self.crop = crop_packed
        self.exposure_range = exposure_range
        self.rng = np.random.default_rng(seed)
        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        scene = random_scene(self.rng)
        exp = float(self.rng.uniform(*self.exposure_range))
        raw_seed = int(self.rng.integers(0, 2 ** 31))
        x, _ = make_lowlight_input(scene, exp, seed=raw_seed)
        h2, w2 = x.shape[:2]
        cy = int(self.rng.integers(0, h2 - self.crop + 1))
        cx = int(self.rng.integers(0, w2 - self.crop + 1))
        xin = torch.from_numpy(
            x[cy:cy + self.crop, cx:cx + self.crop].transpose(2, 0, 1)
        ).float()
        g = torch.from_numpy(
            (scene.astype(np.float32) / 255.0)[
                cy * 2:(cy + self.crop) * 2,
                cx * 2:(cx + self.crop) * 2].transpose(2, 0, 1)
        ).float()
        return xin, g


def load_scene():
    """main.py 的合成演示场景(评估用)。"""
    import cv2
    from isp.utils import imread_unicode
    scene_path = os.path.join(ROOT, "data", "scene.png")
    return cv2.cvtColor(imread_unicode(scene_path), cv2.COLOR_BGR2RGB)
