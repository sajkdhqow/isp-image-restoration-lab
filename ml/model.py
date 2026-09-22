# -*- coding: utf-8 -*-
"""
轻量 U-Net —— 端到端 ISP(SID 风格)
===================================

参考: Learning to See in the Dark (CVPR 2018) / cydonia999 的 PyTorch
复现。核心思想: 传统 ISP 是一条人工设计的模块链, 每个模块都在"局部
最优"上叠加误差; 端到端网络直接从 RAW(带噪声/马赛克/镜头缺陷)映射
到 sRGB, 让数据决定中间处理。

本实现的取舍(CPU 训练约束):
- 输入: packed Bayer 4 通道 (R/Gr/Gb/B), 分辨率 H/2 x W/2
        (每 2x2 Bayer 单元变成 1 个 4ch 位置, 无损重排)
- 输出: 全分辨率 sRGB 3 通道(H x W, 2x 上采样)
- 规模: 3 级编码器 + 跳连, 约 0.4M 参数(SID 原版 U-Net ~8M)
- 上采样用 bilinear+conv, 不用反卷积(避免棋盘伪影)

【输入增益压缩( SID 的 gain compression 技巧)】
极低照度下 RAW 信号幅度只有满量程的百分之几, 直接喂网络学"放大"
效率低。按曝光量对输入做归一化增益: in = clip(raw * g, 0, 1),
g = clip(ref / exposure, 1, g_max), 让不同曝光的输入亮度标准化,
网络专注学去噪/去马赛克/色彩恢复。真实部署时 g 由 AE 元数据给出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout, groups=4):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(groups, cout),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(groups, cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class LightUNet(nn.Module):
    """RAW(4ch + 增益元数据 1ch, H/2) -> sRGB(3ch, H)。范围 [0,1]。

    第 5 通道是全图常数的归一化增益(曝光元数据, 真实相机有 ISO/t):
    网络据此调节降噪强度 —— 同样亮度的输入, 高增益(噪声大)多滤,
    低增益(噪声小)少滤, 避免一刀切。
    """

    def __init__(self, chs=(32, 64, 128), in_ch=5):
        super().__init__()
        c1, c2, c3 = chs
        self.enc1 = conv_block(in_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.pool = nn.AvgPool2d(2)
        self.dec2 = conv_block(c3 + c2, c2)
        self.dec1 = conv_block(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)                      # (c1, h, w)
        e2 = self.enc2(self.pool(e1))          # (c2, h/2, w/2)
        e3 = self.enc3(self.pool(e2))          # (c3, h/4, w/4)
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2,
                                                 mode="bilinear",
                                                 align_corners=False), e2],
                                 dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2,
                                                mode="bilinear",
                                                align_corners=False), e1],
                                 dim=1))
        out = self.head(F.interpolate(d1, scale_factor=2, mode="bilinear",
                                      align_corners=False))
        return torch.sigmoid(out)
