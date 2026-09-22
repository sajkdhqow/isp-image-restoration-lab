# -*- coding: utf-8 -*-
"""
条件 Rectified Flow —— 生成式低照度恢复(流匹配)
=================================================

【为什么是流匹配而不是 DDPM】
Diffusion(DDPM)与 Flow 在数学上同源:都是学一个把噪声分布搬运到
数据分布的速度场 v(x_t, t | c)。区别在训练/采样形式:

    DDPM:      随机微分方程采样, 几十~上千步, 训练目标等价
    Rectified Flow(流匹配): 常微分方程 ODE, 直线路径,
               Euler 离散化 1~8 步即可采样, 训练目标就是 MSE

SD3 / Flux 等主流生成模型都已转向流匹配, 且少步采样是 CPU demo
唯一可行的形态 —— 所以本仓库实现 Rectified Flow, 面试时讲
"扩散/流同源, 我用流匹配的少步形态" 即可覆盖两个叙事。

【路径设定】
x_t = (1-t)·x0 + t·ε,  t ∈ [0,1]  (t=0 数据, t=1 噪声)
速度场目标:  v(x_t, t) = ε - x0   (MSE 回归)
采样(Euler ODE, 从 t=1 积分回 t=0):
    x_{t-δ} = x_t - δ · v(x_t, t, c)

【与回归 U-Net 的关系(核心叙事)】
回归模型输出"训练集的均值", 模糊但 PSNR 高; 生成式模型采样"后验
的一个样本", 纹理更锐但可能偏离均值 —— 两者的取舍正是生成式方法
能否进产品 preview 链路的决定因素(延迟/功耗/稳定性)。本模块让
同一任务上两条路线可量化对比。

条件 c: 低照度 packed Bayer 5ch(4 通道 RAW + 增益元数据)双线性
上采样到全分辨率 —— 网络全卷积, 任意尺寸可采样。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import conv_block


class FlowUNet(nn.Module):
    """速度场 v(x_t, t | c)。输入 = x_t(3ch) + t 图(1ch) + 条件(5ch)。

    与回归 LightUNet 的区别: 工作在全分辨率(无 2x 上采样头),
    时间 t 以常数图形式拼入输入(conv 网络对均匀标量场的表达足够
    本 demo 使用, 省掉 FiLM/正弦嵌入的复杂度)。
    """

    def __init__(self, chs=(32, 64, 128), cond_ch=5):
        super().__init__()
        c1, c2, c3 = chs
        self.cond_ch = cond_ch
        self.enc1 = conv_block(3 + 1 + cond_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.pool = nn.AvgPool2d(2)
        self.dec2 = conv_block(c3 + c2, c2)
        self.dec1 = conv_block(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x_t, t, cond):
        """x_t: (B,3,H,W)  t: (B,)  cond: (B,cond_ch,H,W) -> v: (B,3,H,W)"""
        b, _, h, w = x_t.shape
        tmap = t.view(b, 1, 1, 1).expand(-1, 1, h, w)
        x = torch.cat([x_t, tmap, cond], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat(
            [F.interpolate(e3, scale_factor=2, mode="bilinear",
                           align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat(
            [F.interpolate(d2, scale_factor=2, mode="bilinear",
                           align_corners=False), e1], dim=1))
        return self.head(d1)


def make_cond(xin, out_hw):
    """packed 5ch 低照度输入 -> 全分辨率条件场(双线性上采样)。"""
    return F.interpolate(xin, size=out_hw, mode="bilinear",
                         align_corners=False)


@torch.no_grad()
def euler_sample(model, cond, n_steps=4, seed=None, x1=None):
    """从纯噪声积分回数据分布: x_{t-δ} = x_t - δ·v(x_t, t, c)。

    n_steps=1 即"一步流"( instaflow 风格); 步数越多越接近精确 ODE,
    延迟线性增长 —— 这个 trade-off 本身就是评估的一部分。
    x1: 可选, 直接给定初始噪声(reflow 生成配对时必须传入,
    保证 (z, x0) 对应关系)。
    """
    if x1 is None:
        gen = None
        if seed is not None:
            gen = torch.Generator(device=cond.device).manual_seed(seed)
        x1 = torch.randn(cond.shape[0], 3, cond.shape[2], cond.shape[3],
                         generator=gen)
    x = x1.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((cond.shape[0],), 1.0 - i * dt)
        x = x - dt * model(x, t, cond)
    return x.clamp(0.0, 1.0)
