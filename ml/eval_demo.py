# -*- coding: utf-8 -*-
"""
传统 ISP vs 端到端 U-Net 低照度对比
====================================

公平对比的设计:
- 同一帧低照度 RAW(exposure_scale=0.05, 约正常照度的 1/20)
- 传统路径: BLC 后用数字增益放大回正常亮度再跑完整 pipeline
  (等效于 AE 把模拟增益拉满的代价: 信号和噪声一起放大, 高光 clip)
- U-Net 路径: 低照度 RAW 直接输出 sRGB(训练时见过该噪声/亮度分布)
- GT: 场景真值(合成数据的红利)

运行:  python ml/eval_demo.py   (需先跑 ml/train.py 生成权重)
输出:  output/ml_compare.png + PSNR 打印
"""

import os
import sys

import numpy as np
import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ml.model import LightUNet
from ml.data import pack_bayer, make_lowlight_input, load_scene, BLACK_LEVEL, MAX_VAL
from tools.simulate_raw import simulate_raw
from isp.utils import save_rgb, psnr
from isp.pipeline import run_pipeline
from isp.gamma import srgb_encode

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)
SEED = 555
GAIN_REF = 0.35    # 与训练一致的增益压缩参考


def build_test_scene(h=512, w=512):
    """held-out 测试场景: 训练时从未见过的布局与配色。

    评估必须用训练没见过的场景, 否则模型可能靠记忆场景拿分,
    对比不公平(这是 ML 评估的基本纪律)。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    # 斜向渐变背景(与训练场景的对角渐变方向不同)
    g = 0.15 + 0.45 * (xx + yy) / (h + w)
    img[..., 0] = g * 1.05
    img[..., 1] = g
    img[..., 2] = g * 0.95
    # 竖直色条(训练场景是 8 块方阵, 这里是 6 条竖条, 配色也不同)
    colors = [(0.15, 0.35, 0.85), (0.85, 0.75, 0.10), (0.10, 0.70, 0.45),
              (0.80, 0.20, 0.55), (0.95, 0.45, 0.10), (0.40, 0.15, 0.70)]
    bw = w // len(colors)
    for i, c in enumerate(colors):
        img[40:200, i * bw:(i + 1) * bw] = c
    # 亮圆盘(右下, 训练场景的圆在别处)
    mask = (yy - 370) ** 2 + (xx - 380) ** 2 < 70 ** 2
    img[mask] = 0.92
    # 细横条纹(中部, 检验摩尔纹/锯齿)
    img[300:308, 60:240] = 0.05
    img[308:316, 60:240] = 0.85
    img += np.random.default_rng(3).normal(0, 0.004, img.shape)
    return (np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def compare_at(scene_u8, exp_low, tag, net):
    """单档曝光下的完整对比, 返回各路径 PSNR。"""
    h, w = scene_u8.shape[:2]
    raw_low = simulate_raw(scene_u8, bits=10, black_level=BLACK_LEVEL,
                           analog_gain=1.0, exposure_scale=exp_low,
                           dead_ratio=0, hot_ratio=0, sigma_read=3.0,
                           shot=0.08, seed=SEED)
    xin, g = make_lowlight_input(scene_u8, exp_low, seed=SEED,
                                 gain_ref=GAIN_REF)
    with torch.no_grad():
        pred = net(torch.from_numpy(
            xin.transpose(2, 0, 1)[None]).float())[0].numpy().transpose(1, 2, 0)
    pred = np.clip(pred, 0, 1)

    boosted = np.clip(raw_low - BLACK_LEVEL, 0, None) / exp_low + BLACK_LEVEL
    boosted = np.clip(boosted, 0, MAX_VAL)
    white_raw = simulate_raw(np.full((512, 512, 3), 205, np.uint8),
                             bits=10, black_level=BLACK_LEVEL,
                             analog_gain=0.72, dead_ratio=0, hot_ratio=0,
                             sigma_read=0.5, shot=0.0, seed=7)
    c = 256
    lsc_target = float(np.median(white_raw[c - 32:c + 32, c - 32:c + 32]))
    from isp.lsc import calibrate_lsc_grid
    lsc_gains = calibrate_lsc_grid(white_raw, lsc_target, grid=(33, 33),
                                   max_gain=16.0)
    trad, _ = run_pipeline(boosted.copy(), bayer_format="RGGB",
                           max_val=MAX_VAL, black_level=BLACK_LEVEL,
                           lsc_gains=lsc_gains, capture_stages=True)
    trad = np.clip(trad, 0, 1)

    gt = scene_u8.astype(np.float32) / 255.0
    p_trad = psnr(gt, trad)
    p_unet = psnr(gt, pred)

    # 可视化: GT / 传统 / U-Net / 低照度输入
    inp_gray = pack_bayer(np.clip(raw_low - BLACK_LEVEL, 0, MAX_VAL)
                          / (MAX_VAL - BLACK_LEVEL)).mean(-1)
    inp_vis = srgb_encode(np.clip(inp_gray * g, 0, 1)[..., None]
                          .repeat(3, -1), max_val=1.0)
    inp_vis = np.repeat(np.repeat(inp_vis, 2, axis=0), 2, axis=1)
    panel = np.concatenate([gt, trad, pred, inp_vis], axis=1)
    save_rgb(os.path.join(OUT, f"ml_compare_{tag}.png"), panel)
    return p_trad, p_unet


def main():
    net = LightUNet()
    ckpt = os.path.join(ROOT, "ml", "unet_lowlight.pt")
    net.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    net.eval()

    # held-out 场景(训练从未见过) x 两档曝光
    test_scene = build_test_scene()
    save_rgb(os.path.join(OUT, "ml_test_scene_gt.png"), test_scene)
    rows = []
    for exp, tag in ((0.02, "exp002"), (0.01, "exp001"), (0.0075, "exp00075")):
        p_trad, p_unet = compare_at(test_scene, exp, tag, net)
        rows.append((exp, tag, p_trad, p_unet))
        print(f"exposure={exp:.4f} (1/{1 / exp:.0f} 正常照度)  "
              f"[{tag}]  传统ISP={p_trad:.2f} dB  "
              f"U-Net={p_unet:.2f} dB  差值={p_unet - p_trad:+.2f} dB")
    print("saved: output/ml_compare_exp002/exp001/exp00075.png")
    print("(每张左起: GT / 传统ISP / U-Net / 低照度输入)")


if __name__ == "__main__":
    main()
