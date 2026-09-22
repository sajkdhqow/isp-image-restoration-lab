# -*- coding: utf-8 -*-
"""
训练端到端低照度 ISP U-Net
==========================

CPU 训练配置(求职 repo, 演示原理为主):
- 输入: packed Bayer 4ch crop(64x64 packed = 128x128 全分辨率)
- 损失: L1 + 0.1*MS-SSIM 简化版(纯 L1 会偏糊, 加梯度项保边缘)
- 规模: ~0.4M 参数, 数百 iter 即可见效

运行:  python ml/train.py            # 默认 800 iter
       python ml/train.py --iters 2000
输出:  ml/unet_lowlight.pt + 训练曲线数据
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.model import LightUNet
from ml.data import LowLightDataset, random_scene, make_lowlight_input


def edge_loss(pred, gt):
    """L1 梯度损失: 抑制 L1 的过糊倾向(保边缘的廉价替代)。"""
    def grad_y(im):
        return im[:, :, 1:, :] - im[:, :, :-1, :]

    def grad_x(im):
        return im[:, :, :, 1:] - im[:, :, :, :-1]

    return (F.l1_loss(grad_x(pred), grad_x(gt))
            + F.l1_loss(grad_y(pred), grad_y(gt)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=os.path.join(ROOT, "ml",
                                                  "unet_lowlight.pt"))
    args = ap.parse_args()

    torch.manual_seed(0)
    # 训练集: 在线随机场景(等效无限数据); 验证: 固定场景全图(指标稳定)
    train_ds = LowLightDataset(crop_packed=64, exposure_range=(0.0075, 0.08),
                               seed=0)

    val_scene = random_scene(np.random.default_rng(999))
    val_exp = 0.03
    vx_np, _ = make_lowlight_input(val_scene, val_exp, seed=4242)
    vx = torch.from_numpy(vx_np.transpose(2, 0, 1)[None]).float()
    vg = torch.from_numpy((val_scene.astype(np.float32) / 255.0)
                          .transpose(2, 0, 1)[None]).float()

    model = LightUNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LightUNet params: {n_params / 1e6:.2f}M  (CPU 训练)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)

    best_val = float("inf")
    t0 = time.time()
    losses = []
    for it in range(1, args.iters + 1):
        model.train()
        xin, gt = zip(*[train_ds[i] for i in range(args.batch)])
        xin = torch.stack(xin)
        gt = torch.stack(gt)
        pred = model(xin)
        loss = F.l1_loss(pred, gt) + 0.1 * edge_loss(pred, gt)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if it % 50 == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                vp = model(vx)
                vl = float(F.l1_loss(vp, vg))
            print(f"iter {it:4d}/{args.iters}  loss={loss.item():.4f}  "
                  f"val_l1={vl:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time() - t0:.0f}s")
            if vl < best_val:
                best_val = vl
                torch.save({"state_dict": model.state_dict(),
                            "iters": it, "val_l1": vl}, args.out)
    print(f"best val_l1={best_val:.4f} -> {args.out}")

    # 训练曲线(供文档引用)
    np.save(os.path.join(ROOT, "ml", "train_loss_curve.npy"),
            np.array(losses))


if __name__ == "__main__":
    main()
