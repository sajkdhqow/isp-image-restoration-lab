# -*- coding: utf-8 -*-
"""
训练条件 Rectified Flow(流匹配)低照度恢复模型
================================================

与 train.py(回归 U-Net)完全同源的数据管线与评估口径 —— 唯一变量
是"确定性回归 vs 生成式流匹配", 保证对比公平。

运行:  python ml/train_flow.py             # 默认 5000 iter, CPU ~40min
输出:  ml/flow_lowlight.pt + train_flow_curve.npy
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

from ml.flow_model import FlowUNet, make_cond, euler_sample
from ml.data import LowLightDataset, random_scene, make_lowlight_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=os.path.join(ROOT, "ml",
                                                  "flow_lowlight.pt"))
    args = ap.parse_args()

    torch.manual_seed(0)
    train_ds = LowLightDataset(crop_packed=64, exposure_range=(0.0075, 0.08),
                               seed=0)

    # 固定验证场景(与回归模型验证同源: 随机场景 seed 999, 曝光 0.03)
    val_scene = random_scene(np.random.default_rng(999))
    vx_np, _ = make_lowlight_input(val_scene, 0.03, seed=4242)
    vx = torch.from_numpy(vx_np.transpose(2, 0, 1)[None]).float()
    vg = torch.from_numpy((val_scene.astype(np.float32) / 255.0)
                          .transpose(2, 0, 1)[None]).float()
    vcond = make_cond(vx, vg.shape[-2:])

    model = FlowUNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlowUNet params: {n_params / 1e6:.2f}M  (CPU 训练)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)

    best_val = float("inf")
    t0 = time.time()
    losses = []
    for it in range(1, args.iters + 1):
        model.train()
        xin, gt = zip(*[train_ds[i] for i in range(args.batch)])
        xin = torch.stack(xin)          # (B,5,64,64) packed
        gt = torch.stack(gt)            # (B,3,128,128) sRGB
        cond = make_cond(xin, gt.shape[-2:])

        b = gt.shape[0]
        t = torch.rand(b)
        eps = torch.randn_like(gt)
        x_t = (1 - t.view(b, 1, 1, 1)) * gt + t.view(b, 1, 1, 1) * eps
        v = model(x_t, t, cond)
        loss = F.mse_loss(v, eps - gt)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if it % 50 == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                vp = euler_sample(model, vcond, n_steps=4, seed=123)
                vl = float(F.l1_loss(vp, vg))
            print(f"iter {it:4d}/{args.iters}  loss={loss.item():.4f}  "
                  f"val_l1(4step)={vl:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
            if vl < best_val:
                best_val = vl
                torch.save({"state_dict": model.state_dict(),
                            "iters": it, "val_l1": vl}, args.out)
    print(f"best val_l1(4step)={best_val:.4f} -> {args.out}")
    np.save(os.path.join(ROOT, "ml", "train_flow_curve.npy"),
            np.array(losses))


if __name__ == "__main__":
    main()
