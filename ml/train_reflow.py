# -*- coding: utf-8 -*-
"""
Reflow 蒸馏 —— 把多步 ODE 质量压进 1 步
========================================

【原理(2-Rectified Flow, SDXL-Turbo 的工业思路)】
Rectified Flow 训好后, 用它自己的多步 ODE 输出生成"噪声->图像"的
直线路径对 (z, x0_hat), 再以这些对重训速度场。重训后路径被"拉直",
单步 Euler 即可直达数据分布 —— 1 步采样质量 ≈ 多步教师, 延迟降 8 倍。

【流程】
1. 生成缓存: N 个随机场景, 教师模型 32 步 ODE 采样得 (z, x0_hat, cond)
2. 重训: 从教师权重初始化, 在缓存对上做流匹配训练(少量 iter 即可,
   因为只是把已学好的场"拉直", 不是从头学)
3. 评估: 1 步 PSNR/LPIPS 对比重训前后

运行:  python ml/train_reflow.py          # 缓存 ~2min + 重训 ~4min
输出:  ml/flow_reflow.pt
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

CACHE_SCENE_SEED0 = 4321   # 与训练/验证场景种子不同(held-out 无关紧要,
                           # reflow 只在教师自己的输出上训练)


def build_cache(teacher, n_scenes=48, k_noise=2, teacher_steps=32,
                size=256):
    """教师 32 步 ODE 生成 (z -> x0_hat) 直线对缓存(内存 float16)。"""
    rng = np.random.default_rng(CACHE_SCENE_SEED0)
    conds, zs, x0s = [], [], []
    t0 = time.time()
    for i in range(n_scenes):
        scene = random_scene(rng)
        exp = float(rng.uniform(0.0075, 0.08))
        xin, _ = make_lowlight_input(scene, exp,
                                     seed=int(rng.integers(0, 2 ** 31)))
        xin_t = torch.from_numpy(xin.transpose(2, 0, 1)[None]).float()
        cond = make_cond(xin_t, (size, size))
        for k in range(k_noise):
            seed = 700000 + i * 10 + k
            g = torch.Generator().manual_seed(seed)
            z = torch.randn(1, 3, size, size, generator=g)
            x0 = euler_sample(teacher, cond, n_steps=teacher_steps,
                              x1=z.clone())
            conds.append(cond[0].half())
            zs.append(z[0].half())
            x0s.append(x0[0].half())
        if (i + 1) % 12 == 0:
            print(f"  cache {i + 1}/{n_scenes} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return conds, zs, x0s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--teacher-steps", type=int, default=32)
    ap.add_argument("--base", default=os.path.join(ROOT, "ml",
                                                   "flow_lowlight.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "ml",
                                                  "flow_reflow.pt"))
    args = ap.parse_args()

    teacher = FlowUNet()
    teacher.load_state_dict(torch.load(args.base,
                                       map_location="cpu")["state_dict"])
    teacher.eval()

    print("[1/2] 生成 reflow 缓存(教师 32 步 ODE)")
    conds, zs, x0s = build_cache(teacher, teacher_steps=args.teacher_steps)
    n = len(conds)
    print(f"  缓存对数: {n}")

    print("[2/2] 重训(拉直路径)")
    model = FlowUNet()
    model.load_state_dict(torch.load(args.base,
                                     map_location="cpu")["state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    rng = np.random.default_rng(7)
    crop = 128   # 全分辨率 crop(条件/噪声/图像同位裁剪)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        model.train()
        idx = rng.integers(0, n, args.batch)
        cy = rng.integers(0, 256 - crop + 1)
        cx = rng.integers(0, 256 - crop + 1)
        sl = (slice(cy, cy + crop), slice(cx, cx + crop))
        cond = torch.stack([conds[i][:, sl[0], sl[1]] for i in idx]).float()
        x0 = torch.stack([x0s[i][:, sl[0], sl[1]] for i in idx]).float()
        z = torch.stack([zs[i][:, sl[0], sl[1]] for i in idx]).float()

        b = x0.shape[0]
        t = torch.rand(b)
        x_t = (1 - t.view(b, 1, 1, 1)) * x0 + t.view(b, 1, 1, 1) * z
        v = model(x_t, t, cond)
        loss = F.mse_loss(v, z - x0)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if it % 100 == 0 or it == args.iters:
            print(f"  iter {it:4d}/{args.iters}  loss={loss.item():.5f}  "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    torch.save({"state_dict": model.state_dict(), "iters": args.iters},
               args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
