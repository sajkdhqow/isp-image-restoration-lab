# -*- coding: utf-8 -*-
"""
三路径对比:传统 ISP vs 回归 U-Net vs 生成式 Rectified Flow
==========================================================

在 held-out 场景(训练从未见过)上对比三条路线, 输出:
- PSNR(保真度): 回归模型预期最高(输出条件均值)
- Tenengrad 锐度(感知代理): 生成式模型预期更锐(采样后验样本)
- 单帧延迟(CPU): 预期 flow 随步数线性增长 —— 这是生成式方法
  进不了产品 preview 链路的核心原因, 也是本 demo 的"实际应用"结论

运行:  python ml/eval_flow.py   (需先跑 train.py 与 train_flow.py)
输出:  output/flow_compare_expXXX.png + 指标打印
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import lpips as lpips_lib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ml.model import LightUNet
from ml.flow_model import FlowUNet, make_cond, euler_sample
from ml.data import make_lowlight_input, BLACK_LEVEL, MAX_VAL
from ml.eval_demo import build_test_scene
from tools.simulate_raw import simulate_raw
from isp.utils import save_rgb, psnr
from isp.pipeline import run_pipeline
from isp.gamma import srgb_encode
from isp.af import focus_metric_tenengrad

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)
SEED = 555
GAIN_REF = 0.35

# LPIPS(AlexNet): 感知距离, 越低越好 —— Tenengrad 锐度会被噪声抬高,
# 感知质量必须用学习型指标才有说服力
_lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False)
_lpips_fn.eval()


def lpips_dist(a, b):
    """a, b: (H,W,3) float [0,1] -> LPIPS 距离(越低越接近 b/参考)。"""
    to_t = lambda x: torch.from_numpy(
        x.transpose(2, 0, 1)[None]).float() * 2 - 1
    with torch.no_grad():
        return float(_lpips_fn(to_t(a), to_t(b)))


def sharpness(img01):
    """Tenengrad 锐度(感知代理): 与 GT 锐度的比值更有意义。"""
    return focus_metric_tenengrad(img01.astype(np.float32) * 1023.0, 1023.0)


def main():
    # ---- 加载模型: 回归 U-Net / 基础 Flow / reflow 蒸馏(若存在) ----
    unet = LightUNet()
    unet.load_state_dict(torch.load(
        os.path.join(ROOT, "ml", "unet_lowlight.pt"),
        map_location="cpu")["state_dict"])
    unet.eval()

    flow = FlowUNet()
    flow.load_state_dict(torch.load(
        os.path.join(ROOT, "ml", "flow_lowlight.pt"),
        map_location="cpu")["state_dict"])
    flow.eval()

    reflow_path = os.path.join(ROOT, "ml", "flow_reflow.pt")
    flow_r = None
    if os.path.exists(reflow_path):
        flow_r = FlowUNet()
        flow_r.load_state_dict(torch.load(reflow_path,
                                          map_location="cpu")["state_dict"])
        flow_r.eval()
        print("已加载 reflow 蒸馏模型: ml/flow_reflow.pt")

    scene = build_test_scene()
    save_rgb(os.path.join(OUT, "flow_test_scene_gt.png"), scene)
    gt = scene.astype(np.float32) / 255.0
    s_gt = sharpness(gt)

    # 传统路径的 LSC 白场标定(全套装备, 公平对比)
    white_raw = simulate_raw(np.full((512, 512, 3), 205, np.uint8),
                             bits=10, black_level=BLACK_LEVEL,
                             analog_gain=0.72, dead_ratio=0, hot_ratio=0,
                             sigma_read=0.5, shot=0.0, seed=7)
    c = 256
    lsc_target = float(np.median(white_raw[c - 32:c + 32, c - 32:c + 32]))
    from isp.lsc import calibrate_lsc_grid
    lsc_gains = calibrate_lsc_grid(white_raw, lsc_target, grid=(33, 33),
                                   max_gain=16.0)

    for exp, tag in ((0.02, "exp002"), (0.0075, "exp00075")):
        raw_low = simulate_raw(scene, bits=10, black_level=BLACK_LEVEL,
                               analog_gain=1.0, exposure_scale=exp,
                               dead_ratio=0, hot_ratio=0, sigma_read=3.0,
                               shot=0.08, seed=SEED)
        xin, g = make_lowlight_input(scene, exp, seed=SEED, gain_ref=GAIN_REF)
        xin_t = torch.from_numpy(xin.transpose(2, 0, 1)[None]).float()

        # 回归 U-Net(顺带测延迟)
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_unet = unet(xin_t)[0].numpy().transpose(1, 2, 0)
        dt_unet = time.perf_counter() - t0
        pred_unet = np.clip(pred_unet, 0, 1)

        # 传统 ISP(数字增益放大, 顺带测延迟)
        boosted = np.clip(raw_low - BLACK_LEVEL, 0, None) / exp + BLACK_LEVEL
        boosted = np.clip(boosted, 0, MAX_VAL)
        t0 = time.perf_counter()
        trad, _ = run_pipeline(boosted.copy(), bayer_format="RGGB",
                               max_val=MAX_VAL, black_level=BLACK_LEVEL,
                               lsc_gains=lsc_gains, capture_stages=True)
        dt_trad = time.perf_counter() - t0
        trad = np.clip(trad, 0, 1)

        # 生成式 Flow: 步数扫描(同一次条件, 不同 ODE 步数)
        cond = make_cond(xin_t, (512, 512))
        flow_results = {}
        for steps in (1, 2, 4, 8):
            t0 = time.perf_counter()
            with torch.no_grad():
                pf = euler_sample(flow, cond, n_steps=steps,
                                  seed=20250916)[0].numpy().transpose(1, 2, 0)
            dt = time.perf_counter() - t0
            flow_results[steps] = (np.clip(pf, 0, 1), dt)

        # ---- 打表 ----
        print(f"\nexposure={exp:.4f} (1/{1 / exp:.0f} 正常照度)  "
              f"GT锐度={s_gt:.1f}(锐度比会被噪声抬高, 噪声本身是高频)")
        print(f"{'方法':<16}{'PSNR':>9}{'LPIPS↓':>9}{'锐度/GT':>9}"
              f"{'延迟s':>8}")
        print(f"{'传统ISP(增益放大)':<14}{psnr(gt, trad):>9.2f}"
              f"{lpips_dist(trad, gt):>9.3f}"
              f"{sharpness(trad) / s_gt:>9.2f}{dt_trad:>8.2f}")
        print(f"{'回归U-Net':<14}{psnr(gt, pred_unet):>9.2f}"
              f"{lpips_dist(pred_unet, gt):>9.3f}"
              f"{sharpness(pred_unet) / s_gt:>9.2f}{dt_unet:>8.2f}")
        for steps, (pf, dt) in flow_results.items():
            print(f"{f'Flow({steps}步ODE)':<14}{psnr(gt, pf):>9.2f}"
                  f"{lpips_dist(pf, gt):>9.3f}"
                  f"{sharpness(pf) / s_gt:>9.2f}{dt:>8.2f}")
        if flow_r is not None:
            t0 = time.perf_counter()
            with torch.no_grad():
                pr = euler_sample(flow_r, cond, n_steps=1,
                                  seed=20250916)[0].numpy().transpose(1, 2, 0)
            dt_r = time.perf_counter() - t0
            pr = np.clip(pr, 0, 1)
            print(f"{'Flow1步(reflow)':<14}{psnr(gt, pr):>9.2f}"
                  f"{lpips_dist(pr, gt):>9.3f}"
                  f"{sharpness(pr) / s_gt:>9.2f}{dt_r:>8.2f}")

        # ---- 对比图: GT / 传统 / U-Net / Flow(4步) ----
        pf4 = flow_results[4][0]
        panel = np.concatenate([gt, trad, pred_unet, pf4], axis=1)
        save_rgb(os.path.join(OUT, f"flow_compare_{tag}.png"), panel)

    print("\nsaved: output/flow_compare_exp002.png / flow_compare_exp00075.png")
    print("(左起: GT / 传统ISP / 回归U-Net / Flow 4步)")


if __name__ == "__main__":
    main()
