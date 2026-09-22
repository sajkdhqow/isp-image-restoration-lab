# -*- coding: utf-8 -*-
"""
ML 双链路相机 demo —— 预览(回归) / 拍照(生成式精修)
====================================================

真实产品的"预览/拍照"双链路在本仓库的对应实现:

    preview(30fps 硬约束):  回归 U-Net, 单帧 0.05s, PSNR 最高但偏糊
    capture(单帧可等待):    Rectified Flow 少步 ODE, 锐度贴 GT,
                             reflow 蒸馏后 1 步 0.16s

同一帧低照度 RAW 走两条链路, 输出对比图 + 延迟表。
若存在 ml/flow_reflow.pt(蒸馏模型)则拍照链路用 1 步蒸馏版,
否则回退基础 Flow 4 步。

运行:  python ml/run_ml_camera.py
输出:  output/ml_camera_dualpath.png
"""

import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ml.model import LightUNet
from ml.flow_model import FlowUNet, make_cond, euler_sample
from ml.data import make_lowlight_input
from ml.eval_demo import build_test_scene
from isp.utils import save_rgb, psnr

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)
EXP = 0.01   # 夜晚手持: 1/100 正常照度


def load(path):
    ckpt = torch.load(path, map_location="cpu")["state_dict"]
    return ckpt


def main():
    scene = build_test_scene()
    gt = scene.astype(np.float32) / 255.0

    xin, g = make_lowlight_input(scene, EXP, seed=555, gain_ref=0.35)
    xin_t = torch.from_numpy(xin.transpose(2, 0, 1)[None]).float()
    cond = make_cond(xin_t, gt.shape[:2])

    # ---- 预览链路: 回归 U-Net ----
    unet = LightUNet()
    unet.load_state_dict(load(os.path.join(ROOT, "ml", "unet_lowlight.pt")))
    unet.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        preview = unet(xin_t)[0].numpy().transpose(1, 2, 0)
    dt_preview = time.perf_counter() - t0
    preview = np.clip(preview, 0, 1)

    # ---- 拍照链路: 生成式精修 ----
    reflow_path = os.path.join(ROOT, "ml", "flow_reflow.pt")
    base_path = os.path.join(ROOT, "ml", "flow_lowlight.pt")
    if os.path.exists(reflow_path):
        flow = FlowUNet()
        flow.load_state_dict(load(reflow_path))
        flow.eval()
        steps, tag = 1, "Flow 1步(reflow蒸馏)"
    else:
        flow = FlowUNet()
        flow.load_state_dict(load(base_path))
        flow.eval()
        steps, tag = 4, "Flow 4步(基础)"
    t0 = time.perf_counter()
    with torch.no_grad():
        capture = euler_sample(flow, cond, n_steps=steps,
                               seed=20250916)[0].numpy().transpose(1, 2, 0)
    dt_capture = time.perf_counter() - t0
    capture = np.clip(capture, 0, 1)

    print(f"低照度 RAW exposure={EXP} (1/{1 / EXP:.0f} 正常照度)")
    print(f"  预览链路 回归U-Net:      延迟={dt_preview:.3f}s  "
          f"PSNR={psnr(gt, preview):.2f} dB  (30fps 可用: "
          f"{'是' if dt_preview < 1 / 30 else '否'})")
    print(f"  拍照链路 {tag:<18} 延迟={dt_capture:.3f}s  "
          f"PSNR={psnr(gt, capture):.2f} dB  (单帧可等待)")
    print(f"  GT 参考锐度 vs 预览 vs 拍照: 见 output 对比图")

    panel = np.concatenate([gt, preview, capture], axis=1)
    save_rgb(os.path.join(OUT, "ml_camera_dualpath.png"), panel)
    print("saved: output/ml_camera_dualpath.png (左起: GT / 预览 / 拍照)")


if __name__ == "__main__":
    main()
