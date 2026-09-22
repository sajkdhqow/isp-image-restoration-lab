# -*- coding: utf-8 -*-
"""
main.py —— 一键运行完整 ISP 演示
=================================

流程:
1. 生成/加载测试场景图
2. 模拟带物理缺陷的 Bayer RAW(暗角/色偏/黑电平/噪声/坏点)
3. 运行完整 ISP pipeline, 保存每个中间阶段图像
4. Demosaic 三种算法横向对比(PSNR)
5. NR 滤波器族横向对比(PSNR)
6. MIPI RAW10 打包/解包 round-trip 验证
7. 3A 协同与优化: AE 测光升级/AWB 鲁棒/混合 AF/3A 状态机
   详见 docs/optimization_log.md

运行:  python main.py
输出:  output/ 目录
"""

import os
import sys
import time

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from tools.simulate_raw import make_synthetic_scene, simulate_raw
from isp.pipeline import run_pipeline, DEFAULT_CCM
from isp.utils import (psnr, mipi_raw10_pack, mipi_raw10_unpack,
                       save_rgb, save_gray, imread_unicode, imwrite_unicode)
from isp.demosaic import (demosaic_bilinear, demosaic_malvar,
                          demosaic_edge_aware)
from isp.ccm import apply_ccm
from isp.gamma import gamma_correct, srgb_encode
from isp.nr import (mean_filter, gaussian_filter, bilateral_filter,
                    nlm_denoise, median_filter, mlm_filter, mhmf_filter,
                    weighted_median_filter, median_rational_filter)
from isp.interpolation import (nearest_interpolate, bilinear_interpolate,
                               bicubic_interpolate)

OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

BITS = 10
MAX_VAL = 1023.0
BLACK_LEVEL = 64.0


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main():
    # ------------------------------------------------------------------
    banner("[1] 场景生成与 RAW 模拟")
    # ------------------------------------------------------------------
    scene_path = os.path.join(DATA, "scene.png")
    if os.path.exists(scene_path):
        scene_u8 = cv2.cvtColor(imread_unicode(scene_path), cv2.COLOR_BGR2RGB)
    else:
        scene_lin = make_synthetic_scene(512, 512)
        scene_u8 = (np.clip(scene_lin, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        imwrite_unicode(scene_path, cv2.cvtColor(scene_u8, cv2.COLOR_BGR2RGB))
    print(f"场景图: {scene_path}  {scene_u8.shape}")

    # 模拟 AE 曝光: 色偏矩阵对角 > 1 会把饱和色推过满量程, 真实相机
    # 靠曝光控制预留 headroom, 否则高亮通道 clip 后 CCM 无法恢复色彩
    ANALOG_GAIN = 0.72

    raw = simulate_raw(scene_u8, bits=BITS, black_level=BLACK_LEVEL,
                       analog_gain=ANALOG_GAIN, seed=42)
    raw_path = os.path.join(DATA, "sim_raw10.raw")
    raw.astype(np.uint16).tofile(raw_path)
    save_gray(os.path.join(OUT, "00_raw_mosaic.png"), raw, BITS)
    print(f"RAW: {raw_path}  范围 [{raw.min():.0f}, {raw.max():.0f}]")

    # ------------------------------------------------------------------
    banner("[2] MIPI RAW10 打包/解包 round-trip")
    # ------------------------------------------------------------------
    raw_u16 = raw.astype(np.uint16)  # 打包前先量化成整数(与真实 sensor 数据一致)
    packed = mipi_raw10_pack(raw_u16)
    unpacked = mipi_raw10_unpack(packed, raw.shape[1], raw.shape[0])
    err = np.abs(unpacked - raw_u16).max()
    print(f"packed size: {packed.size} bytes "
          f"(= H*W*10/8 = {raw.size * 10 // 8})")
    print(f"round-trip max error: {err:.0f}")
    assert err == 0, "MIPI pack/unpack round-trip failed"
    imwrite_unicode(os.path.join(OUT, "01_mipi_packed_visual.png"),
                    packed.reshape(raw.shape[0], raw.shape[1] // 4 * 5)[:200])

    # ------------------------------------------------------------------
    banner("[3] 完整 ISP Pipeline")
    # ------------------------------------------------------------------
    # LSC 增益表按产线流程标定: 拍一张均匀白场 -> 逐通道逐 bin 计算
    # target/median 增益网格。不能用场景图自标定, 否则会把彩色物体
    # 的三个通道都拉向同一目标, 色彩先被抹灰再进 CCM, 无法恢复。
    from isp.blc import blc as blc_fn
    from isp.lsc import calibrate_lsc_grid
    white_u8 = np.full((512, 512, 3), 205, np.uint8)  # 线性域 ~0.6 白场
    white_raw = simulate_raw(white_u8, bits=BITS, black_level=BLACK_LEVEL,
                             analog_gain=ANALOG_GAIN, dead_ratio=0,
                             hot_ratio=0, sigma_read=0.5, shot=0.0, seed=7)
    white_blc = blc_fn(white_raw, BLACK_LEVEL, MAX_VAL)
    # target 取白场中心亮度(中心最亮, 不能用全图分位数, 否则增益基准偏低)
    c = white_blc.shape[0] // 2
    lsc_target = float(np.median(white_blc[c - 32:c + 32, c - 32:c + 32]))
    lsc_gains = calibrate_lsc_grid(white_blc, lsc_target, grid=(33, 33),
                                   max_gain=16.0)
    print(f"LSC 白场标定: target={lsc_target:.0f}, "
          f"R gain 范围 [{lsc_gains['R'].min():.2f}, {lsc_gains['R'].max():.2f}]")

    t0 = time.time()
    final, stages = run_pipeline(raw.copy(), bayer_format="RGGB",
                                 max_val=MAX_VAL, black_level=BLACK_LEVEL,
                                 lsc_gains=lsc_gains,
                                 capture_stages=True)
    dt = time.time() - t0
    print(f"pipeline done in {dt:.2f}s, stages: {len(stages)}")
    for name in sorted(stages):
        img = stages[name]
        if not isinstance(img, np.ndarray):  # 跳过增益元组等非图像条目
            continue
        if img.ndim == 2:
            save_gray(os.path.join(OUT, f"stage_{name}.png"), img, BITS)
        else:
            save_rgb(os.path.join(OUT, f"stage_{name}.png"), img)
    save_rgb(os.path.join(OUT, "final_output.png"), final)
    if "05_awb_gains" in stages:
        rg, bg, gr, gb = stages["05_awb_gains"]
        print(f"AWB gains: R={rg:.3f} B={bg:.3f} Gr={gr:.3f} Gb={gb:.3f}")

    # ------------------------------------------------------------------
    banner("[4] Demosaic 算法对比(对线性参考图算 PSNR)")
    # ------------------------------------------------------------------
    # 参考 = 场景线性图(生成 raw 时关闭色偏/渐晕, BLC 后即约等于真值)。
    # PSNR_clean: 纯净输入下的插值误差(反映 zipper/伪影水平)
    # PSNR_noisy: 含噪输入下的误差(反映算法对噪声的敏感度)
    gt_lin = (scene_u8.astype(np.float32) / 255.0) ** 2.2

    clean_raw = simulate_raw(scene_u8, bits=BITS, black_level=BLACK_LEVEL,
                             vignette=0.0, color_shading=0.0,
                             dead_ratio=0, hot_ratio=0,
                             sigma_read=0.01, shot=0.0,
                             apply_color_distortion=False, seed=1)
    clean_lin = blc_fn(clean_raw, BLACK_LEVEL, MAX_VAL)   # RAW 域 [0, max_val]

    noisy_raw = simulate_raw(scene_u8, bits=BITS, black_level=BLACK_LEVEL,
                             vignette=0.0, color_shading=0.0,
                             dead_ratio=0, hot_ratio=0, seed=2,
                             apply_color_distortion=False)
    noisy_lin = blc_fn(noisy_raw, BLACK_LEVEL, MAX_VAL)

    results = {}
    for name, fn in [("bilinear", demosaic_bilinear),
                     ("malvar", demosaic_malvar),
                     ("edge_aware", demosaic_edge_aware)]:
        dem_c = fn(clean_lin, "RGGB", MAX_VAL) / MAX_VAL
        dem_n = fn(noisy_lin, "RGGB", MAX_VAL) / MAX_VAL
        results[name] = (psnr(gt_lin, dem_c), psnr(gt_lin, dem_n))
        save_rgb(os.path.join(OUT, f"02_demosaic_{name}.png"), dem_c)
    print(f"{'algorithm':<12} {'PSNR_clean':>10} {'PSNR_noisy':>10}")
    for k, (v1, v2) in results.items():
        print(f"{k:<12} {v1:>10.2f} {v2:>10.2f}")

    # ------------------------------------------------------------------
    banner("[5] NR 滤波器族对比(彩色噪声图, 对干净图算 PSNR)")
    # ------------------------------------------------------------------
    rng = np.random.default_rng(3)
    clean_rgb = (np.clip(scene_u8 / 255.0, 0, 1)).astype(np.float32)
    noise_rgb = np.clip(
        clean_rgb + rng.normal(0, 12 / 255.0, clean_rgb.shape), 0, 1)
    save_rgb(os.path.join(OUT, "03_nr_noisy.png"), noise_rgb)

    gray = noise_rgb.mean(-1)
    filters = [
        ("mean", lambda im: np.stack([mean_filter(im[..., i], 5)
                                      for i in range(3)], -1)),
        ("gaussian", lambda im: np.stack([gaussian_filter(im[..., i], 5, 1.2)
                                          for i in range(3)], -1)),
        ("bilateral", lambda im: np.stack(
            [bilateral_filter(im[..., i], 7, 1.8, 18.0)
             for i in range(3)], -1)),
        ("nlm", lambda im: np.stack(
            [nlm_denoise(im[..., i], 7, 3, 1.2) for i in range(3)], -1)),
        ("median", lambda im: np.stack([median_filter(im[..., i], 3)
                                        for i in range(3)], -1)),
        ("mlm", lambda im: np.stack([mlm_filter(im[..., i], 3)
                                     for i in range(3)], -1)),
        ("mhmf", lambda im: np.stack([mhmf_filter(im[..., i], 3)
                                      for i in range(3)], -1)),
        ("wmf", lambda im: np.stack(
            [weighted_median_filter(im[..., i], 3, 3) for i in range(3)], -1)),
        ("mrf", lambda im: np.stack(
            [median_rational_filter(im[..., i], 3) for i in range(3)], -1)),
    ]
    print(f"{'filter':<10} {'PSNR(dB)':>10} {'time(s)':>8}")
    for name, fn in filters:
        t0 = time.time()
        den = np.clip(fn(noise_rgb), 0, 1)
        dt = time.time() - t0
        save_rgb(os.path.join(OUT, f"03_nr_{name}.png"), den)
        print(f"{name:<10} {psnr(clean_rgb, den):>10.2f} {dt:>8.2f}")

    # ------------------------------------------------------------------
    banner("[6] 图像插值对比")
    # ------------------------------------------------------------------
    up2_nearest = nearest_interpolate(scene_u8, scale=2.0)
    up2_bilinear = bilinear_interpolate(scene_u8, scale=2.0)
    up2_bicubic = bicubic_interpolate(scene_u8, scale=2.0)
    imwrite_unicode(os.path.join(OUT, "04_interp_nearest_x2.png"),
                    cv2.cvtColor(up2_nearest, cv2.COLOR_RGB2BGR))
    imwrite_unicode(os.path.join(OUT, "04_interp_bilinear_x2.png"),
                    cv2.cvtColor(up2_bilinear, cv2.COLOR_RGB2BGR))
    imwrite_unicode(os.path.join(OUT, "04_interp_bicubic_x2.png"),
                    cv2.cvtColor(up2_bicubic, cv2.COLOR_RGB2BGR))
    print("saved: nearest / bilinear / bicubic x2 (看边缘锯齿与振铃差异)")

    # ------------------------------------------------------------------
    banner("[7] 3A 协同与进阶优化: 测光升级/AWB鲁棒/混合AF/状态机")
    # ------------------------------------------------------------------
    # 3A 是三个闭环控制回路, 不在像素流水线里: 统计 -> 调参 -> 重采图。
    # 协同时序(面试高频): AE 先收敛(亮度可靠) -> AWB(统计不被噪声
    # 主导) -> AF 最后(对焦指标依赖高频能量, 噪声淹没梯度)。
    from functools import partial
    from isp.ae import (AeParams, AeLimits, ae_run, ae_stats,
                        ae_stats_weighted, ae_update)
    from isp.af import (defocus_sigma, blur_linear_scene,
                        focus_metric_tenengrad, af_search)
    from isp.awb import awb, awb_robust
    from isp.pdaf import pdaf_lr_views, pdaf_parallax, af_hybrid_search
    from isp.a3a_state import A3AStateMachine, A3AState

    T_REF_US = 10000.0   # 参考曝光时间, exposure_scale = t / T_REF
    AF_BEST_POS = 617    # "真实"对焦点(demo 内部秘密, 让 AF 搜出来)
    K_FOCUS = 12.0       # 离焦模糊增益: 越大峰越尖(焦深越小), 可分辨
    scene_lin = (scene_u8.astype(np.float32) / 255.0) ** 2.2

    def capture(params, scene=None):
        """虚拟相机: 曝光量 = t/T_REF * 总增益, 返回 BLC 后 RAW。

        数字增益折算进总增益(demo 简化: 未单独建模数字增益对读出
        噪声的放大, 真实系统 dg 部分 SNR 代价更大)。
        """
        r = simulate_raw(scene_u8 if scene is None else scene,
                         bits=BITS, black_level=BLACK_LEVEL,
                         analog_gain=params.total_gain,
                         exposure_scale=params.exposure_time_us / T_REF_US,
                         dead_ratio=0, hot_ratio=0,
                         sigma_read=0.8, shot=0.02, seed=11)
        return blc_fn(r, BLACK_LEVEL, MAX_VAL)

    # --- 1) AE 基础闭环: 曝光三角按噪声代价分配 ---
    def ae_log(it, luma, clip, p):
        print(f"  iter{it}: t={p.exposure_time_us:7.0f}us "
              f"ag={p.analog_gain:.2f} dg={p.digital_gain:.2f} "
              f"-> luma={luma:6.1f} clip={clip * 100:4.1f}%")

    ae_target = 450.0
    limits = AeLimits(t_max_us=4000.0, ag_max=8.0)
    print("AE 基础闭环: t=500us(欠曝26倍) -> 目标 luma=450")
    # black_level 修正后 clip 统计是真实值: 本场景收敛时约 12% 高光
    # 裁切(色块/白圆), 属高key场景的正常代价, guard 放宽到 0.25
    ae_params, _, ae_status = ae_run(
        lambda p: capture(p), ae_target, AeParams(500.0, 1.0), limits,
        max_val=MAX_VAL, log=ae_log, black_level=BLACK_LEVEL,
        clip_guard=0.25)
    print(f"AE {ae_status}: t={ae_params.exposure_time_us:.0f}us "
          f"ag={ae_params.analog_gain:.2f} dg={ae_params.digital_gain:.2f}")

    # --- 2) AE 测光升级: 逆光场景下 全图均值 vs 主体ROI加权 ---
    # 把棋盘格区域压暗充当"逆光主体": 全图均值测光时均值被亮背景
    # 主导, 主体欠曝; ROI 加权(等效人脸检测框加权)把主体拍亮。
    dark = (slice(250, 410), slice(280, 440))
    scene_backlit = scene_u8.copy()
    scene_backlit[dark] = (scene_backlit[dark] * 0.4).astype(np.uint8)
    h2, w2 = scene_u8.shape[0] // 2, scene_u8.shape[1] // 2
    w_map = np.ones((h2, w2), np.float32)
    w_map[125:205, 140:220] = 20.0           # 主体区域权重 x20(人脸框)

    def stats_roi(raw, bf, mv):
        return ae_stats_weighted(raw, weight=w_map, bayer_format=bf,
                                 max_val=mv, black_level=BLACK_LEVEL)

    # 主体优先与高光保护冲突(主体提亮必然推爆背景高光), 优先级
    # 仲裁: 点按/人脸 > 高光 -> 此处关闭 guard; guard 打开的行为
    # 见下面灯牌 demo(高光优先)
    p_plain, _, _ = ae_run(lambda p: capture(p, scene_backlit), ae_target,
                           AeParams(500.0, 1.0), limits, max_val=MAX_VAL,
                           black_level=BLACK_LEVEL, clip_guard=None)
    p_roi, _, _ = ae_run(lambda p: capture(p, scene_backlit), ae_target,
                         AeParams(500.0, 1.0), limits, max_val=MAX_VAL,
                         stats_fn=stats_roi, clip_guard=None)

    def subject_luma(p):
        raw = capture(p, scene_backlit)
        return float(raw[dark].mean()), \
            float((raw > 0.97 * (MAX_VAL - BLACK_LEVEL)).mean() * 100)

    sub_p, clip_p = subject_luma(p_plain)
    sub_r, clip_r = subject_luma(p_roi)
    print(f"AE 测光升级(逆光): 全图均值 vs 主体ROI加权")
    print(f"  普通 AE:   t={p_plain.exposure_time_us:.0f}us "
          f"ag={p_plain.analog_gain:.2f} | 主体亮度={sub_p:6.1f} DN, "
          f"clip={clip_p:.1f}%")
    print(f"  ROI加权 AE: t={p_roi.exposure_time_us:.0f}us "
          f"ag={p_roi.analog_gain:.2f} | 主体亮度={sub_r:6.1f} DN "
          f"(提亮 x{sub_r / max(sub_p, 1e-6):.1f}), clip={clip_r:.1f}%")

    # --- 2c) AE 高光保护: 夜景灯牌场景 guard 开 vs 关(高光优先) ---
    # 修复前 clip 阈值落在不可达量程上(0.97*1023=992 > BLC后上限959),
    # guard 是死代码; 修复后演示"高光优先"仲裁: 追亮度会把灯牌推爆,
    # guard 冻结曝光保细节, 代价是整体偏暗 —— 与 2) 的主体优先互补
    print("AE 高光保护(夜景灯牌): guard 开 vs 关")
    scene_lamp = (scene_u8.astype(np.float32) * 0.35).astype(np.uint8)
    lamp_val = np.linspace(1.0, 0.25, 140, dtype=np.float32)  # 渐变灯牌
    lamp_u8 = (lamp_val ** (1 / 2.2) * 255).astype(np.uint8)
    scene_lamp[60:200, 60:200] = lamp_u8[:, None]

    for guard, tag in ((0.05, "开(高光优先)"), (None, "关(追亮度)")):
        p_l, _, st_l = ae_run(lambda p: capture(p, scene_lamp), ae_target,
                              AeParams(500.0, 1.0), limits, max_val=MAX_VAL,
                              black_level=BLACK_LEVEL, clip_guard=guard)
        raw_l = capture(p_l, scene_lamp)
        st_lamp = ae_stats(raw_l, "RGGB", MAX_VAL, black_level=BLACK_LEVEL)
        lamp_texture = float(raw_l[60:200, 60:200].std())
        print(f"  guard{tag}: status={st_l} luma={st_lamp['luma']:6.1f} "
              f"clip={st_lamp['clip_ratio'] * 100:4.1f}% "
              f"灯牌纹理std={lamp_texture:5.1f} "
              f"(t={p_l.exposure_time_us:.0f} ag={p_l.analog_gain:.2f} "
              f"dg={p_l.digital_gain:.2f})")

    # --- 3) AWB 联动: 信号接近读噪底时, 通道统计被噪声主导 ---
    under_raw = blc_fn(simulate_raw(
        scene_u8, bits=BITS, black_level=BLACK_LEVEL, analog_gain=1.0,
        exposure_scale=0.002, dead_ratio=0, hot_ratio=0,
        sigma_read=0.8, shot=0.02, seed=11), BLACK_LEVEL, MAX_VAL)
    conv_raw = capture(ae_params)
    g_under = awb(under_raw, "dynamic_threshold", "RGGB", MAX_VAL)
    g_conv = awb(conv_raw, "dynamic_threshold", "RGGB", MAX_VAL)
    drift = abs(g_under[0] - g_conv[0]) / g_conv[0] * 100
    print(f"AWB@近噪声底: R gain={g_under[0]:.3f} "
          f"(vs 收敛后 {g_conv[0]:.3f}, 偏差 {drift:.1f}%) <- 先AE后AWB")

    # --- 4) AWB 鲁棒化: 大面积单色场景 多假设 + 黑体轨迹投影 ---
    scene_grass = scene_u8.copy()
    scene_grass[..., 0] = (scene_grass[..., 0] * 0.35).astype(np.uint8)
    scene_grass[..., 2] = (scene_grass[..., 2] * 0.55).astype(np.uint8)
    for tag, scn in (("草地场景", scene_grass), ("正常场景", scene_u8)):
        r_raw = blc_fn(simulate_raw(
            scn, bits=BITS, black_level=BLACK_LEVEL,
            analog_gain=ae_params.analog_gain,
            exposure_scale=ae_params.exposure_time_us / T_REF_US,
            dead_ratio=0, hot_ratio=0, sigma_read=0.8, shot=0.02,
            seed=11), BLACK_LEVEL, MAX_VAL)
        res = awb_robust(r_raw, "RGGB", MAX_VAL)
        c = "  ".join(f"{k[:2]}={v[0]:.2f}/{v[1]:.2f}"
                      for k, v in res["candidates"].items())
        print(f"AWB 鲁棒[{tag}]: 一致性={res['confidence']} "
              f"(spread={res['spread']:.2f}) -> "
              f"R={res['rgain']:.3f} B={res['bgain']:.3f}"
              f"{' [轨迹投影]' if res['projected'] else ''}")
        print(f"  候选: {c}")

    # --- 5) AF 对比: 纯对比度(三段式+多帧) vs PDAF 混合 ---
    def contrast_eval(pos, frame_idx=0):
        """移镜 -> 离焦模糊(线性域) -> 采图 -> demosaic -> Tenengrad。"""
        sigma = defocus_sigma(pos, AF_BEST_POS, k=K_FOCUS)
        blurred = blur_linear_scene(scene_lin, sigma)
        blur_u8 = (np.clip(blurred, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        r = simulate_raw(blur_u8, bits=BITS, black_level=BLACK_LEVEL,
                         analog_gain=ae_params.analog_gain,
                         exposure_scale=ae_params.exposure_time_us / T_REF_US,
                         dead_ratio=0, hot_ratio=0,
                         sigma_read=0.8, shot=0.02, seed=13 + frame_idx)
        rgb = demosaic_bilinear(blc_fn(r, BLACK_LEVEL, MAX_VAL),
                                "RGGB", MAX_VAL)
        luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return focus_metric_tenengrad(luma, MAX_VAL)

    best_c, rec_c = af_search(contrast_eval, n_frames=3)
    print(f"AF 纯对比度: 位置={best_c} 误差={abs(best_c - AF_BEST_POS)} "
          f"位置数={len(rec_c)} 采图={len(rec_c) * 3}")

    def pdaf_eval(pos):
        L, R, _ = pdaf_lr_views(scene_lin, pos, AF_BEST_POS, k=K_FOCUS)
        return pdaf_parallax(L, R)

    best_h, _, h_info = af_hybrid_search(pdaf_eval, contrast_eval, pos0=64)
    print(f"AF 混合(PDAF): PDAF粗定位={h_info['pdaf_est']} -> "
          f"精修={best_h} 误差={abs(best_h - AF_BEST_POS)} "
          f"采图={h_info['n_pdaf'] + h_info['n_contrast']} "
          f"(PDAF {h_info['n_pdaf']} + 对比度 {h_info['n_contrast']})")

    csv_path = os.path.join(OUT, "07_af_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("lens_pos,metric\n")
        for p, m in rec_c:
            f.write(f"{p},{m:.6f}\n")

    # --- 6) 3A 状态机: 收敛 -> 场景突变重收敛 -> 锁定/解锁 ---
    print("3A 状态机: 常驻决策层(控制律与状态解耦), 预览流连续运行")
    sm = A3AStateMachine(target=ae_target, tol=0.03, stable_frames=2,
                         scene_change=0.30)
    params = AeParams(ae_params.exposure_time_us, ae_params.analog_gain)
    scene_now = scene_u8
    DIM = 0.5   # gamma 域压暗一半 ≈ 线性域 x0.22, 曝光余量内可恢复
    for fr in range(18):
        if fr == 7:
            scene_now = (scene_u8.astype(np.float32) * DIM).astype(np.uint8)
            print(f"  -- 帧{fr}: 场景变暗 x{DIM}(进隧道)")
        st = ae_stats(capture(params, scene_now), "RGGB", MAX_VAL,
                      black_level=BLACK_LEVEL)
        prev = sm.state
        state = sm.update(st["luma"], locked=(12 <= fr < 14))
        if state != prev:
            print(f"  帧{fr:2d}: {prev:<12s} -> {state:<12s} "
                  f"luma={st['luma']:6.1f}")
        if state in (A3AState.CONVERGING, A3AState.RECONVERGING):
            params = ae_update(params, st["luma"], ae_target, limits,
                               clip_ratio=st["clip_ratio"])
    print(f"  结束: t={params.exposure_time_us:.0f}us "
          f"ag={params.analog_gain:.2f} dg={params.digital_gain:.2f} "
          f"(场景变暗后 t/模拟增益到顶, 数字增益补足)")

    # --- 7) 预览: 逆光场景 普通 AE vs ROI 加权 AE ---
    for tag, p in (("plain", p_plain), ("roiae", p_roi)):
        rgb_lin = demosaic_malvar(capture(p, scene_backlit), "RGGB",
                                  MAX_VAL) / MAX_VAL
        save_rgb(os.path.join(OUT, f"07_3a_backlit_{tag}.png"),
                 srgb_encode(np.clip(rgb_lin, 0, 1), max_val=1.0))
    print("saved: 07_3a_backlit_plain.png / 07_3a_backlit_roiae.png "
          "(看主体提亮)")

    print("\nAll done. 输出目录:", OUT)


if __name__ == "__main__":
    main()
