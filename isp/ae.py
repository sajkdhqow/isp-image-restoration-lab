# -*- coding: utf-8 -*-
"""
AE —— 自动曝光 (Auto Exposure)
================================

3A 之一。与 DPC/LSC 等"逐像素处理"模块不同, AE 是闭环控制系统,
不直接修改像素, 而是调整 sensor 侧采图参数, 让画面亮度落在目标区间:

    统计(BLC/LSC 后亮度) -> AE 算法 -> 曝光参数 -> sensor 重新采图 -> 再统计

真实 ISP 里 AE 跑在 CPU/固件上, 硬件只提供 3A 统计引擎(直方图/均值)。

【曝光三角与噪声代价】(面试必考"为什么按这个顺序分配")
1. 曝光时间: 线性提升光通量, 无噪声代价; 受手抖/运动模糊限制
   (安全快门 ~1/焦距), 且受防闪烁量化约束;
2. 模拟增益: 信号和读出噪声一起放大, SNR 损失远小于数字增益;
3. 数字增益: 纯放大(信号+噪声一起), 不产生新信息, 最后才用且上限小。
分配策略: 优先拉时间 -> 时间到顶拉模拟增益 -> 数字增益兜底。

【统计位置】
AE 统计必须在 BLC/LSC 之后: 黑电平抬底会虚增亮度, 暗角会拉低均值,
都会让收敛点系统性偏移。同时统计过曝像素占比(clip ratio)做高光保护,
防止"均值达标但高光全炸"。

【收敛策略 / 面试要点】
1. 乘性更新 ratio = target/luma, 单步限幅(如 4x)防大步长来回穿越
   目标造成亮度震荡(实际产品还要限帧率, 收敛时间 3~5 帧为宜);
2. deadband(如 2%)内锁定, 避免参数在目标附近来回抖动;
3. 高光保护: clip 比例超限强制压曝光, 即使均值还没到目标 —— 夜景
   灯牌/太阳直射场景, 均值亮度不高但高光已炸;
4. 防闪烁(flicker): 50Hz 市电灯光以 100Hz 闪烁, 曝光时间不落在
   半周期整数倍上会拍出滚动 banding, 所以曝光时间要量化到 10ms
   倍数(60Hz -> 8.33ms);
5. 收敛判据失败的三种出口: 进 deadband(converged)、参数已到极限
   仍不达标(saturated, 曝光能力到顶)、迭代次数用完(max_iters)。

【3A 协同时序】
- 先 AE 后 AWB: 欠曝时读出噪声占比大, 通道均值被噪声主导, AWB
  算出的增益不可靠;
- AWB 增益会改变亮度(乘了 R/B 增益), 严格方案 AE 用固定参考白
  平衡统计亮度, 或 AWB 收敛后把增益影响反馈给 AE(两回路耦合);
- 最后 AF: 对焦指标依赖高频能量, 噪声会淹没梯度, 必须在 AE 收敛
  后做; 对焦搜索期间显示端冻结或直接回跳峰值位置(防"呼吸"感)。
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AeParams:
    """一组曝光参数: 总曝光量 = t * analog_gain * digital_gain。"""
    exposure_time_us: float = 10000.0
    analog_gain: float = 1.0
    digital_gain: float = 1.0

    @property
    def total_gain(self):
        return self.analog_gain * self.digital_gain

    @property
    def total_exposure(self):
        return self.exposure_time_us * self.total_gain


@dataclass
class AeLimits:
    """硬件能力上限(demo 里 t_max 故意设小, 展示'时间到顶、增益接力')。"""
    t_min_us: float = 50.0
    t_max_us: float = 4000.0
    ag_min: float = 1.0
    ag_max: float = 8.0
    dg_max: float = 2.0
    flicker_hz: float = 0.0   # 50/60 -> 防闪烁量化; 0 = 关闭


def ae_stats(blc_raw, bayer_format="RGGB", max_val=1023.0, black_level=0.0):
    """RAW 域亮度统计(输入须为 BLC 后数据): 加权伪亮度 + 高光比例。

    Bayer 域无法逐像素算 luma(每个像素只有一个通道), 工程上用各通道
    均值按亮度权重合成: luma = 0.299*R + 0.587*G + 0.114*B。

    black_level: BLC 使用(而非剔除)的黑电平 —— clip 阈值必须落在
    BLC 后的实际量程 [0, max_val-black_level] 内, 否则统计恒为 0
    (高光保护变死代码, 本仓库踩过此坑)。
    """
    from .utils import split_bayer
    usable = max_val - black_level
    R, Gr, Gb, B = split_bayer(np.asarray(blc_raw, np.float32), bayer_format)
    r, g, b = R.mean(), (Gr.mean() + Gb.mean()) / 2, B.mean()
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    clip_ratio = float((blc_raw > 0.97 * usable).mean())
    under_ratio = float((blc_raw < 0.05 * usable).mean())
    return {"luma": float(luma), "clip_ratio": clip_ratio,
            "under_ratio": under_ratio,
            "channel_means": (float(r), float(g), float(b))}


def ae_stats_weighted(blc_raw, weight=None, bayer_format="RGGB",
                      max_val=1023.0, black_level=0.0):
    """ROI 加权测光统计(人脸/点按区域优先) + 暗/亮区诊断。

    【为什么需要】全图均值被高光背景主导: 逆光人像里天空/灯光占
    大面积, 均值达标时主体(人脸)只有几个 DN, 拍出来全黑。
    把主体区域(人脸检测框/用户点按位置)的测光权重放大, AE 会
    增加曝光把主体拍亮, 背景过曝由 clip_guard 限步兜底 —— 这就是
    手机"点按哪里哪里亮"和"逆光模式"的统计学实现。

    Parameters
    ----------
    weight : (H, W) or (H/2, W/2) array or None
        测光权重图。(H, W) 输入自动下采样到半分辨率(与 Bayer 四
        通道同尺寸); None 等价全 1(退化为 ae_stats)。
    black_level : float
        BLC 使用的黑电平, clip 阈值按实际量程
        [0, max_val-black_level] 计算(见 ae_stats 说明)。

    Returns
    -------
    dict: luma(加权测光亮度), luma_plain(全图均值, 供对比),
          clip_ratio(>97% 可用量程占比), under_ratio(<5% 占比)
    """
    from .utils import split_bayer
    usable = max_val - black_level
    raw = np.asarray(blc_raw, np.float32)
    R, Gr, Gb, B = split_bayer(raw, bayer_format)
    if weight is None:
        r, g, b = R.mean(), (Gr.mean() + Gb.mean()) / 2, B.mean()
    else:
        w = np.asarray(weight, np.float32)
        if w.shape == raw.shape[:2]:  # 全分辨率 -> 半分辨率
            w = 0.25 * (w[0::2, 0::2] + w[0::2, 1::2] +
                        w[1::2, 0::2] + w[1::2, 1::2])
        # 权重只改分布: (ch*w).sum()/w.sum() 对整体缩放不敏感
        wm = lambda ch: float((ch * w).sum() / max(w.sum(), 1e-6))
        r, g, b = wm(R), (wm(Gr) + wm(Gb)) / 2, wm(B)
    luma_plain = 0.299 * R.mean() + 0.587 * (Gr.mean() + Gb.mean()) / 2 \
        + 0.114 * B.mean()
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return {"luma": float(luma),
            "luma_plain": float(luma_plain),
            "clip_ratio": float((raw > 0.97 * usable).mean()),
            "under_ratio": float((raw < 0.05 * usable).mean()),
            "channel_means": (float(r), float(g), float(b))}


def quantize_time(t_us, flicker_hz):
    """防闪烁: 曝光时间量化到灯光半周期(100/120Hz)的整数倍。"""
    if not flicker_hz:
        return t_us
    period = 1e6 / (2.0 * flicker_hz)  # 半周期, us
    return max(period, round(t_us / period) * period)


def ae_update(params, luma, target, limits, max_step=4.0,
              clip_ratio=None, clip_guard=0.03):
    """一步 AE 更新: 乘性 ratio -> 限步 -> 按噪声代价分配曝光三角。

    高光保护语义(防振荡版): clip 占比超限时 ——
    - 均值还低于目标: 冻结曝光(ratio 上限 1.0), 不许为追亮度把
      高光进一步推爆(夜景灯牌: 灯牌细节优先于整体亮度);
    - 均值已达标或超标: 加速压(ratio 上限 0.8)。
    冻结时参数不变 -> ae_run 判 saturated 退出, 即"保护约束到顶"。
    旧版无条件 min(ratio, 0.8) 会在欠曝+裁切场景把曝光一路压到底
    (ratio 永远被压成 0.8 < 1), 是发散语义, 已修正。
    """
    ratio = target / max(luma, 1e-6)

    # 高光保护: 过曝像素占比超限就限制步长, 即使均值未达标
    # (clip_guard=None 表示关闭保护, 用于主体优先仲裁)
    if (clip_guard is not None and clip_ratio is not None
            and clip_ratio > clip_guard):
        ratio = min(ratio, 1.0 if luma < target else 0.8)

    # 单步限幅, 防止来回穿越目标(亮度震荡)
    ratio = float(np.clip(ratio, 1.0 / max_step, max_step))

    # 总曝光量按 时间 -> 模拟增益 -> 数字增益 的顺序分配(噪声代价递增)
    want = params.total_exposure * ratio
    t = float(np.clip(want, limits.t_min_us, limits.t_max_us))
    rem = want / t
    ag = float(np.clip(rem, limits.ag_min, limits.ag_max))
    dg = float(np.clip(rem / ag, 1.0, limits.dg_max))

    t = quantize_time(t, limits.flicker_hz)
    return AeParams(t, ag, dg)


def ae_run(capture_fn, target, params0, limits, max_iters=8,
           deadband=0.02, clip_guard=0.03, max_val=1023.0, log=None,
           stats_fn=None, black_level=0.0):
    """AE 闭环: capture_fn(params) -> BLC 后 RAW, 迭代到测光亮度进 deadband。

    Parameters
    ----------
    stats_fn : callable(raw, bayer_format, max_val) -> dict, optional
        测光统计函数, 默认 ae_stats(全图均值, 按 black_level 修正
        clip 阈值)。换 ae_stats_weighted(ROI 加权)即实现人脸/点按
        优先测光(黑电平需自行用 functools.partial 绑定)。注意
        target 的口径必须与统计函数一致 —— 收敛的是"测光值",
        不是物理亮度。
    black_level : float
        BLC 使用的黑电平(仅默认统计路径使用, clip/under 阈值按
        实际量程 [0, max_val-black_level] 计算)。

    Returns
    -------
    params : AeParams 收敛(或到顶)后的曝光参数
    history : list[(iter, luma, clip_ratio, params)]
    status : "converged" | "saturated" | "max_iters"
        saturated: 参数已到硬件极限(或防闪烁量化档位/高光保护
        冻结)仍不达标。防闪烁开启时, t 只能取离散档位, 残差靠
        deadband 吸收, 连续两轮参数不变也会被判为 saturated 退出,
        避免死循环。

    Notes
    -----
    收敛判据只看测光亮度 deadband; clip_guard 只限制步长(高光保护),
    不阻塞收敛 —— 逆光场景用 ROI 加权测光时背景高光 clip 是预期
    代价, 若把 clip 纳入收敛条件, 主体永远达不到目标。
    """
    stats_fn = stats_fn or (
        lambda raw, bf, mv: ae_stats(raw, bf, mv, black_level=black_level))
    params = params0
    history = []
    status = "max_iters"
    for it in range(max_iters):
        st = stats_fn(capture_fn(params), "RGGB", max_val)
        history.append((it, st["luma"], st["clip_ratio"], params))
        if log:
            log(it, st["luma"], st["clip_ratio"], params)

        if abs(st["luma"] - target) / target < deadband:
            status = "converged"
            break

        nxt = ae_update(params, st["luma"], target, limits,
                        clip_ratio=st["clip_ratio"], clip_guard=clip_guard)
        same = (nxt.exposure_time_us == params.exposure_time_us
                and nxt.analog_gain == params.analog_gain
                and nxt.digital_gain == params.digital_gain)
        params = nxt
        if same:
            status = "saturated"
            break
    return params, history, status
