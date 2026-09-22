# -*- coding: utf-8 -*-
"""
a3a_state —— 3A 状态机
========================

真实产品里 3A 不是"一次性收敛"而是常驻控制状态机: 预览流每一帧
喂入统计量, 状态机决定"调还是不调", 控制律(ae_update 等)与决策
解耦 —— 状态机管"何时调/何时停", 算法管"怎么调"。

【状态】
    CONVERGING    参数未达标, 持续调整
    CONVERGED     进入容差并稳定 N 帧(参数可微调或冻结)
    LOCKED        用户锁定(点按对焦/曝光锁定), 参数冻结
    RECONVERGING  场景突变(统计漂移超阈值)/解锁后重新收敛

【设计要点 / 面试点】
1. 场景变化检测用"与收敛参考值的相对差", 收敛参考值用 EMA 慢速
   跟随 —— 正常抖动(deadband 内)不触发重收敛, 缓慢的环境渐变
   (日落)能跟随, 只有突变(入隧道/开灯)才触发;
2. 锁定期间忽略场景变化(用户明确要求锁定), 解锁后强制走一次
   重收敛(参数可能已不适合新场景);
3. 重收敛与初收敛共用同一控制律, 状态切换不换算法;
4. 产品化扩展: 每个 A 有自己的子状态与优先级仲裁(用户点按 >
   人脸检测 > 场景检测), 拍照瞬间触发改参数冻结(safe guard)。
"""

import numpy as np


class A3AState:
    CONVERGING = "CONVERGING"
    CONVERGED = "CONVERGED"
    LOCKED = "LOCKED"
    RECONVERGING = "RECONVERGING"


class A3AStateMachine:
    """轻量 3A 决策状态机(AE 亮度通道为例, AWB/AF 同构可复用)。"""

    def __init__(self, target=450.0, tol=0.03, stable_frames=2,
                 scene_change=0.30, ema=0.1):
        """
        Parameters
        ----------
        target : float
            测光目标值(与 AE 的 target 同口径)。
        tol : float
            收敛容差(相对 target), 等价 AE deadband。
        stable_frames : int
            连续稳定帧数, 防止单帧噪声导致状态抖动。
        scene_change : float
            与收敛参考值的相对差超过该值 -> 判定场景突变。
        ema : float
            收敛参考值的慢速跟随系数。
        """
        self.target = float(target)
        self.tol = float(tol)
        self.stable_frames = int(stable_frames)
        self.scene_change = float(scene_change)
        self.ema = float(ema)

        self.state = A3AState.CONVERGING
        self.luma_ref = None   # 收敛时的参考亮度(EMA 跟随)
        self._stable = 0

    def update(self, luma, locked=False):
        """每帧喂入当前测光亮度, 返回(可能更新后的)状态。"""
        # 用户锁定优先级最高: 锁定期间一切统计漂移都忽略
        if locked:
            if self.state != A3AState.LOCKED:
                self.state = A3AState.LOCKED
            return self.state
        if self.state == A3AState.LOCKED:
            # 解锁: 参数可能已不适合当前场景, 强制重收敛
            self.state = A3AState.RECONVERGING
            self._stable = 0
            return self.state

        err = abs(luma - self.target) / self.target
        scene_break = (self.luma_ref is not None and
                       abs(luma - self.luma_ref) /
                       max(self.luma_ref, 1e-6) > self.scene_change)

        # 场景突变: 从稳定态转入重收敛
        if scene_break and self.state == A3AState.CONVERGED:
            self.state = A3AState.RECONVERGING
            self._stable = 0

        if err < self.tol:
            self._stable += 1
            if self._stable >= self.stable_frames:
                if self.state != A3AState.CONVERGED:
                    self.state = A3AState.CONVERGED
                    self.luma_ref = float(luma)
                else:
                    # 参考值慢速跟随, 正常抖动不触发, 缓变能跟随
                    self.luma_ref = ((1 - self.ema) * self.luma_ref
                                     + self.ema * float(luma))
        else:
            self._stable = 0
            self.state = (A3AState.RECONVERGING
                          if self.luma_ref is not None
                          else A3AState.CONVERGING)
        return self.state
