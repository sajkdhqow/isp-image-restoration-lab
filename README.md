# ISP & Image Restoration Lab

从 Bayer RAW 到成片的全栈实践:**12 级传统 ISP 流水线 + 3A 闭环控制 + 生成式低照度恢复**(回归 CNN / Rectified Flow),全部手写、CPU 可复现、每个结论都有量化数字支撑。

## 亮点结果(held-out 场景, 统一噪声口径)

| 阶段 | 成果 |
|------|------|
| 传统流水线 | 12 级全手写;Demosaic 三算法量化对比(edge-aware 33.66 dB) |
| 3A 闭环 | AE 欠曝 26 倍 4 轮收敛;ROI 测光主体提亮 4.1 倍;AF 误差 11/1023;PDAF 混合对焦采图省 87% |
| 端到端 U-Net | 极暗场景反超传统全链路 **+4.7~9.5 dB**(1/50~1/133 照度) |
| 生成式 Rectified Flow | 锐度比 0.99 贴真值;reflow 蒸馏后 **1 步 0.16s** |
| 双链路架构 | 预览(回归)0.055s / 拍照(生成)0.174s |

## 三种打开方式

```bash
# 1) 一键全流程: 12 级流水线 + 算法对比 + 3A 闭环 + 夜景灯牌 AE demo
python main.py

# 2) ML 低照度恢复: 预览/拍照双链路(需先训练或直接用随仓库 demo 权重)
python ml/run_ml_camera.py

# 3) Notebook 教学入口(逐步讲解版)
jupyter notebook notebooks/
```

训练(均可 CPU 复现):

```bash
python ml/train.py --iters 6000       # 回归 U-Net, ~13min
python ml/train_flow.py --iters 5000  # 条件 Rectified Flow, ~17min
python ml/train_reflow.py             # reflow 蒸馏: 8 步质量压进 1 步, ~5min
python ml/eval_demo.py                # 传统 vs U-Net(held-out 三档曝光)
python ml/eval_flow.py                # 传统/U-Net/Flow/蒸馏 四方对比(PSNR/LPIPS/延迟)
```

RAW 文件直通(真实 sensor dump):

```bash
python run_real_raw.py --input your.raw --width 4096 --height 3072 --bits 10 --bayer RGGB [--mipi]
```

## 项目结构

```text
main.py                 # 一键: RAW 模拟 -> 12 级流水线 -> 算法对比 -> 3A 协同
run_real_raw.py         # RAW 文件直通: 裸 dump/MIPI -> 黑电平三来源估计 -> 出图
isp/                    # 传统流水线模块(DPC/BLC/BNR/LSC/AWB/Demosaic/CCM/Gamma/...)
ml/                     # 端到端与生成式(U-Net 回归 / Rectified Flow / reflow 蒸馏 / 双链路)
tools/simulate_raw.py   # sensor 仿真器: 暗角/色偏/黑电平/噪声/坏点
notebooks/              # Jupyter 教学入口(传统 ISP 分步 + CNN 降噪对比)
docs/                   # 深入文档(见下)
data/                   # 场景图(RAW 可由脚本重建)
```

## 文档(面试向深读)

| 文档 | 内容 |
|------|------|
| [docs/interview_notes.md](docs/interview_notes.md) | 18 章面试问答手册(模块原理 + 3A + ML) |
| [docs/optimization_log.md](docs/optimization_log.md) | 3A 五项优化: 方法/结果/剩余空间/迭代方法论 |
| [docs/ml_notes.md](docs/ml_notes.md) | ML 化全程: SID 复现/踩坑/回归 vs 生成/迭代路线 |
| [docs/optimization_report.md](docs/optimization_report.md) | 第二轮优化总结(AE 修复/LPIPS/reflow/双链路) |

## 设计上的几个"真实工程点"

1. **DPC 必须同色域检测**:Bayer 图直接取 3×3 会跨色误杀,先拆四通道再取邻域中值;
2. **CCM 与 AWB 联动标定**:白平衡增益改变色彩响应,CCM 必须按 `inv(D·M)` 重标,否则整体偏色;
3. **LSC 必须白场标定**:用拍摄内容自标定会把彩色物体拉灰且不可恢复;
4. **AE 高光保护的两个真 bug**:clip 阈值落在不可达量程(保护成死代码)、guard 发散语义(欠曝+裁切时曝光一路压到底),修复后才有"夜景灯牌"仲裁能力;
5. **PDAF 迭代符号反转**:混合对焦误差 146→10 的完整调试案例(先验测量、再归因符号、量化验证);
6. **ML 评估三条纪律**:场景隔离(单场景训练=背题)、基线给全套装备、所有路径同帧同噪声口径(踩过:口径不一致导致假"交叉点"结论)。

## 诚实边界

- 离焦用高斯近似、噪声为合成模型(无行噪声/DSNU),**方法可复现,数字不直接迁移真机**;
- 传统基线的 AWB/CCM/LSC 与仿真器配对(完美先验),真实场景需按模组重新标定;
- demo 权重 0.5M 参数、CPU 训练,用于展示方法与取舍,不是 SOTA。

## License

MIT
