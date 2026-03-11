# robotmem Examples

robotmem SDK 集成示例。所有示例定位为 **API 教程**，展示 robotmem 在机器人训练循环中的用法。

## 示例一览

| 示例 | 文件 | 依赖 | 用途 |
|------|------|------|------|
| SDK 全 API 演示 | `lerobot_integration.py` | 无 | 覆盖 7 个 API（learn/recall/save_perception/forget/update/batch_learn/session） |
| LeRobot 回调 | `lerobot_callback.py` | 无 | 可插入训练循环的 RobotMemCallback 类 |
| 记忆引导策略 | `lerobot_train_demo.py` | 无 | Mock 环境三阶段对比：随机 → 写入 → recall 引导 |
| FetchPush Demo | `fetch_push/demo.py` | gymnasium-robotics | MuJoCo 仿真三阶段对比：基线 → 写入 → PhaseAwareMemoryPolicy |
| 跨环境泛化 | `fetch_push/cross_env.py` | gymnasium-robotics | FetchPush 经验帮助 FetchSlide（SDK 版） |

## 快速运行

```bash
# 零依赖 — Mock 环境 API 教程
PYTHONPATH=src python3 examples/lerobot_integration.py
PYTHONPATH=src python3 examples/lerobot_train_demo.py

# 可复现运行
PYTHONPATH=src python3 examples/lerobot_train_demo.py --seed 42

# MuJoCo 仿真（需要 gymnasium-robotics）
cd examples/fetch_push
PYTHONPATH=../../src python demo.py
PYTHONPATH=../../src python demo.py --seed 42 --episodes 50

# 跨环境泛化
PYTHONPATH=../../src python cross_env.py --seed 42
```

## 定位说明

这些示例是 **API 用法教程**，用于帮助开发者快速上手 robotmem SDK。

如需严格实验数据（多 seed、统计显著性），请参考：
- `fetch_push/experiment.py` — 300 episodes, 10 seeds, 动态查询 + spatial_sort

## 文件说明

### lerobot_integration.py

SDK 全 API 演示。不依赖任何外部库，模拟 LeRobot 训练循环，覆盖 robotmem 全部 7 个 API。适合第一次接触 robotmem 的用户。

### lerobot_callback.py

`RobotMemCallback` 类 — 可直接插入 LeRobot 训练循环的回调。所有异常内部捕获，不中断训练。

### lerobot_train_demo.py

Mock 环境有隐藏最优策略 `[0.3, -0.5]`。三阶段演示 recall 引导策略的效果：
- Phase A: 随机策略（~7-15% 成功）
- Phase B: 随机策略 + learn（积累经验）
- Phase C: recall → 引导策略（~90%+ 成功）

支持 `--seed` 和 `--episodes` 参数。

### fetch_push/demo.py

MuJoCo FetchPush-v4 真实仿真。使用 PhaseAwareMemoryPolicy（只在推送阶段施加记忆偏置）。
默认 100 episodes/phase，支持 `--seed` 和 `--episodes` 参数。

### fetch_push/cross_env.py

跨环境泛化演示 — robotmem 核心价值："学一次，换个环境还能用"。
先在 FetchPush 积累经验，再在 FetchSlide 中 recall 使用。自包含，不依赖 experiment.py。

### fetch_push/experiment.py

严格实验脚本。300 episodes, 10 seeds, 动态查询 + spatial_sort。用于论文数据，非 SDK 版本。

### fetch_push/policies.py

策略模块：HeuristicPolicy（四阶段 heuristic）、MemoryPolicy（基础版）、PhaseAwareMemoryPolicy（阶段感知版）、SlidePolicy（FetchSlide 专用）。
