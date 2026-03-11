"""LeRobot + robotmem 训练演示 — 记忆引导策略（API 教程）

Mock 环境有隐藏的最优策略 [0.3, -0.5]。
三阶段演示 robotmem recall 引导策略的 API 用法：
  Phase A: 基线（随机策略，无记忆）→ ~7-15% 成功
  Phase B: 写入记忆（随机策略 + learn）→ ~7-15% 成功
  Phase C: 利用记忆（recall → 引导策略）→ ~90%+ 成功

注: 本示例为 API 用法教程，非严格实验。
严格实验请参考 examples/fetch_push/experiment.py。

运行:
    PYTHONPATH=src python3 examples/lerobot_train_demo.py
    PYTHONPATH=src python3 examples/lerobot_train_demo.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lerobot_callback import RobotMemCallback

# 数据隔离 — 每次运行清空
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-lerobot-train")
DB_PATH = os.path.join(DB_DIR, "memory.db")


class SmartMockEnv:
    """有可学习隐藏策略的 Mock 环境

    隐藏最优动作: [0.3, -0.5]
    每 episode 10 步，成功条件: episode 平均动作与最优的欧式距离 < 0.35
    随机策略成功率 ~15-20%，记忆引导策略成功率 ~90%+
    """

    OPTIMAL_ACTION = [0.3, -0.5]
    SUCCESS_RADIUS = 0.35
    STEPS_PER_EP = 10

    def __init__(self):
        self._step = 0
        self._actions: list[list[float]] = []

    def reset(self):
        self._step = 0
        self._actions = []
        return {"observation": [random.uniform(-1, 1) for _ in range(4)]}

    def step(self, action):
        self._step += 1
        self._actions.append(list(action[:2]))

        # 奖励 = 1 - 距离最优的欧式距离
        dist = math.sqrt(sum(
            (a - o) ** 2 for a, o in zip(action, self.OPTIMAL_ACTION)
        ))
        reward = max(0.0, 1.0 - dist)

        done = self._step >= self.STEPS_PER_EP

        # 成功 = 全 episode 平均动作在最优附近
        success = False
        if done and self._actions:
            dim = len(self.OPTIMAL_ACTION)
            avg = [
                sum(a[i] for a in self._actions) / len(self._actions)
                for i in range(dim)
            ]
            avg_dist = math.sqrt(sum(
                (a - o) ** 2 for a, o in zip(avg, self.OPTIMAL_ACTION)
            ))
            success = avg_dist < self.SUCCESS_RADIUS

        obs = {"observation": [random.uniform(-1, 1) for _ in range(4)]}
        info = {"is_success": success}
        return obs, reward, done, info

    def close(self):
        pass


def extract_guided_action(tips: list[dict], dim: int = 2) -> list[float] | None:
    """从 recall 成功记忆中提取引导动作

    读取 context.params.avg_action.value，取多条成功记忆的等权平均。
    """
    actions: list[list[float]] = []
    for tip in tips:
        ctx = tip.get("context")
        # context 可能是 dict 或 JSON 字符串
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(ctx, dict):
            continue
        avg_act = ctx.get("params", {}).get("avg_action", {}).get("value")
        if isinstance(avg_act, list) and len(avg_act) >= dim:
            actions.append(avg_act[:dim])

    if not actions:
        return None

    # 多条成功经验的等权平均
    result = [
        sum(a[i] for a in actions) / len(actions)
        for i in range(dim)
    ]
    return result


def run_episode(env, cb: RobotMemCallback | None, phase: str, use_memory: bool = False):
    """执行单个 episode"""
    obs = env.reset()

    # Phase C: recall 成功经验 → 提取引导动作
    guided_action = None
    if use_memory and cb is not None:
        tips = cb.recall_tips(
            "successful episode strategy optimal action",
            n=5,
            context_filter={"task.success": True},
        )
        guided_action = extract_guided_action(tips)

    trajectory: list[list[float]] = []
    ep_reward = 0.0
    success = False

    for _ in range(SmartMockEnv.STEPS_PER_EP):
        if guided_action is not None:
            # 记忆引导：基于回忆的最优动作 + 探索噪声
            action = [v + random.gauss(0, 0.1) for v in guided_action]
        else:
            # 随机策略
            action = [random.uniform(-1, 1) for _ in range(2)]

        result = env.step(action)
        obs, reward, done, info = result
        trajectory.append(list(action[:2]))
        ep_reward += reward

        if done:
            success = info.get("is_success", False)
            break

    # Phase B/C: 通过回调记录经验
    if phase in ("B", "C") and cb is not None:
        avg_action = [
            sum(a[i] for a in trajectory) / len(trajectory)
            for i in range(2)
        ]
        cb.on_episode_end(
            episode_data={
                "reward": ep_reward,
                "success": success,
                "steps": len(trajectory),
                "context": {
                    "params": {
                        "avg_action": {"value": avg_action, "type": "vector"},
                        "total_reward": {"value": ep_reward, "type": "scalar"},
                    },
                    "task": {"name": "smart_mock"},
                },
            },
            trajectory=trajectory[:5],
        )

    return success


def run_phase(
    env,
    cb: RobotMemCallback | None,
    phase: str,
    episodes: int,
    use_memory: bool = False,
) -> float:
    """执行一个 Phase，返回成功率"""
    successes = 0
    for ep in range(episodes):
        ok = run_episode(env, cb, phase, use_memory)
        successes += int(ok)
        if (ep + 1) % 10 == 0:
            rate = successes / (ep + 1)
            print(f"  Phase {phase} [{ep+1}/{episodes}] 成功率: {rate:.0%}")
    return successes / episodes


def parse_args():
    parser = argparse.ArgumentParser(
        description="robotmem API 演示 — 记忆引导策略（Mock 环境）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现运行）")
    parser.add_argument("--episodes", type=int, default=30, help="每阶段 episode 数（默认 30）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)

    episodes = args.episodes

    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem API 演示 — 记忆引导策略（Mock 环境）")
    print(f"隐藏最优策略: {SmartMockEnv.OPTIMAL_ACTION}")
    print(f"成功半径: {SmartMockEnv.SUCCESS_RADIUS}")
    print(f"每阶段 {episodes} episodes × {SmartMockEnv.STEPS_PER_EP} 步")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    env = SmartMockEnv()

    # ── Phase A: 基线（随机策略，不用记忆）──
    print("\n--- Phase A: 基线（随机策略，无记忆）---")
    rate_a = run_phase(env, None, "A", episodes)

    # ── Phase B+C: 开始 session，写入 + 利用记忆 ──
    cb = RobotMemCallback(db_path=DB_PATH, collection="smart_mock")
    try:
        sid = cb.on_train_begin({
            "robot": "mock_2d",
            "task": "smart_mock",
            "policy": "random → memory_guided",
        })
        print(f"\nSession: {sid}")

        # Phase B: 随机策略 + 写入记忆（积累经验）
        print("\n--- Phase B: 随机策略 + 写入记忆 ---")
        rate_b = run_phase(env, cb, "B", episodes)

        # Phase C: 利用记忆引导策略
        print("\n--- Phase C: 记忆引导策略 ---")
        rate_c = run_phase(env, cb, "C", episodes, use_memory=True)

        # 结束 session
        result = cb.on_train_end({"success_rate": rate_c})

        # ── 结果 ──
        delta = rate_c - rate_a

        print(f"\n{'=' * 60}")
        print("演示结果")
        print(f"{'=' * 60}")
        print(f"  Phase A (基线):     {rate_a:.0%}")
        print(f"  Phase B (写入):     {rate_b:.0%}")
        print(f"  Phase C (利用):     {rate_c:.0%}")
        print(f"  提升 (C - A):       {delta:+.0%}")
        if result:
            cnt = result.get("summary", {}).get("memory_count", "?")
            cons = result.get("consolidated", {}).get("superseded_count", 0)
            print(f"  Session 记忆数:     {cnt}")
            print(f"  巩固合并:           {cons} 条")
        print(f"\n  注: 本示例演示 recall → 策略引导的 API 用法。")
        print(f"  严格实验请参考 examples/fetch_push/experiment.py。")
        print(f"\n数据存储于: {DB_DIR}")
        print(f"{'=' * 60}")
    finally:
        env.close()
        cb.close()


if __name__ == "__main__":
    main()
