"""robotmem FetchPush 快速 Demo — SDK 版

运行方式:
  cd examples/fetch_push
  pip install gymnasium-robotics  # 需要 MuJoCo
  PYTHONPATH=../../src python demo.py
  PYTHONPATH=../../src python demo.py --seed 42

三阶段（默认各 100 episodes）:
  Phase A: 基线（heuristic，无记忆）
  Phase B: 记忆写入（learn）
  Phase C: 记忆利用（recall → PhaseAwareMemoryPolicy）

注: 本示例为 API 用法教程。严格实验请参考 experiment.py（300 episodes, 10 seeds）。
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time

try:
    import gymnasium_robotics  # noqa: F401
    import gymnasium
    import numpy as np
except ImportError:
    print("需要安装: pip install gymnasium-robotics")
    sys.exit(1)

from robotmem.sdk import RobotMemory

from policies import HeuristicPolicy, PhaseAwareMemoryPolicy

# 数据隔离
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-demo")
DB_PATH = os.path.join(DB_DIR, "memory.db")

COLLECTION = "demo"
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def build_context(obs, actions, success, steps, total_reward):
    """构建 context dict — 四区域结构（params/spatial/task）"""
    recent = actions[-10:] if len(actions) >= 10 else actions
    avg_action = np.mean(recent, axis=0) if recent else np.zeros(4)
    return {
        "params": {
            "approach_velocity": {"value": avg_action[0:3].tolist(), "type": "vector"},
            "grip_force": {"value": float(avg_action[3]), "type": "scalar"},
            "final_distance": {
                "value": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
                "unit": "m",
            },
        },
        "spatial": {
            "grip_position": obs["observation"][0:3].tolist(),
            "object_position": obs["observation"][3:6].tolist(),
            "target_position": obs["desired_goal"].tolist(),
        },
        "task": {
            "name": "push_to_target",
            "success": bool(success),
            "steps": steps,
            "total_reward": float(total_reward),
        },
    }


def run_episode(env, policy, phase, mem, session_id=None):
    """执行单个 episode"""
    # Phase C: recall 成功经验
    recalled = []
    if phase == "C":
        recalled = mem.recall(
            "push cube to target position",
            n=RECALL_N,
            context_filter={"task.success": True},
        )

    # PhaseAwareMemoryPolicy: 只在推送阶段施加 bias（避免干扰升高/绕后/下降）
    active_policy = PhaseAwareMemoryPolicy(policy, recalled, MEMORY_WEIGHT) if recalled else policy

    # 跑 episode
    obs, _ = env.reset()
    actions = []
    total_reward = 0.0
    for _ in range(50):
        action = active_policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action.copy())
        total_reward += reward
        if terminated or truncated:
            break

    success = info.get("is_success", False)

    # Phase B/C: learn 经验
    if phase in ("B", "C"):
        ctx = build_context(obs, actions, success, len(actions), total_reward)
        dist = ctx["params"]["final_distance"]["value"]
        mem.learn(
            insight=f"FetchPush: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步",
            context=ctx,
            session_id=session_id,
        )

    return success


def run_phase(env, policy, phase, mem, episodes, session_id=None):
    """执行一个 Phase"""
    successes = 0
    for ep in range(episodes):
        ok = run_episode(env, policy, phase, mem, session_id)
        successes += int(ok)
        if (ep + 1) % 10 == 0:
            print(f"  Phase {phase} [{ep+1}/{episodes}] 成功率: {successes/(ep+1):.0%}")
    return successes / episodes


def parse_args():
    parser = argparse.ArgumentParser(
        description="robotmem FetchPush Demo — 记忆驱动推送（SDK 版）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现运行）")
    parser.add_argument("--episodes", type=int, default=100, help="每阶段 episode 数（默认 100）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    episodes = args.episodes

    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    print("=" * 50)
    print("robotmem FetchPush Demo (SDK)")
    print(f"每阶段 {episodes} episodes")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print(f"DB: {DB_PATH}")
    print("=" * 50)

    mem = RobotMemory(db_path=DB_PATH, collection=COLLECTION, embed_backend="onnx")
    env = gymnasium.make("FetchPush-v4")
    policy = HeuristicPolicy()

    t0 = time.time()

    try:
        # Phase A: 基线（无记忆）
        print("\n--- Phase A: 基线（无记忆）---")
        rate_a = run_phase(env, policy, "A", mem, episodes)

        # Phase B+C: 记忆写入 + 利用（同一 session）
        with mem.session(context={"task": "push_to_target", "env": "FetchPush-v4"}) as sid:
            print("\n--- Phase B: 写入记忆 ---")
            rate_b = run_phase(env, policy, "B", mem, episodes, sid)

            print("\n--- Phase C: 利用记忆 ---")
            rate_c = run_phase(env, policy, "C", mem, episodes, sid)

        elapsed = time.time() - t0

        # 结果
        delta = rate_c - rate_a

        print(f"\n{'=' * 50}")
        print("演示结果")
        print(f"{'=' * 50}")
        print(f"  Phase A: {rate_a:.0%}")
        print(f"  Phase B: {rate_b:.0%}")
        print(f"  Phase C: {rate_c:.0%}")
        print(f"  提升 (C - A): {delta:+.0%}")
        print(f"  耗时: {elapsed:.0f}s")
        print(f"\n  注: 本示例演示 recall → PhaseAwareMemoryPolicy 的 API 用法。")
        print(f"  严格实验请参考 experiment.py（300 episodes, 10 seeds）。")
        print(f"\n数据存储于: {DB_DIR}")
    finally:
        env.close()
        mem.close()


if __name__ == "__main__":
    main()
