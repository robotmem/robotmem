"""跨环境泛化 Demo — FetchPush 经验帮助 FetchSlide（SDK 版）

验证 robotmem 核心命题："学一次，换个环境还能用"

FetchSlide vs FetchPush:
- 相同：7-DOF Fetch 机器人，4 维动作，桌面场景
- 不同：Slide 需要把物体滑到远处目标（物体可能滑出桌面）
  Slide 的目标通常更远，需要更大力度和惯性

实验设计（自包含，三阶段）:
  Phase 0: FetchPush 写入记忆（50 episodes）— 积累 Push 经验
  Phase A: FetchSlide 基线（50 episodes，无记忆）
  Phase B: FetchSlide + FetchPush 记忆（50 episodes，跨环境 recall）

运行:
  cd examples/fetch_push
  pip install gymnasium-robotics  # 需要 MuJoCo
  PYTHONPATH=../../src python cross_env.py
  PYTHONPATH=../../src python cross_env.py --seed 42

注: 本示例为跨环境泛化的 API 教程。
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

from policies import HeuristicPolicy, SlidePolicy, PhaseAwareMemoryPolicy

# 数据隔离
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-crossenv")
DB_PATH = os.path.join(DB_DIR, "memory.db")

PUSH_COLLECTION = "push_exp"
SLIDE_COLLECTION = "slide_exp"
MEMORY_WEIGHT = 0.3
RECALL_N = 5


def build_context(obs, actions, success, steps, total_reward, task_name):
    """构建 context dict"""
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
            "name": task_name,
            "success": bool(success),
            "steps": steps,
            "total_reward": float(total_reward),
        },
    }


def run_episode(env, policy, mem, collection, recall_collection=None, session_id=None, task_name="push"):
    """执行单个 episode

    Args:
        recall_collection: 从哪个 collection recall（None=不 recall）
        collection: learn 写入到哪个 collection
        session_id: 当前 session（None=不 learn）
    """
    # recall 成功经验（跨环境时 recall_collection != collection）
    recalled = []
    if recall_collection:
        recalled = mem.recall(
            "push or slide object to target position",
            n=RECALL_N,
            context_filter={"task.success": True},
            collection=recall_collection,
        )

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

    # learn 经验
    if session_id:
        ctx = build_context(obs, actions, success, len(actions), total_reward, task_name)
        dist = ctx["params"]["final_distance"]["value"]
        mem.learn(
            insight=f"Fetch{task_name.capitalize()}: {'成功' if success else '失败'}, 距离 {dist:.3f}m, {len(actions)} 步",
            context=ctx,
            session_id=session_id,
            collection=collection,
        )

    return success


def run_phase(env, policy, mem, episodes, collection, recall_collection=None,
              session_id=None, task_name="push", label=""):
    """执行一个 Phase"""
    successes = 0
    for ep in range(episodes):
        ok = run_episode(env, policy, mem, collection, recall_collection, session_id, task_name)
        successes += int(ok)
        if (ep + 1) % 10 == 0:
            print(f"  {label} [{ep+1}/{episodes}] 成功率: {successes/(ep+1):.0%}")
    return successes / episodes


def parse_args():
    parser = argparse.ArgumentParser(
        description="robotmem 跨环境泛化 Demo — FetchPush → FetchSlide（SDK 版）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现运行）")
    parser.add_argument("--episodes", type=int, default=50, help="每阶段 episode 数（默认 50）")
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

    print("=" * 60)
    print("robotmem 跨环境泛化 Demo: FetchPush → FetchSlide")
    print(f"每阶段 {episodes} episodes")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    mem = RobotMemory(db_path=DB_PATH, embed_backend="onnx")
    t0 = time.time()

    try:
        # ── Phase 0: FetchPush 写入记忆 ──
        print(f"\n--- Phase 0: FetchPush 写入记忆（{episodes} episodes）---")
        push_env = gymnasium.make("FetchPush-v4")
        push_policy = HeuristicPolicy()
        try:
            with mem.session(context={"task": "push_to_target", "env": "FetchPush-v4"}) as push_sid:
                rate_push = run_phase(
                    push_env, push_policy, mem, episodes,
                    collection=PUSH_COLLECTION, session_id=push_sid,
                    task_name="push", label="Push",
                )
            print(f"  FetchPush 完成: {rate_push:.0%} 成功率")
        finally:
            push_env.close()

        # ── FetchSlide 阶段 ──
        slide_env = gymnasium.make("FetchSlide-v4")
        slide_policy = SlidePolicy()

        try:
            # Phase A: FetchSlide 基线（无记忆）
            print(f"\n--- Phase A: FetchSlide 基线（无记忆）---")
            rate_a = run_phase(
                slide_env, slide_policy, mem, episodes,
                collection=SLIDE_COLLECTION,
                task_name="slide", label="Phase A",
            )

            # Phase B: FetchSlide + FetchPush 记忆（跨环境 recall）
            # 注: Phase B 只 recall Push 记忆、不 learn Slide 经验（纯跨环境测试）
            print(f"\n--- Phase B: FetchSlide + FetchPush 记忆（跨环境）---")
            rate_b = run_phase(
                slide_env, slide_policy, mem, episodes,
                collection=SLIDE_COLLECTION, recall_collection=PUSH_COLLECTION,
                task_name="slide", label="Phase B",
            )
        finally:
            slide_env.close()

        elapsed = time.time() - t0

        # 结果
        delta = rate_b - rate_a

        print(f"\n{'=' * 60}")
        print("跨环境泛化演示结果")
        print(f"{'=' * 60}")
        print(f"  Phase 0 (Push 写入):   {rate_push:.0%}")
        print(f"  Phase A (Slide 基线):  {rate_a:.0%}")
        print(f"  Phase B (Slide+Push):  {rate_b:.0%}")
        print(f"  跨环境提升 (B - A):    {delta:+.0%}")
        print(f"  耗时: {elapsed:.0f}s")
        print(f"\n  注: 本示例演示跨环境 recall 的 API 用法。")
        print(f"  \"学一次，换个环境还能用\" — robotmem 核心价值。")
        print(f"\n数据存储于: {DB_DIR}")

    finally:
        mem.close()


if __name__ == "__main__":
    main()
