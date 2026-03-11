"""Meta-World push-v3 跨实例空间记忆 Demo

验证: robotmem 空间回忆在 Meta-World 上的效果。

机制:
  - 50 个 task instance（不同 obj/target 位置）
  - Phase A: 启发式 + 噪声（无记忆）
  - Phase B: 同策略 + learn（积累经验）
  - Phase C: recall + 记忆驱动参数（空间回忆）
  - 空间回忆: 按 obj_pos 检索最近的成功经验

运行:
  source .venv-pusht/bin/activate
  PYTHONPATH=src python examples/metaworld/demo.py [--seed 42] [--episodes 100]

注: 这是 API 教程。严格实验请参考 experiment.py。
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time

import numpy as np

try:
    import metaworld
except ImportError:
    print("需要: pip install metaworld")
    sys.exit(1)

from robotmem.sdk import RobotMemory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import MetaWorldPushPolicy, MetaWorldMemoryPolicy

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-metaworld")
DB_PATH = os.path.join(DB_DIR, "memory.db")
COLLECTION = "metaworld-push"
NOISE_SCALE = 0.3
MEMORY_WEIGHT = 0.3


def build_context(obs_initial, target_pos, final_reward, success, approach_offset, push_speed):
    return {
        "params": {
            "approach_offset": {"value": float(approach_offset), "type": "scalar"},
            "push_speed": {"value": float(push_speed), "type": "scalar"},
        },
        "spatial": {
            "x": float(obs_initial[4]),  # obj_x
            "y": float(obs_initial[5]),  # obj_y
            "z": float(obs_initial[6]),  # obj_z
        },
        "task": {
            "name": "push-v3",
            "reward": float(final_reward),
            "success": bool(success),
            "target": [float(v) for v in target_pos],
        },
    }


def run_episode(env, policy, target_pos, learn_mem=None, session_id=None, policy_params=None):
    """执行单个 episode，返回 (success, total_reward)"""
    obs_initial, _ = env.reset()
    obs = obs_initial
    total_reward = 0.0
    success = False

    for step in range(150):
        action = policy.act(obs, target_pos)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info.get("success", 0) > 0:
            success = True
        if terminated or truncated:
            break

    if learn_mem is not None:
        approach_offset = getattr(policy, "approach_offset", 0.05)
        push_speed = getattr(policy, "push_speed", 8.0)
        if policy_params:
            approach_offset = policy_params.get("approach_offset", approach_offset)
            push_speed = policy_params.get("push_speed", push_speed)

        ctx = build_context(obs_initial, target_pos, total_reward, success, approach_offset, push_speed)
        learn_mem.learn(
            insight=f"push-v3: reward={total_reward:.1f}, success={success}",
            context=ctx,
            session_id=session_id,
        )

    return success, total_reward


def run_phase_on_instances(env_cls, tasks, policy_factory, episodes_per_instance,
                           learn_mem=None, session_id=None, recall_mem=None, label=""):
    """在多个 task instance 上运行，返回总成功率"""
    total_success = 0
    total_episodes = 0

    for i, task in enumerate(tasks):
        env = env_cls()
        env.set_task(task)

        try:
            for ep in range(episodes_per_instance):
                obs_peek, _ = env.reset()
                target = env.unwrapped._target_pos.copy()
                obj_pos = obs_peek[4:7]

                # 如果有 recall_mem，按空间检索
                if recall_mem is not None:
                    recalled = recall_mem.recall(
                        "successful push strategy",
                        n=3,
                        context_filter={"task.success": True},
                        spatial_sort={
                            "field": "spatial",
                            "origin": [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])],
                        },
                    )
                    policy = MetaWorldMemoryPolicy(
                        policy_factory(), recalled, MEMORY_WEIGHT,
                    )
                else:
                    policy = policy_factory()

                success, reward = run_episode(
                    env, policy, target,
                    learn_mem=learn_mem,
                    session_id=session_id,
                    policy_params={"approach_offset": policy.base.approach_offset if hasattr(policy, "base") else 0.05,
                                   "push_speed": policy.base.push_speed if hasattr(policy, "base") else 8.0},
                )
                total_success += int(success)
                total_episodes += 1
        finally:
            env.close()

    rate = total_success / total_episodes if total_episodes > 0 else 0
    print(f"  {label}: {total_success}/{total_episodes} ({rate:.1%})")
    return rate


def main():
    parser = argparse.ArgumentParser(description="Meta-World push-v3 空间记忆 Demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3, help="每个 instance 的 episode 数")
    parser.add_argument("--instances", type=int, default=20, help="使用多少个 task instance")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # 清空 DB
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    mem = RobotMemory(db_path=DB_PATH, collection=COLLECTION, embed_backend="onnx")

    # 获取 push-v3 task instances
    ml10 = metaworld.ML10(seed=seed)
    env_cls = ml10.train_classes["push-v3"]
    all_tasks = [t for t in ml10.train_tasks if t.env_name == "push-v3"]
    tasks = all_tasks[: args.instances]

    print("=" * 60)
    print(f"Meta-World push-v3 空间记忆 Demo")
    print(f"Seed: {seed}, Instances: {len(tasks)}, Episodes/instance: {args.episodes}")
    print(f"Noise: {NOISE_SCALE}, Memory weight: {MEMORY_WEIGHT}")
    print("=" * 60)

    def make_policy():
        return MetaWorldPushPolicy(noise_scale=NOISE_SCALE)

    try:
        # Phase A: 基线（无记忆）
        print("\n--- Phase A: 基线（启发式 + 噪声，无记忆）---")
        random.seed(seed)
        np.random.seed(seed)
        rate_a = run_phase_on_instances(
            env_cls, tasks, make_policy, args.episodes, label="Phase A",
        )

        # Phase B: 写入记忆
        print("\n--- Phase B: 探索 + 写入记忆 ---")
        random.seed(seed + 1000)
        np.random.seed(seed + 1000)
        with mem.session(context={"task": "push-v3"}) as sid:
            rate_b = run_phase_on_instances(
                env_cls, tasks, make_policy, args.episodes,
                learn_mem=mem, session_id=sid, label="Phase B",
            )

        # Phase C: 空间回忆 + 记忆驱动策略
        print("\n--- Phase C: 空间回忆 + 记忆驱动策略 ---")
        random.seed(seed + 2000)
        np.random.seed(seed + 2000)
        rate_c = run_phase_on_instances(
            env_cls, tasks, make_policy, args.episodes,
            recall_mem=mem, label="Phase C",
        )

        # 结果
        delta = rate_c - rate_a
        print(f"\n{'=' * 60}")
        print("结果:")
        print(f"  Phase A (基线):     {rate_a:.1%}")
        print(f"  Phase B (写入):     {rate_b:.1%}")
        print(f"  Phase C (记忆回忆): {rate_c:.1%}")
        print(f"  Delta (C - A):      {delta:+.1%}")
        print(f"{'=' * 60}")

        if delta > 0.05:
            print(f"记忆提升 {delta:.1%} — 空间回忆对 Meta-World push 有效")
        elif delta > 0:
            print(f"记忆提升 {delta:.1%} — 小幅提升")
        else:
            print(f"记忆未提升 ({delta:.1%})")

        print("\n注: 这是 API 教程，严格实验请参考 experiment.py")

    finally:
        mem.close()


if __name__ == "__main__":
    main()
