"""robotmem SDK 集成示例 — 模拟 LeRobot 训练循环

不依赖 LeRobot 安装，展示 RobotMemory SDK 全部 API 在机器人训练循环中的用法。

运行:
    PYTHONPATH=src python examples/lerobot_integration.py
"""

from __future__ import annotations

import json
import os
import random
import shutil

from robotmem.sdk import RobotMemory

# 数据隔离 — 每次运行清空
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-lerobot-demo")
DB_PATH = os.path.join(DB_DIR, "memory.db")

EPISODES = 20
RECALL_N = 3


def random_obs():
    """模拟机器人观测"""
    return {
        "position": [round(random.uniform(0.5, 1.5), 2) for _ in range(3)],
        "velocity": [round(random.uniform(-0.1, 0.1), 3) for _ in range(3)],
        "trajectory": [[round(random.uniform(-1, 1), 2) for _ in range(4)] for _ in range(20)],
    }


def random_action():
    """模拟动作"""
    return [round(random.uniform(-1, 1), 2) for _ in range(4)]


def main():
    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem SDK 集成示例 — 模拟 LeRobot 训练循环")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    mem = RobotMemory(db_path=DB_PATH, embed_backend="onnx")
    print(f"SDK: {mem}")

    # ── Phase 1: 训练循环（learn + recall + save_perception + session） ──

    print(f"\n{'─'*40}")
    print(f"Phase 1: 训练循环 ({EPISODES} episodes)")
    print(f"{'─'*40}")

    successes = 0
    recall_hits = 0

    with mem.session(context={"robot": "aloha", "task": "pick_place", "env": "sim"}) as sid:
        for ep in range(EPISODES):
            obs = random_obs()
            action = random_action()
            success = random.random() > 0.5
            reward = 1.0 if success else -0.5

            # recall 过去经验（前几个 episode 没有数据，返回空）
            tips = mem.recall(
                f"pick object at position {obs['position']}",
                n=RECALL_N,
                session_id=None,  # 不限 session，搜全库
            )
            if tips:
                recall_hits += 1

            # learn 本次经验
            result = mem.learn(
                insight=f"Episode {ep}: {'成功' if success else '失败'}, reward={reward:.1f}, "
                        f"object_pos={obs['position']}",
                context={
                    "params": {
                        "action": {"value": action, "type": "vector"},
                        "reward": {"value": reward, "type": "scalar"},
                    },
                    "spatial": {
                        "frame": "world",
                        "object_position": obs["position"],
                    },
                    "robot": {"id": "aloha-01", "type": "ALOHA", "dof": 14},
                    "task": {
                        "name": "pick_place",
                        "success": success,
                        "steps": len(obs["trajectory"]),
                    },
                },
                session_id=sid,
            )

            # save_perception 轨迹
            mem.save_perception(
                description=f"pick_place 轨迹: {len(obs['trajectory'])} 步, pos={obs['position']}",
                perception_type="procedural",
                data=json.dumps({"trajectory": obs["trajectory"][:5]}),  # 截取前 5 步
                session_id=sid,
            )

            if success:
                successes += 1

            if (ep + 1) % 10 == 0:
                rate = successes / (ep + 1)
                print(f"  [{ep+1}/{EPISODES}] 成功率: {rate:.0%}, recall 命中: {recall_hits}/{ep+1}")

    print(f"\n  训练完成: {successes}/{EPISODES} 成功, recall 命中 {recall_hits}/{EPISODES}")

    # ── Phase 2: batch_learn 批量写入 ──

    print(f"\n{'─'*40}")
    print("Phase 2: batch_learn 批量写入")
    print(f"{'─'*40}")

    batch_results = mem.batch_learn([
        "ALOHA pick_place: 左臂先接近，右臂辅助稳定",
        "ALOHA pick_place: 力矩超过 15N 时容易滑落",
        {"insight": "ALOHA pick_place: 从侧面接近比正面更稳定", "context": "多次实验对比结论"},
    ])
    created = sum(1 for r in batch_results if r.get("status") == "created")
    print(f"  批量写入: {created}/{len(batch_results)} 成功")

    # ── Phase 3: recall 验证 ──

    print(f"\n{'─'*40}")
    print("Phase 3: recall 验证")
    print(f"{'─'*40}")

    # 检索成功经验
    tips = mem.recall("pick object successfully", n=5)
    print(f"  搜 'pick object successfully': {len(tips)} 条结果")
    for i, t in enumerate(tips[:3]):
        print(f"    [{i+1}] {t['content'][:70]}")

    # 带 context_filter 检索
    tips_filtered = mem.recall(
        "pick_place",
        n=5,
        context_filter={"task.success": True},
    )
    print(f"  搜 'pick_place' (success=True): {len(tips_filtered)} 条结果")

    # ── Phase 4: forget + update ──

    print(f"\n{'─'*40}")
    print("Phase 4: forget + update")
    print(f"{'─'*40}")

    # 找一条记忆来 update
    all_memories = mem.recall("Episode", n=1)
    if all_memories:
        mid = all_memories[0]["id"]
        old = all_memories[0]["content"][:50]

        # update
        update_result = mem.update(mid, new_content="[已修正] 实验条件变更后的新结论")
        print(f"  update #{mid}: '{old}...' → '{update_result['new_content']}'")

        # 再找一条来 forget
        more = mem.recall("Episode", n=2)
        if len(more) >= 2:
            fid = more[1]["id"]
            forget_result = mem.forget(fid, reason="测试 forget 功能")
            print(f"  forget #{fid}: {forget_result['status']}")

    # ── 统计 ──

    print(f"\n{'='*60}")
    try:
        mem.recall("", n=1)  # 空查询会被拒绝
        print("  空查询未被拒绝（异常）")
    except Exception:
        print("  空查询被正确拒绝 ✓")
    total = mem.recall("pick", n=100)
    print(f"SDK 集成验证完成")
    print(f"  总记忆数（估计）: {len(total)}+ 条")
    print(f"  API 覆盖: learn, recall, save_perception, batch_learn, session, forget, update")
    print(f"{'='*60}")

    mem.close()


if __name__ == "__main__":
    main()
