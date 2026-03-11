"""robotmem Perception 回放测试 — 录制 → 回放 → 验证

验证目的: PerceptionBuffer 在真实数据量下存储完整、无丢失

运行方式:
  cd examples/fetch_push
  source .venv/bin/activate
  PYTHONPATH=../../src:../../ros/robotmem_ros python perception_replay.py
  PYTHONPATH=../../src:../../ros/robotmem_ros python perception_replay.py --seed 42

流程:
  1. 录制: MuJoCo FetchPush 50 episodes × 50 steps → JSON
  2. 回放: JSON → PerceptionData → PerceptionBuffer → DB
  3. 验证: received == written, dropped == 0, recall 能检索
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

# ── Mock ROS 2 ──

def _mock_module(name, attrs=None):
    mod = ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _FakeNode:
    def __init__(self, name):
        self._name = name
        self._params = {}
        self._logger = MagicMock()

    def declare_parameter(self, name, default):
        self._params[name] = default

    def get_parameter(self, name):
        val = MagicMock()
        val.value = self._params.get(name, "")
        return val

    def get_logger(self):
        return self._logger

    def create_service(self, *a, **kw):
        pass

    def create_subscription(self, *a, **kw):
        pass

    def create_timer(self, *a, **kw):
        pass

    def create_publisher(self, *a, **kw):
        return MagicMock()

    def destroy_node(self):
        pass


class _FakeMsg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        return getattr(super(), name, "") if name.startswith("_") else ""


class _FakeMemory(_FakeMsg):
    def __init__(self):
        self.id = 0
        self.content = ""
        self.type = ""
        self.perception_type = ""
        self.confidence = 0.0
        self.rrf_score = 0.0
        self.context_json = ""
        self.session_id = ""
        self.created_at = ""


class _FakeNodeStatus(_FakeMsg):
    def __init__(self):
        self.ready = False
        self.db_path = ""
        self.embed_backend = ""
        self.collection = ""
        self.active_memories_count = 0


class _FakePerceptionData(_FakeMsg):
    def __init__(self, **kwargs):
        self.header = None
        self.seq = 0
        self.perception_type = "visual"
        self.description = ""
        self.data = ""
        self.metadata = ""
        self.session_id = ""
        self.collection = ""
        for k, v in kwargs.items():
            setattr(self, k, v)


_mock_module("rclpy")
_mock_module("rclpy.callback_group", {
    "MutuallyExclusiveCallbackGroup": MagicMock,
    "ReentrantCallbackGroup": MagicMock,
})
_mock_module("rclpy.executors", {"MultiThreadedExecutor": MagicMock})
_mock_module("rclpy.node", {"Node": _FakeNode})
_mock_module("rclpy.qos", {
    "DurabilityPolicy": MagicMock(),
    "HistoryPolicy": MagicMock(),
    "QoSProfile": MagicMock,
    "ReliabilityPolicy": MagicMock(),
})
_mock_module("robotmem_msgs", {})
_mock_module("robotmem_msgs.msg", {
    "Memory": _FakeMemory,
    "NodeStatus": _FakeNodeStatus,
    "PerceptionData": _FakePerceptionData,
})
_mock_module("robotmem_msgs.srv", {
    "EndSession": MagicMock, "Forget": MagicMock, "Learn": MagicMock,
    "Recall": MagicMock, "SavePerception": MagicMock,
    "StartSession": MagicMock, "Update": MagicMock,
})

try:
    import gymnasium_robotics  # noqa: F401
    import gymnasium
    import numpy as np
except ImportError:
    print("需要安装: pip install gymnasium-robotics")
    sys.exit(1)

from robotmem.sdk import RobotMemory
from robotmem_ros.node import PerceptionBuffer, RobotMemNode

from policies import HeuristicPolicy

# 路径
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".robotmem-replay")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COLLECTION = "replay_test"


def create_node(db_path, collection):
    """构造 Node（mock rclpy + 真实 SDK，embed=none 加速）"""
    with patch.object(RobotMemNode, "__init__", lambda self: None):
        node = RobotMemNode.__new__(RobotMemNode)

    mem = RobotMemory(db_path=db_path, collection=collection, embed_backend="none")
    perc_mem = RobotMemory(db_path=db_path, collection=collection, embed_backend="none")

    node._name = "robotmem_replay"
    node._logger = MagicMock()
    node._mem = mem
    node._perception_mem = perc_mem
    node._collection = collection
    node._db_path_str = db_path
    node._embed_backend = "none"
    node._ready_pub = MagicMock()
    node._perc_buffer = PerceptionBuffer(perc_mem, batch_size=50, flush_interval=1.0)
    return node, mem, perc_mem


def record_episodes(env, policy, n_episodes):
    """录制阶段: 运行 MuJoCo 仿真，记录每步 observation"""
    recording = {"episodes": []}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_data = {"episode": ep, "observations": [], "success": False, "total_reward": 0.0}

        for step in range(50):
            action = policy.act(obs)
            if isinstance(action, np.ndarray):
                action = action.tolist()

            # 录制当前状态
            episode_data["observations"].append({
                "step": step,
                "grip": obs["observation"][0:3].tolist(),
                "obj": obs["observation"][3:6].tolist(),
                "target": obs["desired_goal"].tolist(),
                "action": action,
                "reward": float(obs.get("reward", -1.0)) if isinstance(obs, dict) and "reward" in obs else -1.0,
            })

            obs, reward, terminated, truncated, info = env.step(action)
            episode_data["total_reward"] += reward
            if terminated or truncated:
                break

        episode_data["success"] = bool(info.get("is_success", False))
        episode_data["total_reward"] = float(episode_data["total_reward"])
        recording["episodes"].append(episode_data)

        if (ep + 1) % 10 == 0:
            successes = sum(1 for e in recording["episodes"] if e["success"])
            print(f"  录制 [{ep+1}/{n_episodes}] 成功率: {successes/(ep+1):.0%}")

    return recording


def replay_perceptions(node, recording, session_id=None):
    """回放阶段: 将录制数据通过 PerceptionBuffer 写入"""
    total_observations = 0
    seq = 0

    for ep_data in recording["episodes"]:
        ep = ep_data["episode"]
        for obs_data in ep_data["observations"]:
            seq += 1
            total_observations += 1

            msg = _FakePerceptionData(
                seq=seq,
                description=f"FetchPush ep{ep} step{obs_data['step']}: "
                            f"grip={obs_data['grip']}, obj={obs_data['obj']}",
                perception_type="proprioceptive",
                data=json.dumps({
                    "grip": obs_data["grip"],
                    "obj": obs_data["obj"],
                    "target": obs_data["target"],
                    "action": obs_data["action"],
                }),
                metadata=json.dumps({
                    "episode": ep,
                    "step": obs_data["step"],
                    "success": ep_data["success"],
                }),
                session_id=session_id or "",
                collection="",
            )
            node._perception_cb(msg)

    # 最终 flush
    node._perc_buffer.flush()
    return total_observations


def verify_storage(node, total_observations):
    """验证阶段: 检查存储完整性"""
    stats = node._perc_buffer.get_stats()

    checks = {
        "received == total": stats["received"] == total_observations,
        "written == received": stats["written"] == stats["received"],
        "dropped == 0": stats["dropped"] == 0,
        "failed == 0": stats["failed"] == 0,
    }

    # recall 验证
    req = _FakeMsg(
        query="FetchPush grip obj",
        n=10,
        min_confidence=0.0,
        session_id="",
        context_filter="",
        spatial_sort="",
        collection="",
    )
    resp = _FakeMsg(success=False, error="", memories=[], total=0, mode="")
    resp = node._recall_cb(req, resp)
    checks["recall 有结果"] = resp.total > 0

    return checks, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="robotmem Perception 回放测试",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--episodes", type=int, default=50, help="录制 episode 数（默认 50）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 清空旧数据
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("robotmem Perception 回放测试")
    print(f"录制 {args.episodes} episodes × 50 steps")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")
    print(f"DB: {DB_PATH}")
    print("=" * 60)

    env = gymnasium.make("FetchPush-v4")
    policy = HeuristicPolicy()
    node, mem, perc_mem = create_node(DB_PATH, COLLECTION)

    t0 = time.time()

    try:
        # 1. 录制
        print("\n--- 阶段 1: 录制 ---")
        recording = record_episodes(env, policy, args.episodes)

        total_obs = sum(len(ep["observations"]) for ep in recording["episodes"])
        successes = sum(1 for ep in recording["episodes"] if ep["success"])
        print(f"  录制完成: {total_obs} 条观测, {successes}/{args.episodes} 成功")

        # 保存录制 JSON
        seed_str = str(args.seed) if args.seed is not None else "random"
        rec_file = os.path.join(RESULTS_DIR, f"perception_recording_{seed_str}.json")
        rec_meta = {
            "metadata": {
                "env": "FetchPush-v4",
                "seed": args.seed,
                "episodes": args.episodes,
                "total_observations": total_obs,
            },
            "episodes": recording["episodes"],
        }
        with open(rec_file, "w") as f:
            json.dump(rec_meta, f, indent=2)
        print(f"  录制保存: {rec_file}")

        # 2. 回放
        print("\n--- 阶段 2: 回放（通过 PerceptionBuffer）---")
        # start_session
        req = _FakeMsg(context='{"task": "perception_replay"}', collection="")
        resp = _FakeMsg(success=False, error="", session_id="", collection="", active_memories_count=0)
        resp = node._start_session_cb(req, resp)
        sid = resp.session_id

        replayed = replay_perceptions(node, recording, session_id=sid)
        print(f"  回放完成: {replayed} 条通过 PerceptionBuffer")

        # 3. 验证
        print("\n--- 阶段 3: 验证 ---")
        checks, stats = verify_storage(node, replayed)

        elapsed = time.time() - t0

        # 输出结果
        result_lines = [
            f"{'=' * 60}",
            "Perception 回放测试结果",
            f"{'=' * 60}",
            f"  录制: {args.episodes} episodes, {total_obs} 观测",
            f"  回放: {replayed} 条通过 PerceptionBuffer",
            "",
            "  PerceptionBuffer stats:",
            f"    received: {stats['received']}",
            f"    written:  {stats['written']}",
            f"    dropped:  {stats['dropped']}",
            f"    failed:   {stats['failed']}",
            "",
            "  验证结果:",
        ]
        all_passed = True
        for check_name, passed in checks.items():
            mark = "PASS" if passed else "FAIL"
            result_lines.append(f"    [{mark}] {check_name}")
            if not passed:
                all_passed = False

        result_lines.extend([
            "",
            f"  总结: {'全部通过' if all_passed else '有失败项'}",
            f"  耗时: {elapsed:.0f}s",
            f"  录制文件: {rec_file}",
        ])
        print("\n".join(result_lines))

        # 保存结果
        result_file = os.path.join(RESULTS_DIR, f"perception_replay_{seed_str}.txt")
        with open(result_file, "w") as f:
            f.write("\n".join(result_lines))
        print(f"\n结果已保存: {result_file}")

        if not all_passed:
            sys.exit(1)

    finally:
        env.close()
        mem.close()
        perc_mem.close()


if __name__ == "__main__":
    main()
