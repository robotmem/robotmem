"""robotmem LeRobot 回调 — 可插入 LeRobot 训练循环的记忆回调

用法 1（独立使用）:
    from lerobot_callback import RobotMemCallback

    cb = RobotMemCallback(collection="my_experiment")
    cb.on_train_begin({"robot": "aloha", "task": "pick_place"})

    for ep in range(100):
        # ... 训练 ...
        cb.on_episode_end({"episode": ep, "reward": 1.0, "success": True})

    cb.on_train_end({"success_rate": 0.85})
    cb.close()

用法 2（LeRobot 集成）:
    见 lerobot_train_demo.py
"""

from __future__ import annotations

import json
import logging
from typing import Any

from robotmem.sdk import RobotMemory

logger = logging.getLogger(__name__)


class RobotMemCallback:
    """LeRobot 训练回调 — 自动记录经验到 robotmem

    设计原则（宪法第 1 条：机器能用）:
    - 每个 episode 结束时自动 learn
    - 可选 save_perception 保存轨迹数据
    - recall_tips 供策略推理使用
    - 所有异常内部捕获，不中断训练（宪法第 4 条：坏了就喊）
    """

    def __init__(
        self,
        db_path: str | None = None,
        collection: str = "lerobot",
        embed_backend: str = "onnx",
    ):
        self.mem = RobotMemory(
            db_path=db_path,
            collection=collection,
            embed_backend=embed_backend,
        )
        self._sid: str | None = None
        self._episode_count = 0

    def on_train_begin(self, config: dict[str, Any] | None = None) -> str:
        """训练开始 — 创建 session，返回 session_id"""
        config = config or {}
        try:
            self._sid = self.mem.start_session(context={
                "robot": config.get("robot", "unknown"),
                "task": config.get("task", "unknown"),
                "policy": config.get("policy", "unknown"),
                "framework": "lerobot",
            })
            self._episode_count = 0
            logger.info("robotmem session 开始: %s", self._sid)
            return self._sid
        except Exception as e:
            logger.warning("robotmem on_train_begin 失败: %s", e)
            return ""

    def on_episode_end(
        self,
        episode_data: dict[str, Any],
        trajectory: list | None = None,
    ) -> dict | None:
        """Episode 结束 — 记录经验

        Args:
            episode_data: 必须包含 episode(int), reward(float)
                         可选: success(bool), steps(int), context(dict)
            trajectory: 可选，轨迹数据（存为 perception）
        """
        self._episode_count += 1
        ep = episode_data.get("episode", self._episode_count)
        reward = episode_data.get("reward", 0.0)
        success = episode_data.get("success", reward > 0)
        steps = episode_data.get("steps", 0)

        # learn 经验
        insight = f"Episode {ep}: {'成功' if success else '失败'}, reward={reward:.2f}"
        if steps:
            insight += f", {steps} steps"

        context = episode_data.get("context", {})
        if "task" not in context:
            context["task"] = {}
        context["task"]["success"] = success
        context["task"]["reward"] = reward
        context["task"]["steps"] = steps

        try:
            result = self.mem.learn(
                insight=insight,
                context=context,
                session_id=self._sid,
            )
        except Exception as e:
            logger.warning("robotmem on_episode_end learn 失败: %s", e)
            result = None

        # 可选：保存轨迹
        if trajectory is not None:
            try:
                self.mem.save_perception(
                    description=f"Episode {ep} 轨迹: {len(trajectory)} 步",
                    perception_type="procedural",
                    data=json.dumps(trajectory[:50]) if len(trajectory) > 50 else json.dumps(trajectory),
                    session_id=self._sid,
                )
            except Exception as e:
                logger.warning("robotmem save_perception 失败: %s", e)

        return result

    def on_train_end(self, metrics: dict[str, Any] | None = None) -> dict | None:
        """训练结束 — 结束 session"""
        metrics = metrics or {}
        if not self._sid:
            return None

        try:
            result = self.mem.end_session(
                session_id=self._sid,
                outcome_score=metrics.get("success_rate"),
            )
            logger.info(
                "robotmem session 结束: %s, %d episodes",
                self._sid, self._episode_count,
            )
            return result
        except Exception as e:
            logger.warning("robotmem on_train_end 失败: %s", e)
            return None

    def recall_tips(
        self,
        query: str,
        n: int = 5,
        context_filter: dict | None = None,
    ) -> list[dict]:
        """策略推理时检索经验"""
        try:
            return self.mem.recall(
                query=query,
                n=n,
                context_filter=context_filter,
            )
        except Exception as e:
            logger.warning("robotmem recall_tips 失败: %s", e)
            return []

    def close(self):
        """释放资源"""
        self.mem.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
