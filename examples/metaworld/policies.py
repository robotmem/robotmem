"""Meta-World 策略模块 — MetaWorldPushPolicy + MetaWorldMemoryPolicy

Meta-World push-v3 环境:
- Action: [dx, dy, dz, grip] ∈ [-1, 1]^4
- Obs: (39,) — hand_pos(3), grip(1), obj_pos(3), ...
- Reward: 连续（距离越近越高）
- Success: info["success"] > 0
- Episode: 150 步（默认 max_path_length=150）

注意: env._target_pos 提供目标位置，但不在 obs 中。
"""

import numpy as np


class MetaWorldPushPolicy:
    """push-v3 启发式策略: 绕到物体背面 → 向目标推

    与 FetchPush HeuristicPolicy 类比，但动作空间不同:
    - FetchPush: 4D 相对位移 (dx, dy, dz, grip)
    - Meta-World: 4D 归一化控制 [-1, 1]^4

    策略分两阶段:
    1. Approach: 从物体相对于目标的"背面"接近
    2. Push: 穿过物体推向目标
    """

    def __init__(self, noise_scale=0.3, approach_offset=0.05, push_speed=8.0):
        self.noise_scale = noise_scale
        self.approach_offset = approach_offset
        self.push_speed = push_speed

    def act(self, obs, target_pos):
        hand = obs[:3]
        obj = obs[4:7]

        # 推方向: obj → target
        push_dir = target_pos - obj
        push_dist = np.linalg.norm(push_dir[:2])
        if push_dist > 0.001:
            push_dir_norm = push_dir / push_dist
        else:
            push_dir_norm = np.array([0.0, 1.0, 0.0])

        dist_to_obj_xy = np.linalg.norm(hand[:2] - obj[:2])

        action = np.zeros(4)

        if dist_to_obj_xy > self.approach_offset:
            # Phase 1: 移到物体背面
            behind = obj.copy()
            behind[:2] -= push_dir_norm[:2] * self.approach_offset
            behind[2] = obj[2] + 0.02  # 略高于物体
            d = behind - hand
            action[:3] = np.clip(d * self.push_speed, -1, 1)
        else:
            # Phase 2: 推向目标
            action[:2] = np.clip(push_dir_norm[:2] * 5.0, -1, 1)
            action[2] = -0.3  # 向下压住

        # 探索噪声
        noise = np.random.randn(3) * self.noise_scale
        action[:3] += noise

        return np.clip(action, -1, 1)


class MetaWorldMemoryPolicy:
    """记忆驱动 push 策略 — 从成功经验学习推送参数

    核心机制 — 空间回忆:
    - Phase B 记录每个 episode 的 (obj_pos, target_pos, approach_offset, push_speed, success)
    - Phase C 按 obj_pos 空间检索最近的成功经验
    - 用成功经验的参数替代默认值
    - 降低噪声（成功参数不需要大量探索）

    与 FetchPush PhaseAwareMemoryPolicy 的对比:
    - FetchPush: 学习推方向 bias → 修正 4D 动作
    - Meta-World: 学习 approach 参数 → 修正绕行策略
    - 相同: 都用 spatial_sort 按物体位置检索
    """

    def __init__(self, base_policy, recalled_memories, memory_weight=0.3):
        self.weight = memory_weight
        self.base_noise = base_policy.noise_scale

        # 从记忆学习参数
        params = self._extract_params(recalled_memories)
        self.base = MetaWorldPushPolicy(
            noise_scale=params["noise_scale"],
            approach_offset=params["approach_offset"],
            push_speed=params["push_speed"],
        )

    def _extract_params(self, memories):
        """从成功记忆中学习最优参数"""
        if not memories:
            return {
                "noise_scale": self.base_noise,
                "approach_offset": 0.05,
                "push_speed": 8.0,
            }

        offsets = []
        speeds = []
        for m in memories:
            params = m.get("params", {})
            off = params.get("approach_offset", {})
            if isinstance(off, dict) and "value" in off:
                offsets.append(off["value"])
            spd = params.get("push_speed", {})
            if isinstance(spd, dict) and "value" in spd:
                speeds.append(spd["value"])

        return {
            # 成功经验的参数 + 降低噪声
            "noise_scale": self.base_noise * (1 - self.weight),
            "approach_offset": float(np.mean(offsets)) if offsets else 0.05,
            "push_speed": float(np.mean(speeds)) if speeds else 8.0,
        }

    def act(self, obs, target_pos):
        return self.base.act(obs, target_pos)
