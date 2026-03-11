"""SAC + HER + robotmem 实验 — 验证 learned policy + 记忆增强

实验设计：
1. 训练 SAC+HER 到 ~50-60% 成功率（部分训练，非完全收敛）
2. 三阶段协议：A(基线) → B(写入) → C(读写)
3. 3 seeds，每 phase 100 episodes
4. 记忆增强：recall → 提取 approach velocity → 与 SAC 动作混合

Issue: #23 补充实验
"""

import asyncio
import json
import os
import sys
import time
import shutil
import numpy as np

# 确保 robotmem 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC, HerReplayBuffer

# robotmem imports
from robotmem.config import Config
from robotmem.db_cog import CogDatabase
from robotmem.db import floats_to_blob
from robotmem.embed_onnx import FastEmbedEmbedder
from robotmem.ops.memories import insert_memory
from robotmem.search import recall as recall_fn

gym.register_envs(gymnasium_robotics)

# ── 配置 ──
ROBOTMEM_HOME = os.path.join(os.path.dirname(__file__), ".robotmem_sac")
os.environ["ROBOTMEM_HOME"] = ROBOTMEM_HOME
DB_PATH = os.path.join(ROBOTMEM_HOME, "memory.db")
COLLECTION = "sac_fetchpush"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

TRAIN_STEPS = 50_000  # SAC+HER 50k → ~30-50%
EPISODES_PER_PHASE = 50
NUM_SEEDS = 3
MEMORY_WEIGHT = 0.3

# async helper
_loop = asyncio.new_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


def p(*args, **kwargs):
    """print with flush"""
    print(*args, **kwargs, flush=True)


def train_or_load_sac(seed: int, env_id="FetchPush-v4") -> SAC:
    """训练 SAC+HER 或从缓存加载已训练模型"""
    model_path = os.path.join(MODELS_DIR, f"sac_her_seed{seed}_{TRAIN_STEPS}")
    env = gym.make(env_id, max_episode_steps=50)

    if os.path.exists(model_path + ".zip"):
        p(f"    从缓存加载模型: {model_path}.zip")
        model = SAC.load(model_path, env=env, device="cpu")
        env.close()
        return model

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=100_000,
        learning_starts=1000,
        tau=0.05,
        gamma=0.95,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=TRAIN_STEPS)
    env.close()

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(model_path)
    p(f"    模型已保存: {model_path}.zip")
    return model


def extract_episode_stats(obs_history, action_history, success, final_dist):
    """从 episode 历史中提取统计量"""
    if len(action_history) < 2:
        return None

    last_actions = action_history[-10:]
    avg_velocity = np.mean(last_actions, axis=0)[:3]

    last_obs = obs_history[-1]
    grip_pos = last_obs["observation"][:3].tolist()
    obj_pos = last_obs["observation"][3:6].tolist()
    target_pos = last_obs["desired_goal"].tolist()

    return {
        "approach_velocity": avg_velocity.tolist(),
        "grip_pos": grip_pos,
        "obj_pos": obj_pos,
        "target_pos": target_pos,
        "success": bool(success),
        "final_dist": float(final_dist),
    }


def learn_memory(db, embedder, stats, ep_num):
    """将 episode 经验写入 robotmem"""
    if stats is None:
        return

    vel = stats["approach_velocity"]
    status = "成功" if stats["success"] else "失败"
    content = (
        f"SAC policy episode {ep_num}: {status}, "
        f"approach velocity [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}], "
        f"final distance {stats['final_dist']:.3f}m"
    )

    context = {
        "params": {
            "approach_velocity": {"value": vel, "unit": "m/step"},
            "final_distance": {"value": stats["final_dist"], "unit": "m"},
        },
        "spatial": {
            "grip_position": stats["grip_pos"],
            "object_position": stats["obj_pos"],
            "target_position": stats["target_pos"],
        },
        "task": {
            "name": "FetchPush-v4",
            "success": stats["success"],
            "policy": "SAC+HER",
        },
    }

    embedding = _run(embedder.embed_one(content))

    memory = {
        "content": content,
        "context": json.dumps(context),
        "embedding": floats_to_blob(embedding),
        "type": "fact",
        "collection": COLLECTION,
        "source": "sac_experiment",
    }
    insert_memory(db.conn, memory, vec_loaded=db.vec_loaded)


def recall_memories(db, embedder, obs, top_k=5):
    """检索相关记忆"""
    obj_pos = obs["observation"][3:6]

    query = f"FetchPush successful push near [{obj_pos[0]:.2f}, {obj_pos[1]:.2f}]"
    context_filter = {"task.success": True}
    spatial_sort = {
        "field": "spatial.object_position",
        "target": obj_pos.tolist(),
    }

    result = _run(recall_fn(
        query=query,
        db=db,
        embedder=embedder,
        collection=COLLECTION,
        top_k=top_k,
        context_filter=context_filter,
        spatial_sort=spatial_sort,
    ))
    return result.memories


def extract_memory_bias(memories):
    """从记忆中提取动作偏置"""
    if not memories:
        return np.zeros(4)

    velocities = []
    for m in memories:
        params = m.get("params", {})
        if not params:
            ctx = m.get("context", {})
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
            params = ctx.get("params", {})
        if "approach_velocity" in params:
            vel = params["approach_velocity"]["value"]
            velocities.append(vel)

    if not velocities:
        return np.zeros(4)

    avg_vel = np.mean(velocities, axis=0)
    return np.array([avg_vel[0], avg_vel[1], avg_vel[2], 0.0])


def run_episodes(model, env, n_episodes, db=None, embedder=None,
                 read_memory=False, write_memory=False):
    """运行 N 个 episode，返回成功率和平均距离"""
    successes = 0
    distances = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_history = [obs]
        action_history = []

        memory_bias = np.zeros(4)
        if read_memory and db and embedder:
            memories = recall_memories(db, embedder, obs)
            memory_bias = extract_memory_bias(memories)

        done = False
        truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)

            if read_memory and not np.allclose(memory_bias, 0):
                action = np.clip(
                    (1 - MEMORY_WEIGHT) * action + MEMORY_WEIGHT * memory_bias,
                    -1.0, 1.0,
                )

            obs, reward, done, truncated, info = env.step(action)
            obs_history.append(obs)
            action_history.append(action)

        success = info.get("is_success", False)
        obj_pos = obs["observation"][3:6]
        target_pos = obs["desired_goal"]
        dist = np.linalg.norm(obj_pos - target_pos)

        if success:
            successes += 1
        distances.append(dist)

        if write_memory and db and embedder:
            stats = extract_episode_stats(obs_history, action_history, success, dist)
            learn_memory(db, embedder, stats, ep)

    rate = successes / n_episodes * 100
    avg_dist = np.mean(distances)
    return rate, avg_dist


def run_one_seed(seed: int, embedder):
    """单 seed 完整实验"""
    if os.path.exists(ROBOTMEM_HOME):
        shutil.rmtree(ROBOTMEM_HOME)
    os.makedirs(ROBOTMEM_HOME, exist_ok=True)

    p(f"    训练/加载 SAC+HER ({TRAIN_STEPS} steps)...")
    t0 = time.time()
    model = train_or_load_sac(seed)
    p(f"    训练/加载耗时: {time.time() - t0:.0f}s")

    config = Config(db_path=DB_PATH)
    db = CogDatabase(config)

    env = gym.make("FetchPush-v4", max_episode_steps=50)

    rate_a, dist_a = run_episodes(model, env, EPISODES_PER_PHASE)
    p(f"    Phase A (基线): {rate_a:.0f}% (dist={dist_a:.3f}m)")

    rate_b, dist_b = run_episodes(
        model, env, EPISODES_PER_PHASE,
        db=db, embedder=embedder, write_memory=True,
    )
    p(f"    Phase B (写入): {rate_b:.0f}% (dist={dist_b:.3f}m)")

    rate_c, dist_c = run_episodes(
        model, env, EPISODES_PER_PHASE,
        db=db, embedder=embedder, read_memory=True, write_memory=True,
    )
    p(f"    Phase C (利用): {rate_c:.0f}% (dist={dist_c:.3f}m)")

    env.close()
    db.close()

    return {
        "a": (rate_a, dist_a),
        "b": (rate_b, dist_b),
        "c": (rate_c, dist_c),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    p("=" * 60)
    p("SAC+HER + robotmem 实验 — Issue #23 补充")
    p(f"ROBOTMEM_HOME: {ROBOTMEM_HOME}")
    p(f"RESULTS_DIR: {RESULTS_DIR}")
    p("=" * 60)

    embedder = FastEmbedEmbedder()

    results = []
    t_start = time.time()

    for i in range(NUM_SEEDS):
        seed = 42 + i * 111
        p(f"\n  Seed {i+1}/{NUM_SEEDS} (seed={seed})")
        r = run_one_seed(seed, embedder)
        results.append(r)

    # 汇总
    p("\n" + "─" * 50)
    p(f"SAC+HER 实验结果（{NUM_SEEDS} seeds, {EPISODES_PER_PHASE} ep/phase）")
    p("─" * 50)

    summary_lines = []
    for phase in ["a", "b", "c"]:
        rates = [r[phase][0] for r in results]
        dists = [r[phase][1] for r in results]
        label = {"a": "Phase A (基线)", "b": "Phase B (写入)", "c": "Phase C (利用)"}[phase]
        line = (f"  {label}: {np.mean(rates):.1f}% ± {np.std(rates):.1f}%  "
                f"dist={np.mean(dists):.3f} ± {np.std(dists):.3f}")
        p(line)
        summary_lines.append(line)

    delta = [r["c"][0] - r["a"][0] for r in results]
    delta_line = f"  Delta (C-A): {np.mean(delta):+.1f}% ± {np.std(delta):.1f}%"
    p(delta_line)
    summary_lines.append(delta_line)

    elapsed = f"  耗时: {time.time() - t_start:.0f}s"
    p(elapsed)
    summary_lines.append(elapsed)

    # 保存 JSON 结果
    result_file = os.path.join(RESULTS_DIR, "sac_her_results.json")
    json_data = {
        "config": {
            "train_steps": TRAIN_STEPS,
            "episodes_per_phase": EPISODES_PER_PHASE,
            "num_seeds": NUM_SEEDS,
            "memory_weight": MEMORY_WEIGHT,
            "policy": "SAC+HER",
        },
        "seeds": [],
    }
    for i, r in enumerate(results):
        seed = 42 + i * 111
        json_data["seeds"].append({
            "seed": seed,
            "phase_a": {"rate": r["a"][0], "dist": r["a"][1]},
            "phase_b": {"rate": r["b"][0], "dist": r["b"][1]},
            "phase_c": {"rate": r["c"][0], "dist": r["c"][1]},
        })
    json_data["summary"] = {
        "phase_a_mean": np.mean([r["a"][0] for r in results]),
        "phase_a_std": np.std([r["a"][0] for r in results]),
        "phase_c_mean": np.mean([r["c"][0] for r in results]),
        "phase_c_std": np.std([r["c"][0] for r in results]),
        "delta_mean": np.mean(delta),
        "delta_std": np.std(delta),
    }
    with open(result_file, "w") as f:
        json.dump(json_data, f, indent=2)
    p(f"\n  结果已保存: {result_file}")


if __name__ == "__main__":
    main()
