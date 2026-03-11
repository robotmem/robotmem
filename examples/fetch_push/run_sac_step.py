"""分步运行 SAC 实验 — 训练和评估分离，避免长进程被杀

用法：
  python run_sac_step.py train 42       # 训练 seed 42
  python run_sac_step.py train 153      # 训练 seed 153
  python run_sac_step.py train 264      # 训练 seed 264
  python run_sac_step.py eval           # 评估所有已训练 seed
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# 训练步数和网络配置
TRAIN_STEPS = 50_000
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def train_one(seed: int):
    """训练单个 seed 并保存"""
    import gymnasium as gym
    import gymnasium_robotics
    from stable_baselines3 import SAC, HerReplayBuffer

    gym.register_envs(gymnasium_robotics)
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f"sac_her_seed{seed}_{TRAIN_STEPS}")
    if os.path.exists(model_path + ".zip"):
        print(f"模型已存在: {model_path}.zip，跳过训练", flush=True)
        return

    print(f"训练 SAC+HER seed={seed}, {TRAIN_STEPS} steps...", flush=True)
    env = gym.make("FetchPush-v4", max_episode_steps=50)
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
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
    t0 = time.time()
    model.learn(total_timesteps=TRAIN_STEPS)
    elapsed = time.time() - t0
    env.close()

    model.save(model_path)
    print(f"训练完成: {elapsed:.0f}s, 已保存: {model_path}.zip", flush=True)


def eval_all():
    """评估所有已训练模型"""
    import asyncio
    import shutil
    import numpy as np
    import gymnasium as gym
    import gymnasium_robotics
    from stable_baselines3 import SAC

    from robotmem.config import Config
    from robotmem.db_cog import CogDatabase
    from robotmem.db import floats_to_blob
    from robotmem.embed_onnx import FastEmbedEmbedder
    from robotmem.ops.memories import insert_memory
    from robotmem.search import recall as recall_fn

    gym.register_envs(gymnasium_robotics)

    ROBOTMEM_HOME = os.path.join(os.path.dirname(__file__), ".robotmem_sac")
    os.environ["ROBOTMEM_HOME"] = ROBOTMEM_HOME
    DB_PATH = os.path.join(ROBOTMEM_HOME, "memory.db")
    COLLECTION = "sac_fetchpush"
    EPISODES = 50
    MEMORY_WEIGHT = 0.3

    loop = asyncio.new_event_loop()

    def _run(coro):
        return loop.run_until_complete(coro)

    embedder = FastEmbedEmbedder()

    # 找当前 TRAIN_STEPS 对应的模型
    seeds = []
    suffix = f"_{TRAIN_STEPS}.zip"
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.startswith("sac_her_seed") and f.endswith(suffix):
            seed = int(f.split("seed")[1].split("_")[0])
            seeds.append(seed)

    if not seeds:
        print("没有找到训练好的模型！先运行 train 命令。", flush=True)
        return

    print(f"找到 {len(seeds)} 个模型: seeds={seeds}", flush=True)
    print(f"每 phase {EPISODES} episodes, memory_weight={MEMORY_WEIGHT}", flush=True)

    results = []
    for seed in seeds:
        print(f"\n评估 seed={seed}...", flush=True)

        # 清理 DB
        if os.path.exists(ROBOTMEM_HOME):
            shutil.rmtree(ROBOTMEM_HOME)
        os.makedirs(ROBOTMEM_HOME, exist_ok=True)

        # 加载模型
        model_path = os.path.join(MODELS_DIR, f"sac_her_seed{seed}_{TRAIN_STEPS}")
        env = gym.make("FetchPush-v4", max_episode_steps=50)
        model = SAC.load(model_path, env=env, device="cpu")

        config = Config(db_path=DB_PATH)
        db = CogDatabase(config)

        # Phase A
        sa, da = _run_episodes(model, env, EPISODES)
        print(f"  Phase A: {sa:.0f}% (dist={da:.3f}m)", flush=True)

        # Phase B
        sb, db_ = _run_episodes_with_mem(
            model, env, EPISODES, db, embedder, COLLECTION, MEMORY_WEIGHT,
            _run, write=True, read=False,
        )
        print(f"  Phase B: {sb:.0f}% (dist={db_:.3f}m)", flush=True)

        # Phase C
        sc, dc = _run_episodes_with_mem(
            model, env, EPISODES, db, embedder, COLLECTION, MEMORY_WEIGHT,
            _run, write=True, read=True,
        )
        print(f"  Phase C: {sc:.0f}% (dist={dc:.3f}m)", flush=True)

        env.close()
        db.close()
        results.append({"seed": seed, "a": (sa, da), "b": (sb, db_), "c": (sc, dc)})

    # 汇总
    import numpy as np
    print("\n" + "─" * 50, flush=True)
    for phase in ["a", "b", "c"]:
        rates = [r[phase][0] for r in results]
        dists = [r[phase][1] for r in results]
        label = {"a": "Phase A", "b": "Phase B", "c": "Phase C"}[phase]
        print(f"  {label}: {np.mean(rates):.1f}% ± {np.std(rates):.1f}%  "
              f"dist={np.mean(dists):.3f} ± {np.std(dists):.3f}", flush=True)
    delta = [r["c"][0] - r["a"][0] for r in results]
    print(f"  Delta (C-A): {np.mean(delta):+.1f}% ± {np.std(delta):.1f}%", flush=True)

    # 保存
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, f"sac_her_{TRAIN_STEPS}_results.json")
    json_data = {
        "config": {"train_steps": TRAIN_STEPS, "episodes": EPISODES, "seeds": seeds},
        "results": [{"seed": r["seed"],
                     "a": {"rate": r["a"][0], "dist": r["a"][1]},
                     "b": {"rate": r["b"][0], "dist": r["b"][1]},
                     "c": {"rate": r["c"][0], "dist": r["c"][1]}} for r in results],
        "summary": {
            "a_mean": float(np.mean([r["a"][0] for r in results])),
            "c_mean": float(np.mean([r["c"][0] for r in results])),
            "delta_mean": float(np.mean(delta)),
            "delta_std": float(np.std(delta)),
        },
    }
    with open(result_file, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n结果已保存: {result_file}", flush=True)


def _run_episodes(model, env, n):
    """纯 SAC 评估"""
    import numpy as np
    successes, dists = 0, []
    for _ in range(n):
        obs, _ = env.reset()
        done, trunc = False, False
        while not done and not trunc:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, info = env.step(action)
        if info.get("is_success", False):
            successes += 1
        dists.append(np.linalg.norm(obs["observation"][3:6] - obs["desired_goal"]))
    return successes / n * 100, np.mean(dists)


def _run_episodes_with_mem(model, env, n, db, embedder, collection, w, _run,
                           write=False, read=False):
    """SAC + robotmem 评估"""
    import numpy as np
    from robotmem.db import floats_to_blob
    from robotmem.ops.memories import insert_memory
    from robotmem.search import recall as recall_fn

    successes, dists = 0, []
    for ep in range(n):
        obs, _ = env.reset()
        obs_hist, act_hist = [obs], []

        # recall
        bias = np.zeros(4)
        if read:
            obj = obs["observation"][3:6]
            q = f"FetchPush push near [{obj[0]:.2f}, {obj[1]:.2f}]"
            result = _run(recall_fn(
                query=q, db=db, embedder=embedder, collection=collection, top_k=5,
                context_filter={"task.success": True},
                spatial_sort={"field": "spatial.object_position", "target": obj.tolist()},
            ))
            vels = []
            for m in result.memories:
                p = m.get("params", {})
                if not p:
                    import json as _json
                    ctx = m.get("context", "{}")
                    if isinstance(ctx, str):
                        ctx = _json.loads(ctx)
                    p = ctx.get("params", {})
                if "approach_velocity" in p:
                    vels.append(p["approach_velocity"]["value"])
            if vels:
                avg = np.mean(vels, axis=0)
                bias = np.array([avg[0], avg[1], avg[2], 0.0])

        done, trunc = False, False
        while not done and not trunc:
            action, _ = model.predict(obs, deterministic=True)
            if read and not np.allclose(bias, 0):
                action = np.clip((1 - w) * action + w * bias, -1.0, 1.0)
            obs, _, done, trunc, info = env.step(action)
            obs_hist.append(obs)
            act_hist.append(action)

        success = info.get("is_success", False)
        dist = np.linalg.norm(obs["observation"][3:6] - obs["desired_goal"])
        if success:
            successes += 1
        dists.append(dist)

        # learn
        if write and len(act_hist) >= 2:
            import json as _json
            last_acts = act_hist[-10:]
            vel = [float(x) for x in np.mean(last_acts, axis=0)[:3]]
            last_obs = obs_hist[-1]
            status = "成功" if success else "失败"
            content = (f"SAC episode {ep}: {status}, "
                       f"velocity [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}], "
                       f"dist {dist:.3f}m")
            ctx = {
                "params": {"approach_velocity": {"value": vel, "unit": "m/step"},
                           "final_distance": {"value": float(dist), "unit": "m"}},
                "spatial": {"object_position": [float(x) for x in last_obs["observation"][3:6]],
                            "target_position": [float(x) for x in last_obs["desired_goal"]]},
                "task": {"name": "FetchPush-v4", "success": bool(success), "policy": "SAC+HER"},
            }
            emb = _run(embedder.embed_one(content))
            insert_memory(db.conn, {
                "content": content, "context": _json.dumps(ctx, default=lambda o: float(o) if hasattr(o, 'item') else o),
                "embedding": floats_to_blob(emb), "type": "fact",
                "collection": collection, "source": "sac_experiment",
            }, vec_loaded=db.vec_loaded)

    return successes / n * 100, np.mean(dists)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "train":
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        train_one(seed)
    elif cmd == "eval":
        eval_all()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
