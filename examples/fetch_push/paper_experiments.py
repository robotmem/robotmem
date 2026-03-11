"""论文实验补充 — 多 seed + 跨环境扩量 + ablation
Issue: #23

三个实验：
1. 多 seed FetchPush（5 seeds × 300 episodes）
2. 多 seed 跨环境 FetchSlide（5 seeds × 200 episodes/condition）
3. Ablation（push-only vs all-phase, spatial vs vector vs FTS5）

运行:
  cd examples/fetch_push
  PYTHONPATH=../../src .venv/bin/python3 paper_experiments.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time

import numpy as np

# ── 环境变量（必须在 import robotmem 之前）──
_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_HOME = os.path.join(_DIR, ".robotmem_paper")
os.environ["ROBOTMEM_HOME"] = _EXP_HOME

import gymnasium_robotics  # noqa: F401
import gymnasium

from robotmem.config import load_config
from robotmem.db_cog import CogDatabase
from robotmem.embed import create_embedder
from robotmem.search import recall as do_recall
from robotmem.db import floats_to_blob
from robotmem.ops.memories import insert_memory, consolidate_session
from robotmem.ops.sessions import get_or_create_session, mark_session_ended
from robotmem.auto_classify import classify_category, estimate_confidence

from policies import HeuristicPolicy, PhaseAwareMemoryPolicy, MemoryPolicy, SlidePolicy


# ── 通用 ──

def fresh_db():
    """每个 seed 清空 DB"""
    if os.path.exists(_EXP_HOME):
        shutil.rmtree(_EXP_HOME)
    os.makedirs(_EXP_HOME, exist_ok=True)
    config = load_config()
    db = CogDatabase(config)
    return db, config


def build_context(obs, actions, success, steps, total_reward, task_name="push_to_target", scene="tabletop_push", obj_name="cube"):
    recent = actions[-10:] if len(actions) >= 10 else actions
    avg = np.mean(recent, axis=0) if recent else np.zeros(4)
    return {
        "params": {
            "approach_velocity": {"value": avg[0:3].tolist(), "type": "vector"},
            "grip_force": {"value": float(avg[3]), "type": "scalar"},
            "final_distance": {"value": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])), "unit": "m"},
        },
        "spatial": {
            "frame": "world",
            "grip_position": obs["observation"][0:3].tolist(),
            "object_position": obs["observation"][3:6].tolist(),
            "target_position": obs["desired_goal"].tolist(),
            "scene_tag": scene,
        },
        "task": {"name": task_name, "object": obj_name, "success": bool(success), "steps": steps, "total_reward": float(total_reward)},
    }


async def run_ep(env, policy, phase, ep, db, embedder, collection, memory_weight=0.3, recall_n=5,
                 use_spatial=True, use_vector=True, use_fts=True,
                 task_name="push_to_target", scene="tabletop_push", obj_name="cube"):
    """通用 episode runner"""
    ext_id = f"{collection}_{phase}_{ep:04d}"
    if phase in ("B", "C"):
        get_or_create_session(db.conn, ext_id, collection)

    obs, _ = env.reset()
    recalled = []

    if phase == "C":
        obj_pos = obs["observation"][3:6].tolist()
        target_pos = obs["desired_goal"].tolist()
        q = f"push [{obj_pos[0]:.2f},{obj_pos[1]:.2f}] to [{target_pos[0]:.2f},{target_pos[1]:.2f}]"

        spatial_sort = {"field": "spatial.object_position", "target": obj_pos} if use_spatial else None
        result = await do_recall(
            q, db, embedder, collection=collection, top_k=recall_n,
            context_filter={"task.success": True},
            spatial_sort=spatial_sort,
        )
        recalled = result.memories

    if recalled:
        active = PhaseAwareMemoryPolicy(policy, recalled, memory_weight)
    else:
        active = policy

    actions = []
    total_reward = 0.0
    for _ in range(50):
        action = active.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        actions.append(action.copy())
        total_reward += reward
        if term or trunc:
            break

    success = info.get("is_success", False)
    dist = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

    if phase in ("B", "C"):
        ctx = build_context(obs, actions, success, len(actions), total_reward, task_name, scene, obj_name)
        content = f"{task_name}: {'成功' if success else '失败'}, {dist:.3f}m, {len(actions)}步"
        emb = await embedder.embed_one(content)
        insert_memory(db.conn, {
            "content": content, "context": json.dumps(ctx), "collection": collection,
            "session_id": ext_id, "type": "fact",
            "category": classify_category(content), "confidence": estimate_confidence(content),
            "embedding": floats_to_blob(emb),
        }, vec_loaded=db.vec_loaded)
        mark_session_ended(db.conn, ext_id)
        try:
            consolidate_session(db.conn, ext_id, collection)
        except Exception:
            pass

    return success, dist


async def run_phase(env, policy, phase, n_episodes, db, embedder, collection, **kw):
    successes = 0
    dists = []
    for ep in range(n_episodes):
        ok, d = await run_ep(env, policy, phase, ep, db, embedder, collection, **kw)
        successes += int(ok)
        dists.append(d)
    return successes / n_episodes, float(np.mean(dists))


# ═══════════════════════════════════════
# 实验 1: 多 seed FetchPush
# ═══════════════════════════════════════

async def exp1_multi_seed_push(n_seeds=5, ep_per_phase=100):
    print("\n" + "=" * 60)
    print(f"实验 1: 多 seed FetchPush ({n_seeds} seeds × {ep_per_phase*3} episodes)")
    print("=" * 60)

    embedder = create_embedder(load_config())
    if not await embedder.check_availability():
        print(f"Embedder 不可用: {embedder.unavailable_reason}")
        return None
    print(f"Embedder: {embedder.model}")

    results = {"A": [], "B": [], "C": [], "A_dist": [], "B_dist": [], "C_dist": []}
    t0 = time.time()

    for seed in range(n_seeds):
        db, _ = fresh_db()
        col = f"push_seed{seed}"
        env = gymnasium.make("FetchPush-v4")
        env.reset(seed=seed * 42)
        np.random.seed(seed * 42)

        print(f"\n  Seed {seed+1}/{n_seeds}")
        for phase, label in [("A", "基线"), ("B", "写入"), ("C", "利用")]:
            rate, dist = await run_phase(env, HeuristicPolicy(), phase, ep_per_phase, db, embedder, col)
            results[phase].append(rate)
            results[f"{phase}_dist"].append(dist)
            print(f"    Phase {phase} ({label}): {rate:.0%} (dist={dist:.3f}m)")

        env.close()
        db.close()

    elapsed = time.time() - t0
    await embedder.close()

    # 统计
    print(f"\n{'─'*40}")
    print(f"实验 1 结果（{n_seeds} seeds, {ep_per_phase} ep/phase）")
    print(f"{'─'*40}")
    for phase in ["A", "B", "C"]:
        rates = results[phase]
        dists = results[f"{phase}_dist"]
        print(f"  Phase {phase}: {np.mean(rates):.1%} ± {np.std(rates):.1%}  dist={np.mean(dists):.3f} ± {np.std(dists):.3f}")
    delta = np.array(results["C"]) - np.array(results["A"])
    print(f"  Delta (C-A): {np.mean(delta):+.1%} ± {np.std(delta):.1%}")
    print(f"  耗时: {elapsed:.0f}s")

    return results


# ═══════════════════════════════════════
# 实验 2: 多 seed 跨环境 FetchSlide
# ═══════════════════════════════════════

async def exp2_multi_seed_slide(n_seeds=5, ep_per_cond=200):
    print("\n" + "=" * 60)
    print(f"实验 2: 多 seed FetchSlide ({n_seeds} seeds × {ep_per_cond} ep/condition)")
    print("=" * 60)

    embedder = create_embedder(load_config())
    if not await embedder.check_availability():
        return None

    results = {"baseline": [], "cross": [], "same": []}
    t0 = time.time()

    for seed in range(n_seeds):
        db, _ = fresh_db()
        push_col = f"push_for_slide_s{seed}"
        slide_col = f"slide_s{seed}"

        # 先生成 Push 记忆（100 episodes）
        push_env = gymnasium.make("FetchPush-v4")
        push_env.reset(seed=seed * 42)
        np.random.seed(seed * 42)
        print(f"\n  Seed {seed+1}/{n_seeds}: 生成 Push 记忆...")
        await run_phase(push_env, HeuristicPolicy(), "B", 100, db, embedder, push_col)
        push_env.close()

        slide_env = gymnasium.make("FetchSlide-v4")
        slide_env.reset(seed=seed * 42)
        np.random.seed(seed * 42)

        # Phase A: 基线
        print(f"    Slide 基线...")
        rate_a, _ = await run_phase(slide_env, SlidePolicy(), "A", ep_per_cond, db, embedder, slide_col,
                                     task_name="slide_to_target", scene="tabletop_slide", obj_name="puck")
        results["baseline"].append(rate_a)
        print(f"    基线: {rate_a:.0%}")

        # Phase B: 跨环境（用 Push 记忆）
        print(f"    Slide + Push 记忆...")
        rate_b, _ = await run_phase(slide_env, SlidePolicy(), "C", ep_per_cond, db, embedder, push_col,
                                     task_name="slide_to_target", scene="tabletop_slide", obj_name="puck")
        results["cross"].append(rate_b)
        print(f"    跨环境: {rate_b:.0%}")

        # 写入 Slide 记忆
        print(f"    写入 Slide 记忆...")
        await run_phase(slide_env, SlidePolicy(), "B", ep_per_cond, db, embedder, slide_col,
                         task_name="slide_to_target", scene="tabletop_slide", obj_name="puck")

        # Phase C: 同环境
        print(f"    Slide + Slide 记忆...")
        rate_c, _ = await run_phase(slide_env, SlidePolicy(), "C", ep_per_cond, db, embedder, slide_col,
                                     task_name="slide_to_target", scene="tabletop_slide", obj_name="puck")
        results["same"].append(rate_c)
        print(f"    同环境: {rate_c:.0%}")

        slide_env.close()
        db.close()

    elapsed = time.time() - t0
    await embedder.close()

    print(f"\n{'─'*40}")
    print(f"实验 2 结果（{n_seeds} seeds, {ep_per_cond} ep/condition）")
    print(f"{'─'*40}")
    for k, label in [("baseline", "基线"), ("cross", "跨环境"), ("same", "同环境")]:
        rates = results[k]
        print(f"  {label}: {np.mean(rates):.1%} ± {np.std(rates):.1%}")
    delta = np.array(results["cross"]) - np.array(results["baseline"])
    print(f"  跨环境 Delta: {np.mean(delta):+.1%} ± {np.std(delta):.1%}")
    print(f"  耗时: {elapsed:.0f}s")

    return results


# ═══════════════════════════════════════
# 实验 3: Ablation
# ═══════════════════════════════════════

async def exp3_ablation(n_seeds=3, ep_per_phase=100):
    print("\n" + "=" * 60)
    print(f"实验 3: Ablation ({n_seeds} seeds × {ep_per_phase} ep/phase)")
    print("=" * 60)

    embedder = create_embedder(load_config())
    if not await embedder.check_availability():
        return None

    # Ablation 1: push-only vs all-phase
    print("\n  --- Ablation: push-only vs all-phase ---")
    push_only_rates = []
    all_phase_rates = []
    baseline_rates = []

    for seed in range(n_seeds):
        db, _ = fresh_db()
        col = f"abl_phase_s{seed}"
        env = gymnasium.make("FetchPush-v4")
        env.reset(seed=seed * 42)
        np.random.seed(seed * 42)

        # 基线
        rate_base, _ = await run_phase(env, HeuristicPolicy(), "A", ep_per_phase, db, embedder, col)
        baseline_rates.append(rate_base)

        # 写入记忆
        await run_phase(env, HeuristicPolicy(), "B", ep_per_phase, db, embedder, col)

        # push-only（PhaseAwareMemoryPolicy）
        rate_push, _ = await run_phase(env, HeuristicPolicy(), "C", ep_per_phase, db, embedder, col)
        push_only_rates.append(rate_push)

        env.close()
        db.close()

        # all-phase（MemoryPolicy — 全阶段施加 bias）
        db, _ = fresh_db()
        col2 = f"abl_all_s{seed}"
        env = gymnasium.make("FetchPush-v4")
        env.reset(seed=seed * 42)
        np.random.seed(seed * 42)

        await run_phase(env, HeuristicPolicy(), "A", ep_per_phase, db, embedder, col2)
        await run_phase(env, HeuristicPolicy(), "B", ep_per_phase, db, embedder, col2)

        # all-phase: 手动替换策略为 MemoryPolicy
        successes = 0
        for ep in range(ep_per_phase):
            ext_id = f"{col2}_C_{ep:04d}"
            get_or_create_session(db.conn, ext_id, col2)
            obs, _ = env.reset()
            obj_pos = obs["observation"][3:6].tolist()
            q = f"push [{obj_pos[0]:.2f},{obj_pos[1]:.2f}]"
            result = await do_recall(q, db, embedder, collection=col2, top_k=5,
                                     context_filter={"task.success": True},
                                     spatial_sort={"field": "spatial.object_position", "target": obj_pos})
            recalled = result.memories
            base = HeuristicPolicy()
            active = MemoryPolicy(base, recalled, 0.3) if recalled else base

            actions = []
            for _ in range(50):
                action = active.act(obs)
                obs, reward, term, trunc, info = env.step(action)
                actions.append(action.copy())
                if term or trunc:
                    break
            successes += int(info.get("is_success", False))

            ctx = build_context(obs, actions, info.get("is_success", False), len(actions), 0.0)
            content = f"push: {info.get('is_success', False)}"
            emb = await embedder.embed_one(content)
            insert_memory(db.conn, {
                "content": content, "context": json.dumps(ctx), "collection": col2,
                "session_id": ext_id, "type": "fact",
                "category": "observation", "confidence": 0.8,
                "embedding": floats_to_blob(emb),
            }, vec_loaded=db.vec_loaded)
            mark_session_ended(db.conn, ext_id)

        all_phase_rates.append(successes / ep_per_phase)
        env.close()
        db.close()
        print(f"    Seed {seed+1}: base={rate_base:.0%} push-only={rate_push:.0%} all-phase={successes/ep_per_phase:.0%}")

    print(f"\n  Phase Ablation:")
    print(f"    Baseline:   {np.mean(baseline_rates):.1%} ± {np.std(baseline_rates):.1%}")
    print(f"    Push-only:  {np.mean(push_only_rates):.1%} ± {np.std(push_only_rates):.1%}")
    print(f"    All-phase:  {np.mean(all_phase_rates):.1%} ± {np.std(all_phase_rates):.1%}")

    await embedder.close()

    return {
        "baseline": baseline_rates,
        "push_only": push_only_rates,
        "all_phase": all_phase_rates,
    }


# ═══════════════════════════════════════
# 主入口
# ═══════════════════════════════════════

async def main():
    print("=" * 60)
    print("robotmem 论文实验补充 — Issue #23")
    print(f"ROBOTMEM_HOME: {_EXP_HOME}")
    print("=" * 60)

    # 实验 1: 多 seed FetchPush
    r1 = await exp1_multi_seed_push(n_seeds=5, ep_per_phase=100)

    # 实验 2: 多 seed 跨环境
    r2 = await exp2_multi_seed_slide(n_seeds=5, ep_per_cond=100)

    # 实验 3: Ablation
    r3 = await exp3_ablation(n_seeds=3, ep_per_phase=100)

    # 汇总
    print("\n" + "=" * 60)
    print("全部实验完成")
    print("=" * 60)

    if r1:
        print(f"\n实验 1 (FetchPush 5 seeds):")
        for p in ["A", "B", "C"]:
            print(f"  Phase {p}: {np.mean(r1[p]):.1%} ± {np.std(r1[p]):.1%}")

    if r2:
        print(f"\n实验 2 (FetchSlide 5 seeds):")
        for k, l in [("baseline", "基线"), ("cross", "跨环境"), ("same", "同环境")]:
            print(f"  {l}: {np.mean(r2[k]):.1%} ± {np.std(r2[k]):.1%}")

    if r3:
        print(f"\n实验 3 (Ablation 3 seeds):")
        for k, l in [("baseline", "基线"), ("push_only", "Push-only"), ("all_phase", "All-phase")]:
            print(f"  {l}: {np.mean(r3[k]):.1%} ± {np.std(r3[k]):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
