"""robotmem MCP Server — 7 个工具

写入: learn, save_perception
读取: recall
修改: forget, update
会话: start_session, end_session
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

from .config import Config, load_config
from .db_cog import CogDatabase
from .embed import Embedder, create_embedder
from .exceptions import DatabaseError, ValidationError
from .resilience import mcp_error_boundary
from .sdk import RobotMemory

logger = logging.getLogger(__name__)

# ── AppContext ──

@dataclass
class AppContext:
    """MCP 服务全局上下文"""
    config: Config
    db_cog: CogDatabase
    embedder: Embedder
    sdk: RobotMemory
    default_collection: str = "default"


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """服务生命周期 — 启动: DB + Embedder，关闭: 释放资源"""
    config = load_config()
    db_cog = CogDatabase(config)
    # 触发 lazy connect
    try:
        _ = db_cog.conn
    except Exception as e:
        logger.error("robotmem 启动失败: 数据库连接异常 — %s", e)
        raise

    embedder = create_embedder(config)

    # 检查 embedding 可用性
    try:
        embed_ok = await embedder.check_availability()
        if embed_ok:
            logger.info(
                "robotmem 启动: %s 后端可用，模型 %s (%dd)",
                config.embed_backend, embedder.model, embedder.dim,
            )
        else:
            logger.warning(
                "robotmem 启动: embedding 不可用 — %s。仅 BM25 搜索可用",
                embedder.unavailable_reason,
            )
    except Exception as e:
        logger.warning("robotmem 启动: embedding 检测异常 — %s", e)

    # 创建 SDK 实例（复用 db/embedder，不创建新连接）
    sdk = RobotMemory._from_components(
        db=db_cog,
        embedder=embedder if embedder.available else None,
        collection=config.default_collection,
    )

    try:
        yield AppContext(
            config=config,
            db_cog=db_cog,
            embedder=embedder,
            sdk=sdk,
            default_collection=config.default_collection,
        )
    finally:
        sdk.close()  # _owns_resources=False，不释放共享资源
        await embedder.close()
        db_cog.close()


mcp = FastMCP("robotmem", lifespan=app_lifespan)


def _get_ctx(ctx: Context) -> AppContext:
    """从 MCP Context 获取 AppContext"""
    return ctx.request_context.lifespan_context


def _resolve_collection(app: AppContext, collection: str | None) -> str:
    """collection 参数解析 — 优先用户传入，否则用默认值"""
    return collection.strip() if collection and collection.strip() else app.default_collection


# ── Tool 1: learn ──

@mcp.tool()
@mcp_error_boundary
async def learn(
    insight: str,
    ctx: Context,
    context: str = "",
    collection: str | None = None,
    session_id: str | None = None,
) -> dict:
    """记录物理经验（declarative memory）

    委托给 SDK.learn()，MCP 层只负责：
    1. 获取 AppContext
    2. 解析 collection
    3. 捕获 SDK 异常 → 转为 {"error": ...}
    """
    app = _get_ctx(ctx)
    coll = _resolve_collection(app, collection)

    try:
        return app.sdk.learn(
            insight=insight,
            context=context,
            session_id=session_id,
            collection=coll,
        )
    except ValidationError as e:
        return {"error": str(e)}
    except DatabaseError as e:
        return {"error": str(e)}


# ── Tool 2: recall ──

@mcp.tool()
@mcp_error_boundary
async def recall(
    query: str,
    ctx: Context,
    collection: str | None = None,
    n: int = 5,
    min_confidence: float = 0.3,
    session_id: str | None = None,
    context_filter: str | None = None,
    spatial_sort: str | None = None,
) -> dict:
    """检索经验 — BM25 + Vec 混合搜索

    传 session_id 时进入 episode 回放模式，返回该 session 全部记忆按时间排序。

    context_filter: JSON 字符串，结构化过滤条件。
        等值: '{"task.success": true}'
        范围: '{"params.final_distance.value": {"$lt": 0.05}}'

    spatial_sort: JSON 字符串，空间近邻排序。
        '{"field": "spatial.object_position", "target": [1.3, 0.7, 0.42]}'
        可选 max_distance 截断: '{"field": "...", "target": [...], "max_distance": 0.1}'

    MCP 层负责 JSON 字符串解析，SDK 接受 dict。
    """
    app = _get_ctx(ctx)
    coll = _resolve_collection(app, collection)

    # MCP 层：JSON 字符串 → dict（SDK 直接接受 dict）
    cf_dict = None
    if context_filter:
        try:
            cf_dict = json.loads(context_filter)
            if not isinstance(cf_dict, dict):
                return {"error": "context_filter 必须为 JSON 对象"}
            if len(cf_dict) > 10:
                return {"error": "context_filter 过滤条件不得超过 10 项"}
        except json.JSONDecodeError as e:
            return {"error": f"context_filter JSON 解析失败: {e}"}

    ss_dict = None
    if spatial_sort:
        try:
            ss_dict = json.loads(spatial_sort)
            if not isinstance(ss_dict, dict):
                return {"error": "spatial_sort 必须为 JSON 对象"}
            if "field" not in ss_dict or "target" not in ss_dict:
                return {"error": "spatial_sort 必须包含 field 和 target 字段"}
            if not isinstance(ss_dict.get("target"), list):
                return {"error": "spatial_sort.target 必须为数组"}
        except json.JSONDecodeError as e:
            return {"error": f"spatial_sort JSON 解析失败: {e}"}

    try:
        memories = app.sdk.recall(
            query=query,
            n=n,
            min_confidence=min_confidence,
            session_id=session_id,
            context_filter=cf_dict,
            spatial_sort=ss_dict,
            collection=coll,
        )
    except ValidationError as e:
        return {"error": str(e)}

    return {
        "memories": memories,
        "total": len(memories),
        "mode": "hybrid" if app.embedder.available else "bm25_only",
    }


# ── Tool 3: save_perception ──

@mcp.tool()
@mcp_error_boundary
async def save_perception(
    description: str,
    ctx: Context,
    perception_type: str = "visual",
    data: str | None = None,
    metadata: str | None = None,
    collection: str | None = None,
    session_id: str | None = None,
) -> dict:
    """保存感知/轨迹/力矩（procedural memory）

    委托给 SDK.save_perception()，MCP 层只负责异常转换。
    """
    app = _get_ctx(ctx)
    coll = _resolve_collection(app, collection)

    try:
        return app.sdk.save_perception(
            description=description,
            perception_type=perception_type,
            data=data,
            metadata=metadata,
            session_id=session_id,
            collection=coll,
        )
    except ValidationError as e:
        return {"error": str(e)}
    except DatabaseError as e:
        return {"error": str(e)}


# ── Tool 4: forget ──

@mcp.tool()
@mcp_error_boundary
async def forget(
    memory_id: int,
    reason: str,
    ctx: Context,
) -> dict:
    """删除错误记忆（软删除）

    委托给 SDK.forget()，MCP 层只负责异常转换。
    """
    app = _get_ctx(ctx)

    try:
        return app.sdk.forget(memory_id=memory_id, reason=reason)
    except ValidationError as e:
        return {"error": str(e)}
    except DatabaseError as e:
        return {"error": str(e)}


# ── Tool 5: update ──

@mcp.tool()
@mcp_error_boundary
async def update(
    memory_id: int,
    new_content: str,
    ctx: Context,
    context: str = "",
) -> dict:
    """修正记忆内容

    委托给 SDK.update()，MCP 层只负责异常转换。
    """
    app = _get_ctx(ctx)

    try:
        return app.sdk.update(
            memory_id=memory_id,
            new_content=new_content,
            context=context,
        )
    except ValidationError as e:
        return {"error": str(e)}
    except DatabaseError as e:
        return {"error": str(e)}


# ── Tool 6: start/end session ──

@mcp.tool()
@mcp_error_boundary
async def start_session(
    ctx: Context,
    collection: str | None = None,
    context: str | None = None,
) -> dict:
    """开始新会话（episode）

    机器人应用推荐的 context 格式::

        start_session(context='{"robot_id": "arm-01", "robot_model": "UR5e",
                                "environment": "kitchen-3F", "task_domain": "pick-and-place"}')

    委托给 SDK.start_session()，MCP 层添加 active_memories_count。
    """
    app = _get_ctx(ctx)
    coll = _resolve_collection(app, collection)

    try:
        sid = app.sdk.start_session(context=context, collection=coll)
    except ValidationError as e:
        return {"error": str(e)}
    except DatabaseError as e:
        return {"error": str(e)}

    # MCP-specific: 统计 active 记忆数
    try:
        active_count = app.db_cog.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE collection=? AND status='active'",
            (coll,),
        ).fetchone()[0]
    except Exception:
        active_count = 0

    logger.info("MCP start_session: session_id=%s, collection=%s", sid, coll)

    return {
        "session_id": sid,
        "collection": coll,
        "active_memories_count": active_count,
    }


@mcp.tool()
@mcp_error_boundary
async def end_session(
    session_id: str,
    ctx: Context,
    outcome_score: float | None = None,
) -> dict:
    """结束会话 — 标记结束 + 时间衰减 + 巩固 + 评分

    委托给 SDK.end_session()，MCP 层只负责异常转换 + 日志。
    """
    app = _get_ctx(ctx)

    try:
        result = app.sdk.end_session(
            session_id=session_id,
            outcome_score=outcome_score,
        )
    except ValidationError as e:
        return {"error": str(e)}
    except DatabaseError as e:
        return {"error": str(e)}

    logger.info(
        "MCP end_session: session_id=%s, memories=%d, decayed=%d, consolidated=%d",
        session_id,
        result.get("summary", {}).get("memory_count", 0),
        result.get("decayed_count", 0),
        result.get("consolidated", {}).get("superseded_count", 0),
    )

    return result


# ── 入口 ──

def main():
    """CLI 入口 — python -m robotmem"""
    mcp.run()


if __name__ == "__main__":
    main()
