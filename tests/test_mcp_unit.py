"""mcp_server.py 单元测试 — 7 个 MCP tool 的薄包装逻辑

所有 MCP tool 已委托给 SDK，测试重点：
1. 参数正确传递给 SDK
2. SDK 异常 → MCP 返回 {"error": ...}
3. MCP 层特有逻辑（JSON 解析、active_memories_count）
"""

import json
import pytest
from unittest.mock import MagicMock

from robotmem.exceptions import DatabaseError, ValidationError
from robotmem.mcp_server import (
    AppContext,
    _resolve_collection,
    learn,
    recall,
    save_perception,
    forget,
    update,
    start_session,
    end_session,
)


# ── 测试辅助 ──

def _make_app_context(*, default_collection="default", vec_loaded=False):
    """构造 AppContext mock"""
    config = MagicMock()
    config.default_collection = default_collection

    db_cog = MagicMock()
    db_cog.vec_loaded = vec_loaded
    db_cog.conn = MagicMock()

    embedder = MagicMock()
    embedder.available = False
    embedder.unavailable_reason = "test"

    sdk = MagicMock()
    sdk._closed = False
    sdk._db = db_cog
    sdk._embedder = embedder
    sdk._collection = default_collection

    return AppContext(
        config=config,
        db_cog=db_cog,
        embedder=embedder,
        sdk=sdk,
        default_collection=default_collection,
    )


def _make_ctx(app):
    """构造 MCP Context mock"""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


# ── _resolve_collection ──


class TestResolveCollection:
    def test_user_value(self):
        app = _make_app_context()
        assert _resolve_collection(app, "my_project") == "my_project"

    def test_empty_string(self):
        app = _make_app_context(default_collection="default")
        assert _resolve_collection(app, "") == "default"

    def test_none(self):
        app = _make_app_context(default_collection="default")
        assert _resolve_collection(app, None) == "default"

    def test_whitespace(self):
        app = _make_app_context(default_collection="default")
        assert _resolve_collection(app, "   ") == "default"

    def test_strip(self):
        app = _make_app_context()
        assert _resolve_collection(app, " trimmed ") == "trimmed"


# ── Tool 1: learn ──


class TestLearn:
    @pytest.mark.asyncio
    async def test_learn_success(self):
        """MCP learn 委托给 SDK — 成功路径"""
        app = _make_app_context()
        app.sdk.learn = MagicMock(return_value={
            "status": "created", "memory_id": 42,
            "auto_inferred": {"category": "observation", "confidence": 0.9, "tags": [], "scope_files": []},
        })
        ctx = _make_ctx(app)
        result = await learn("重要经验", ctx, collection="test")
        assert result["status"] == "created"
        assert result["memory_id"] == 42

    @pytest.mark.asyncio
    async def test_learn_duplicate(self):
        """MCP learn — SDK 返回 duplicate"""
        app = _make_app_context()
        app.sdk.learn = MagicMock(return_value={
            "status": "duplicate", "method": "exact", "existing_id": 1, "similarity": 1.0,
        })
        ctx = _make_ctx(app)
        result = await learn("重复的经验", ctx, collection="test")
        assert result["status"] == "duplicate"

    @pytest.mark.asyncio
    async def test_learn_empty_insight(self):
        """MCP learn — SDK 抛 ValidationError → MCP 返回 error"""
        app = _make_app_context()
        app.sdk.learn = MagicMock(side_effect=ValidationError("insight 不能为空"))
        ctx = _make_ctx(app)
        result = await learn("", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_learn_write_failure(self):
        """MCP learn — SDK 抛 DatabaseError → MCP 返回 error"""
        app = _make_app_context()
        app.sdk.learn = MagicMock(side_effect=DatabaseError("写入失败"))
        ctx = _make_ctx(app)
        result = await learn("经验", ctx, collection="test")
        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_learn_passes_collection(self):
        """MCP learn — collection 参数正确传递给 SDK"""
        app = _make_app_context()
        app.sdk.learn = MagicMock(return_value={"status": "created", "memory_id": 1})
        ctx = _make_ctx(app)
        await learn("test", ctx, collection="custom_coll")
        app.sdk.learn.assert_called_once()
        call_kwargs = app.sdk.learn.call_args
        assert call_kwargs.kwargs.get("collection") == "custom_coll"


# ── Tool 2: recall ──


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_success(self):
        """MCP recall 委托给 SDK — 成功路径"""
        app = _make_app_context()
        app.sdk.recall = MagicMock(return_value=[{"id": 1, "content": "test"}])
        ctx = _make_ctx(app)
        result = await recall("test query", ctx, collection="test")
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_recall_empty_query(self):
        """MCP recall — SDK 抛 ValidationError"""
        app = _make_app_context()
        app.sdk.recall = MagicMock(side_effect=ValidationError("query 不能为空"))
        ctx = _make_ctx(app)
        result = await recall("", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_with_context_filter(self):
        """MCP recall — context_filter JSON 解析 + 传递给 SDK"""
        app = _make_app_context()
        app.sdk.recall = MagicMock(return_value=[])
        ctx = _make_ctx(app)
        result = await recall(
            "test", ctx, collection="test",
            context_filter='{"task.success": true}',
        )
        assert result["total"] == 0
        # 验证 SDK 收到解析后的 dict
        call_kwargs = app.sdk.recall.call_args.kwargs
        assert call_kwargs["context_filter"] == {"task.success": True}

    @pytest.mark.asyncio
    async def test_recall_invalid_context_filter_json(self):
        """MCP recall — JSON 解析在 MCP 层，不调 SDK"""
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await recall("test", ctx, context_filter="not json")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_context_filter_not_dict(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await recall("test", ctx, context_filter="[1,2,3]")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_context_filter_too_many(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        many = {f"key{i}": i for i in range(11)}
        result = await recall("test", ctx, context_filter=json.dumps(many))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_with_spatial_sort(self):
        """MCP recall — spatial_sort JSON 解析 + 传递给 SDK"""
        app = _make_app_context()
        app.sdk.recall = MagicMock(return_value=[])
        ctx = _make_ctx(app)
        ss = json.dumps({"field": "spatial.pos", "target": [1.0, 2.0]})
        result = await recall("test", ctx, collection="test", spatial_sort=ss)
        assert result["total"] == 0
        call_kwargs = app.sdk.recall.call_args.kwargs
        assert call_kwargs["spatial_sort"] == {"field": "spatial.pos", "target": [1.0, 2.0]}

    @pytest.mark.asyncio
    async def test_recall_invalid_spatial_sort(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        result = await recall("test", ctx, spatial_sort='{"bad": "format"}')
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_spatial_sort_target_not_list(self):
        app = _make_app_context()
        ctx = _make_ctx(app)
        ss = json.dumps({"field": "pos", "target": "not_list"})
        result = await recall("test", ctx, spatial_sort=ss)
        assert "error" in result


# ── Tool 3: save_perception ──


class TestSavePerception:
    @pytest.mark.asyncio
    async def test_success(self):
        """MCP save_perception 委托给 SDK — 成功路径"""
        app = _make_app_context()
        app.sdk.save_perception = MagicMock(return_value={
            "memory_id": 10, "perception_type": "visual",
            "collection": "test", "has_embedding": False,
        })
        ctx = _make_ctx(app)
        result = await save_perception("触觉反馈数据", ctx, collection="test")
        assert result["memory_id"] == 10
        assert result["perception_type"] == "visual"

    @pytest.mark.asyncio
    async def test_empty_description(self):
        """MCP save_perception — SDK 抛 ValidationError"""
        app = _make_app_context()
        app.sdk.save_perception = MagicMock(
            side_effect=ValidationError("description 至少 5 个字符"),
        )
        ctx = _make_ctx(app)
        result = await save_perception("", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_custom_perception_type(self):
        """MCP save_perception — perception_type 正确传递"""
        app = _make_app_context()
        app.sdk.save_perception = MagicMock(return_value={
            "memory_id": 1, "perception_type": "tactile",
            "collection": "test", "has_embedding": False,
        })
        ctx = _make_ctx(app)
        result = await save_perception(
            "力矩数据记录", ctx, perception_type="tactile", collection="test",
        )
        assert result["perception_type"] == "tactile"

    @pytest.mark.asyncio
    async def test_write_failure(self):
        """MCP save_perception — SDK 抛 DatabaseError"""
        app = _make_app_context()
        app.sdk.save_perception = MagicMock(
            side_effect=DatabaseError("写入失败"),
        )
        ctx = _make_ctx(app)
        result = await save_perception("test perception data", ctx, collection="test")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_passes_collection(self):
        """MCP save_perception — collection 参数正确传递"""
        app = _make_app_context()
        app.sdk.save_perception = MagicMock(return_value={
            "memory_id": 1, "perception_type": "visual",
            "collection": "my_coll", "has_embedding": False,
        })
        ctx = _make_ctx(app)
        await save_perception("sensor data test", ctx, collection="my_coll")
        call_kwargs = app.sdk.save_perception.call_args.kwargs
        assert call_kwargs["collection"] == "my_coll"


# ── Tool 4: forget ──


class TestForget:
    @pytest.mark.asyncio
    async def test_success(self):
        """MCP forget 委托给 SDK — 成功路径"""
        app = _make_app_context()
        app.sdk.forget = MagicMock(return_value={
            "status": "forgotten", "memory_id": 1,
            "content": "old memory", "reason": "错误记忆",
        })
        ctx = _make_ctx(app)
        result = await forget(1, "错误记忆", ctx)
        assert result["status"] == "forgotten"

    @pytest.mark.asyncio
    async def test_not_found(self):
        """MCP forget — SDK 抛 ValidationError（记忆不存在）"""
        app = _make_app_context()
        app.sdk.forget = MagicMock(
            side_effect=ValidationError("记忆 #999 不存在"),
        )
        ctx = _make_ctx(app)
        result = await forget(999, "test", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_already_deleted(self):
        """MCP forget — SDK 抛 ValidationError（状态不允许）"""
        app = _make_app_context()
        app.sdk.forget = MagicMock(
            side_effect=ValidationError("记忆 #1 状态为 superseded，无法删除"),
        )
        ctx = _make_ctx(app)
        result = await forget(1, "test", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_reason(self):
        """MCP forget — SDK 抛 ValidationError（reason 为空）"""
        app = _make_app_context()
        app.sdk.forget = MagicMock(
            side_effect=ValidationError("reason 不能为空白"),
        )
        ctx = _make_ctx(app)
        result = await forget(1, "", ctx)
        assert "error" in result


# ── Tool 5: update ──


class TestUpdate:
    @pytest.mark.asyncio
    async def test_success(self):
        """MCP update 委托给 SDK — 成功路径"""
        app = _make_app_context()
        app.sdk.update = MagicMock(return_value={
            "status": "updated", "memory_id": 1,
            "old_content": "old content", "new_content": "新内容",
            "auto_inferred": {"category": "observation", "confidence": 0.9},
        })
        ctx = _make_ctx(app)
        result = await update(1, "新内容", ctx)
        assert result["status"] == "updated"
        assert result["old_content"] == "old content"

    @pytest.mark.asyncio
    async def test_not_found(self):
        """MCP update — SDK 抛 ValidationError（记忆不存在）"""
        app = _make_app_context()
        app.sdk.update = MagicMock(
            side_effect=ValidationError("记忆 #999 不存在"),
        )
        ctx = _make_ctx(app)
        result = await update(999, "new", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_not_active(self):
        """MCP update — SDK 抛 ValidationError（状态不允许）"""
        app = _make_app_context()
        app.sdk.update = MagicMock(
            side_effect=ValidationError("记忆 #1 状态为 superseded，无法更新"),
        )
        ctx = _make_ctx(app)
        result = await update(1, "new", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """MCP update — SDK 抛 ValidationError（内容为空）"""
        app = _make_app_context()
        app.sdk.update = MagicMock(
            side_effect=ValidationError("new_content 不能为空白"),
        )
        ctx = _make_ctx(app)
        result = await update(1, "", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_passes_context(self):
        """MCP update — context 参数正确传递给 SDK"""
        app = _make_app_context()
        app.sdk.update = MagicMock(return_value={
            "status": "updated", "memory_id": 1,
            "old_content": "old", "new_content": "new",
            "auto_inferred": {"category": "observation", "confidence": 0.9},
        })
        ctx = _make_ctx(app)
        await update(1, "new content", ctx, context="extra context")
        call_kwargs = app.sdk.update.call_args.kwargs
        assert call_kwargs["context"] == "extra context"


# ── Tool 6: start/end session ──


class TestStartSession:
    @pytest.mark.asyncio
    async def test_success(self):
        """MCP start_session 委托给 SDK — 成功路径"""
        app = _make_app_context()
        app.sdk.start_session = MagicMock(return_value="sess-uuid-123")
        app.db_cog.conn.execute.return_value.fetchone.return_value = (5,)
        ctx = _make_ctx(app)
        result = await start_session(ctx, collection="test")
        assert result["session_id"] == "sess-uuid-123"
        assert result["collection"] == "test"
        assert result["active_memories_count"] == 5

    @pytest.mark.asyncio
    async def test_create_failure(self):
        """MCP start_session — SDK 抛 DatabaseError"""
        app = _make_app_context()
        app.sdk.start_session = MagicMock(
            side_effect=DatabaseError("创建 session 失败"),
        )
        ctx = _make_ctx(app)
        result = await start_session(ctx, collection="test")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_with_context(self):
        """MCP start_session — context 参数正确传递"""
        app = _make_app_context()
        app.sdk.start_session = MagicMock(return_value="sess-uuid-456")
        app.db_cog.conn.execute.return_value.fetchone.return_value = (0,)
        ctx = _make_ctx(app)
        result = await start_session(
            ctx, collection="test",
            context='{"robot_id": "arm-01"}',
        )
        assert result["session_id"] == "sess-uuid-456"
        call_kwargs = app.sdk.start_session.call_args.kwargs
        assert call_kwargs["context"] == '{"robot_id": "arm-01"}'

    @pytest.mark.asyncio
    async def test_passes_collection(self):
        """MCP start_session — collection 参数正确传递"""
        app = _make_app_context()
        app.sdk.start_session = MagicMock(return_value="sid")
        app.db_cog.conn.execute.return_value.fetchone.return_value = (0,)
        ctx = _make_ctx(app)
        await start_session(ctx, collection="custom")
        call_kwargs = app.sdk.start_session.call_args.kwargs
        assert call_kwargs["collection"] == "custom"


class TestEndSession:
    @pytest.mark.asyncio
    async def test_success(self):
        """MCP end_session 委托给 SDK — 成功路径"""
        app = _make_app_context()
        app.sdk.end_session = MagicMock(return_value={
            "status": "ended", "session_id": "sess-1",
            "summary": {"memory_count": 3},
            "decayed_count": 5,
            "consolidated": {"merged_groups": 0, "superseded_count": 0},
            "related_memories": [],
        })
        ctx = _make_ctx(app)
        result = await end_session("sess-1", ctx)
        assert result["status"] == "ended"
        assert result["decayed_count"] == 5

    @pytest.mark.asyncio
    async def test_empty_session_id(self):
        """MCP end_session — SDK 抛 ValidationError"""
        app = _make_app_context()
        app.sdk.end_session = MagicMock(
            side_effect=ValidationError("session_id 不能为空"),
        )
        ctx = _make_ctx(app)
        result = await end_session("", ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_with_outcome_score(self):
        """MCP end_session — outcome_score 参数正确传递"""
        app = _make_app_context()
        app.sdk.end_session = MagicMock(return_value={
            "status": "ended", "session_id": "sess-1",
            "summary": {}, "decayed_count": 0,
            "consolidated": {"merged_groups": 0, "superseded_count": 0},
            "related_memories": [],
        })
        ctx = _make_ctx(app)
        result = await end_session("sess-1", ctx, outcome_score=0.85)
        assert result["status"] == "ended"
        call_kwargs = app.sdk.end_session.call_args.kwargs
        assert call_kwargs["outcome_score"] == 0.85

    @pytest.mark.asyncio
    async def test_database_error(self):
        """MCP end_session — SDK 抛 DatabaseError"""
        app = _make_app_context()
        app.sdk.end_session = MagicMock(
            side_effect=DatabaseError("数据库异常"),
        )
        ctx = _make_ctx(app)
        result = await end_session("sess-1", ctx)
        assert "error" in result


# ── AppContext ──


class TestAppContext:
    def test_dataclass_fields(self):
        app = _make_app_context()
        assert app.default_collection == "default"
        assert app.config is not None
        assert app.db_cog is not None
        assert app.embedder is not None
        assert app.sdk is not None
