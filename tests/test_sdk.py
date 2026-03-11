"""sdk.py 单元测试 — RobotMemory 实例化接口

测试隔离：embed_backend="none" + db_path=tmp_path（无真实 ONNX/Ollama）
"""

import pytest

from robotmem.sdk import RobotMemory
from robotmem.exceptions import DatabaseError, ValidationError


# ── fixtures ──

@pytest.fixture
def mem(tmp_path):
    """纯 BM25 模式的 RobotMemory 实例"""
    m = RobotMemory(
        db_path=str(tmp_path / "test.db"),
        collection="test",
        embed_backend="none",
    )
    yield m
    m.close()


# ── 构造函数 ──

class TestConstructor:
    """RobotMemory 构造函数"""

    def test_default_init(self, tmp_path):
        """默认初始化 — ONNX 后端（延迟加载，不需要真实模型）"""
        m = RobotMemory(db_path=str(tmp_path / "default.db"))
        assert not m._closed
        assert m._collection == "default"
        m.close()

    def test_none_backend(self, tmp_path):
        """embed_backend="none" — 纯 BM25"""
        m = RobotMemory(
            db_path=str(tmp_path / "none.db"),
            embed_backend="none",
        )
        assert m._embedder is None
        m.close()

    def test_custom_collection(self, tmp_path):
        """自定义 collection"""
        m = RobotMemory(
            db_path=str(tmp_path / "coll.db"),
            collection="robot-arm",
            embed_backend="none",
        )
        assert m._collection == "robot-arm"
        m.close()

    def test_context_manager(self, tmp_path):
        """with 语句自动关闭"""
        with RobotMemory(
            db_path=str(tmp_path / "ctx.db"),
            embed_backend="none",
        ) as m:
            assert not m._closed
        assert m._closed

    def test_repr(self, mem):
        """__repr__ 输出"""
        r = repr(mem)
        assert "collection='test'" in r
        assert "embed=none" in r


# ── learn ──

class TestLearn:
    """RobotMemory.learn()"""

    def test_learn_basic(self, mem):
        """基本 learn — 返回 created + memory_id"""
        result = mem.learn("force 0.8N works for push")
        assert result["status"] == "created"
        assert isinstance(result["memory_id"], int)
        assert result["memory_id"] > 0

    def test_learn_with_context_str(self, mem):
        """learn 带 context 字符串"""
        result = mem.learn(
            "grip pressure 12N is optimal",
            context="robot=UR5e, task=pick",
        )
        assert result["status"] == "created"

    def test_learn_with_context_dict(self, mem):
        """learn 带 context dict — 自动 JSON 序列化"""
        result = mem.learn(
            "approach speed 0.1m/s is safe",
            context={"params": {"speed": 0.1}, "robot": "UR5e"},
        )
        assert result["status"] == "created"

    def test_learn_with_session_id(self, mem):
        """learn 带 session_id"""
        sid = mem.start_session()
        result = mem.learn("push succeeded", session_id=sid)
        assert result["status"] == "created"

    def test_learn_auto_inferred(self, mem):
        """learn 返回 auto_inferred 字段"""
        result = mem.learn("教训：approach too fast causes collision")
        assert "auto_inferred" in result
        inferred = result["auto_inferred"]
        assert "category" in inferred
        assert "confidence" in inferred
        assert "tags" in inferred

    def test_learn_duplicate(self, mem):
        """重复 learn — 返回 duplicate 而非异常"""
        mem.learn("force 0.8N works")
        result = mem.learn("force 0.8N works")
        assert result["status"] == "duplicate"
        assert "method" in result

    def test_learn_empty_insight_raises(self, mem):
        """空 insight — 抛 ValidationError"""
        with pytest.raises(ValidationError):
            mem.learn("")

    def test_learn_whitespace_insight_raises(self, mem):
        """纯空白 insight — 抛 ValidationError"""
        with pytest.raises(ValidationError):
            mem.learn("   ")

    def test_learn_after_close_raises(self, tmp_path):
        """关闭后 learn — 抛 DatabaseError"""
        m = RobotMemory(
            db_path=str(tmp_path / "closed.db"),
            embed_backend="none",
        )
        m.close()
        with pytest.raises(DatabaseError):
            m.learn("should fail")

    def test_learn_source_is_sdk(self, mem):
        """learn 写入的 source 字段应为 'sdk'"""
        result = mem.learn("test source field")
        mid = result["memory_id"]
        row = mem._db.conn.execute(
            "SELECT source FROM memories WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] == "sdk"


# ── recall ──

class TestRecall:
    """RobotMemory.recall()"""

    def test_recall_empty_db(self, mem):
        """空数据库 recall — 返回空列表"""
        results = mem.recall("push technique")
        assert results == []

    def test_recall_basic(self, mem):
        """基本 recall — 返回 list[dict]"""
        mem.learn("force 0.8N works for push task")
        results = mem.recall("push force")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

    def test_recall_returns_list_not_dict(self, mem):
        """recall 返回 list[dict]，不是 RecallResult 或 dict"""
        mem.learn("test recall return type")
        results = mem.recall("test recall")
        assert isinstance(results, list)

    def test_recall_n_parameter(self, mem):
        """recall n 参数限制返回数量"""
        for i in range(5):
            mem.learn(f"experience number {i} about pushing objects")
        results = mem.recall("pushing objects experience", n=2)
        assert len(results) <= 2

    def test_recall_min_confidence(self, mem):
        """recall min_confidence 过滤"""
        mem.learn("low confidence test memory")
        results = mem.recall("low confidence", min_confidence=0.99)
        # 自动分类的 confidence 不太可能 >= 0.99
        assert len(results) == 0

    def test_recall_with_session_id(self, mem):
        """recall 限定 session"""
        sid = mem.start_session()
        mem.learn("session scoped memory", session_id=sid)
        mem.learn("global memory without session")

        results = mem.recall("memory", session_id=sid)
        for m in results:
            assert m.get("session_id") == sid

    def test_recall_with_context_filter(self, mem):
        """recall 结构化过滤"""
        mem.learn(
            "push succeeded with force 0.8",
            context={"task": {"success": True}},
        )
        mem.learn(
            "push failed with force 0.3",
            context={"task": {"success": False}},
        )
        results = mem.recall(
            "push",
            context_filter={"task.success": True},
        )
        # context_filter 对已解析的 context JSON 字段过滤
        # 至少不应崩溃
        assert isinstance(results, list)

    def test_recall_empty_query_raises(self, mem):
        """空 query — SDK 抛 ValidationError"""
        with pytest.raises(ValidationError):
            mem.recall("")

    def test_recall_after_close_raises(self, tmp_path):
        """关闭后 recall — 抛 DatabaseError"""
        m = RobotMemory(
            db_path=str(tmp_path / "closed.db"),
            embed_backend="none",
        )
        m.close()
        with pytest.raises(DatabaseError):
            m.recall("should fail")


# ── session ──

class TestSession:
    """RobotMemory 会话管理"""

    def test_start_session(self, mem):
        """start_session 返回 str"""
        sid = mem.start_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_start_session_with_context(self, mem):
        """start_session 带 context"""
        sid = mem.start_session(context={"robot": "UR5e", "task": "push"})
        assert isinstance(sid, str)

    def test_end_session(self, mem):
        """end_session 返回 dict"""
        sid = mem.start_session()
        result = mem.end_session(session_id=sid)
        assert result["status"] == "ended"
        assert result["session_id"] == sid

    def test_end_session_with_score(self, mem):
        """end_session 带 outcome_score"""
        sid = mem.start_session()
        mem.learn("push ok", session_id=sid)
        result = mem.end_session(session_id=sid, outcome_score=0.85)
        assert result["status"] == "ended"

    def test_session_context_manager(self, mem):
        """session() 上下文管理器"""
        with mem.session(context={"task": "push"}) as sid:
            assert isinstance(sid, str)
            mem.learn("push experience", session_id=sid)
        # session 应已自动结束

    def test_session_context_manager_on_exception(self, mem):
        """session() 异常时也自动关闭"""
        with pytest.raises(ValueError):
            with mem.session() as sid:
                raise ValueError("test exception")
        # session 应已自动结束（不崩溃）


# ── 生命周期 ──

class TestLifecycle:
    """关闭和资源管理"""

    def test_close_idempotent(self, tmp_path):
        """多次 close 不崩溃"""
        m = RobotMemory(
            db_path=str(tmp_path / "idem.db"),
            embed_backend="none",
        )
        m.close()
        m.close()  # 第二次不应报错

    def test_double_close_no_error(self, mem):
        """fixture 的 mem 被 close 两次不崩溃"""
        mem.close()
        # fixture teardown 会再调一次 close


# ── save_perception ──

class TestSavePerception:
    """RobotMemory.save_perception()"""

    def test_basic(self, mem):
        """基本 save_perception — 返回 memory_id"""
        result = mem.save_perception("视觉传感器检测到物体在桌上位置 [1.3, 0.7]")
        assert result["memory_id"] > 0
        assert result["perception_type"] == "visual"

    def test_custom_perception_type(self, mem):
        """自定义 perception_type"""
        result = mem.save_perception(
            "关节力矩数据 [1.2, 3.4, 0.8, 2.1]",
            perception_type="tactile",
        )
        assert result["perception_type"] == "tactile"

    def test_with_data_and_metadata(self, mem):
        """带 data 和 metadata"""
        result = mem.save_perception(
            "视觉观察到物体位于桌面中央",
            data='{"image_shape": [640, 480]}',
            metadata='{"camera": "front", "fps": 30}',
        )
        assert result["memory_id"] > 0

    def test_with_session_id(self, mem):
        """带 session_id"""
        sid = mem.start_session()
        result = mem.save_perception(
            "夹爪接触力 5.2N 在安全范围",
            session_id=sid,
        )
        assert result["memory_id"] > 0

    def test_empty_description_raises(self, mem):
        """空 description — 抛 ValidationError"""
        with pytest.raises(ValidationError):
            mem.save_perception("")

    def test_short_description_raises(self, mem):
        """description 太短 — 抛 ValidationError"""
        with pytest.raises(ValidationError):
            mem.save_perception("abc")

    def test_source_is_sdk(self, mem):
        """save_perception 写入的 source 字段应为 'sdk'"""
        result = mem.save_perception("感知数据来源测试 source field check")
        mid = result["memory_id"]
        row = mem._db.conn.execute(
            "SELECT source FROM memories WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] == "sdk"

    def test_type_is_perception(self, mem):
        """save_perception 写入的 type 字段应为 'perception'"""
        result = mem.save_perception("感知类型字段测试 type check")
        mid = result["memory_id"]
        row = mem._db.conn.execute(
            "SELECT type FROM memories WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] == "perception"

    def test_after_close_raises(self, tmp_path):
        """关闭后 save_perception — 抛 DatabaseError"""
        m = RobotMemory(
            db_path=str(tmp_path / "closed.db"),
            embed_backend="none",
        )
        m.close()
        with pytest.raises(DatabaseError):
            m.save_perception("should fail test")


# ── forget ──

class TestForget:
    """RobotMemory.forget()"""

    def test_basic(self, mem):
        """基本 forget — 返回 forgotten"""
        result = mem.learn("要删除的错误记忆")
        mid = result["memory_id"]
        forget_result = mem.forget(mid, "这是错误的")
        assert forget_result["status"] == "forgotten"
        assert forget_result["memory_id"] == mid

    def test_not_found_raises(self, mem):
        """记忆不存在 — 抛 ValidationError"""
        with pytest.raises(ValidationError):
            mem.forget(99999, "不存在")

    def test_already_deleted_raises(self, mem):
        """已删除的记忆再次 forget — 抛 ValidationError"""
        result = mem.learn("会被删两次的记忆")
        mid = result["memory_id"]
        mem.forget(mid, "第一次删除")
        with pytest.raises(ValidationError):
            mem.forget(mid, "第二次删除")

    def test_empty_reason_raises(self, mem):
        """空 reason — 抛 ValidationError"""
        result = mem.learn("test memory for forget")
        with pytest.raises(ValidationError):
            mem.forget(result["memory_id"], "")

    def test_after_close_raises(self, tmp_path):
        """关闭后 forget — 抛 DatabaseError"""
        m = RobotMemory(
            db_path=str(tmp_path / "closed.db"),
            embed_backend="none",
        )
        m.close()
        with pytest.raises(DatabaseError):
            m.forget(1, "should fail")

    def test_recall_after_forget(self, mem):
        """forget 后 recall 不到该记忆"""
        result = mem.learn("unique memory to be forgotten xyz123")
        mid = result["memory_id"]
        mem.forget(mid, "错误记忆")
        results = mem.recall("unique memory forgotten xyz123")
        found_ids = [m.get("id") for m in results]
        assert mid not in found_ids


# ── update ──

class TestUpdate:
    """RobotMemory.update()"""

    def test_basic(self, mem):
        """基本 update — 返回 updated"""
        result = mem.learn("原始内容 original content")
        mid = result["memory_id"]
        update_result = mem.update(mid, "更新后的内容 updated")
        assert update_result["status"] == "updated"
        assert update_result["memory_id"] == mid

    def test_not_found_raises(self, mem):
        """记忆不存在 — 抛 ValidationError"""
        with pytest.raises(ValidationError):
            mem.update(99999, "新内容 not found")

    def test_empty_content_raises(self, mem):
        """空 new_content — 抛 ValidationError"""
        result = mem.learn("test memory for update")
        with pytest.raises(ValidationError):
            mem.update(result["memory_id"], "")

    def test_already_deleted_raises(self, mem):
        """已删除的记忆无法 update — 抛 ValidationError"""
        result = mem.learn("会被删的记忆 for update test")
        mid = result["memory_id"]
        mem.forget(mid, "删了")
        with pytest.raises(ValidationError):
            mem.update(mid, "新内容 after forget")

    def test_recall_updated_content(self, mem):
        """update 后 recall 能找到新内容"""
        result = mem.learn("old push force 0.3N insufficient")
        mid = result["memory_id"]
        mem.update(mid, "push force 0.8N is optimal")
        results = mem.recall("push force optimal")
        contents = [m.get("content", "") for m in results]
        assert any("0.8N" in c for c in contents)

    def test_after_close_raises(self, tmp_path):
        """关闭后 update — 抛 DatabaseError"""
        m = RobotMemory(
            db_path=str(tmp_path / "closed.db"),
            embed_backend="none",
        )
        m.close()
        with pytest.raises(DatabaseError):
            m.update(1, "should fail new")


# ── batch_learn ──

class TestBatchLearn:
    """RobotMemory.batch_learn()"""

    def test_basic_dict_items(self, mem):
        """dict 元素批量 learn"""
        results = mem.batch_learn([
            {"insight": "batch experience number one"},
            {"insight": "batch experience number two"},
        ])
        assert len(results) == 2
        assert all(r["status"] == "created" for r in results)

    def test_string_items(self, mem):
        """字符串元素批量 learn"""
        results = mem.batch_learn([
            "string batch insight alpha",
            "string batch insight beta",
        ])
        assert len(results) == 2
        assert all(r["status"] == "created" for r in results)

    def test_partial_failure(self, mem):
        """部分失败 — 不影响其他"""
        results = mem.batch_learn([
            {"insight": "valid batch experience"},
            {"insight": ""},  # 空 insight → error
        ])
        assert results[0]["status"] == "created"
        assert results[1]["status"] == "error"

    def test_empty_list(self, mem):
        """空列表 — 返回空"""
        results = mem.batch_learn([])
        assert results == []

    def test_with_context(self, mem):
        """带 context 的批量 learn"""
        results = mem.batch_learn([
            {"insight": "batch with context test", "context": {"robot": "UR5e"}},
        ])
        assert len(results) == 1
        assert results[0]["status"] == "created"

    def test_after_close_raises(self, tmp_path):
        """关闭后 batch_learn — 抛 DatabaseError"""
        m = RobotMemory(
            db_path=str(tmp_path / "closed.db"),
            embed_backend="none",
        )
        m.close()
        with pytest.raises(DatabaseError):
            m.batch_learn(["should fail"])


# ── 集成：learn + recall 往返 ──

class TestLearnRecallRoundtrip:
    """learn → recall 端到端"""

    def test_roundtrip_basic(self, mem):
        """learn 后能 recall 到"""
        mem.learn("robot arm push velocity should be 0.05 m/s for fragile objects")
        results = mem.recall("push velocity fragile")
        assert len(results) > 0
        assert any("0.05" in m.get("content", "") for m in results)

    def test_roundtrip_multiple(self, mem):
        """多条 learn 后 recall"""
        insights = [
            "approach speed 0.1 m/s is safe for pick task",
            "grip force 8N optimal for plastic cups",
            "push force 0.5N works for lightweight boxes",
        ]
        for ins in insights:
            mem.learn(ins)

        results = mem.recall("grip force cups", n=3)
        assert len(results) > 0

    def test_roundtrip_with_session(self, mem):
        """session 内 learn + recall"""
        with mem.session(context={"task": "push"}) as sid:
            mem.learn("push succeeded at position [1.3, 0.7, 0.4]", session_id=sid)
            results = mem.recall("push position", session_id=sid)
            assert len(results) > 0
