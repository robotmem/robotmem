"""ROS 2 Node 集成测试 — mock rclpy + 真实 SDK

和 test_ros_node.py 的区别：
- test_ros_node.py: mock rclpy + mock SDK → 测试 Node 层代码逻辑
- 本文件: mock rclpy + 真实 SDK(:memory:) → 测试全链路数据流

验证：ROS msg → Node callback → SDK → SQLite → recall → ROS response
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ── Mock ROS 2 模块（复用 test_ros_node.py 模式）──

_ros_mocks = {}


def _mock_module(name, attrs=None):
    mod = ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    _ros_mocks[name] = mod
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


# rclpy
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


# Fake ROS msg 类
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


# robotmem_msgs mock
_mock_module("robotmem_msgs", {})
_mock_module("robotmem_msgs.msg", {
    "Memory": _FakeMemory,
    "NodeStatus": _FakeNodeStatus,
    "PerceptionData": _FakePerceptionData,
})
_mock_module("robotmem_msgs.srv", {
    "EndSession": MagicMock,
    "Forget": MagicMock,
    "Learn": MagicMock,
    "Recall": MagicMock,
    "SavePerception": MagicMock,
    "StartSession": MagicMock,
    "Update": MagicMock,
})

# 现在可以 import node.py
from robotmem.sdk import RobotMemory
from robotmem_ros.node import PerceptionBuffer, RobotMemNode


# ── Fixtures ──

@pytest.fixture
def integration_node(tmp_path):
    """RobotMemNode with 真实 SDK（共享文件 DB, embed=none）

    P0 修复：mem 和 perc_mem 共用同一个文件 DB（和真实 Node 一致），
    不用 :memory:（每次连接独立，两个实例不共享）。
    """
    with patch.object(RobotMemNode, "__init__", lambda self: None):
        node = RobotMemNode.__new__(RobotMemNode)

    # 真实 SDK — 共享同一个文件 DB
    db_path = str(tmp_path / "test_ros.db")
    mem = RobotMemory(db_path=db_path, collection="test_ros", embed_backend="none")
    perc_mem = RobotMemory(db_path=db_path, collection="test_ros", embed_backend="none")

    # 手动初始化 Node 属性
    node._name = "robotmem_test"
    node._logger = MagicMock()
    node._mem = mem
    node._perception_mem = perc_mem
    node._collection = "test_ros"
    node._db_path_str = db_path
    node._embed_backend = "none"
    node._ready_pub = MagicMock()
    node._perc_buffer = PerceptionBuffer(perc_mem, batch_size=50)

    yield node

    # P0 修复：先 flush 再关闭（和 destroy_node 顺序一致）
    node._perc_buffer.flush()
    perc_mem.close()
    mem.close()


def _make_request(**kwargs):
    """构造 fake ROS request"""
    return _FakeMsg(**kwargs)


def _make_response():
    """构造 fake ROS response（所有字段默认空）"""
    resp = _FakeMsg()
    resp.success = False
    resp.error = ""
    return resp


def _make_learn_response():
    resp = _make_response()
    resp.memory_id = 0
    resp.status = ""
    resp.auto_inferred_json = ""
    return resp


def _make_recall_response():
    resp = _make_response()
    resp.memories = []
    resp.total = 0
    resp.mode = ""
    return resp


def _make_start_response():
    resp = _make_response()
    resp.session_id = ""
    resp.collection = ""
    resp.active_memories_count = 0
    return resp


def _make_end_response():
    resp = _make_response()
    resp.summary_json = ""
    resp.decayed_count = 0
    resp.consolidated_json = ""
    resp.related_memories = []
    return resp


def _make_perception_response():
    resp = _make_response()
    resp.memory_id = 0
    return resp


def _make_forget_response():
    resp = _make_response()
    return resp


def _make_update_response():
    resp = _make_response()
    resp.old_content = ""
    resp.new_content_out = ""
    return resp


# ══════════════════════════════════════════════════════════════
# Test A: 全链路集成测试
# ══════════════════════════════════════════════════════════════


class TestLearnRecallIntegration:
    """learn 写入 → recall 检索 全链路"""

    def test_learn_then_recall(self, integration_node):
        """learn 写入后 recall 能用 BM25 检索到"""
        node = integration_node

        # learn
        req = _make_request(
            insight="FetchPush: 成功推送，距离 0.01m",
            context=json.dumps({"task": {"success": True}, "params": {"force": 0.8}}),
            session_id="",
            collection="",
        )
        resp = node._learn_cb(req, _make_learn_response())
        assert resp.success is True
        assert resp.memory_id > 0

        # recall
        req = _make_request(
            query="FetchPush 推送",
            n=5,
            min_confidence=0.0,
            session_id="",
            context_filter="",
            spatial_sort="",
            collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        assert resp.total >= 1
        assert any("推送" in m.content or "FetchPush" in m.content for m in resp.memories)

    def test_learn_multiple_recall_context_filter(self, integration_node):
        """learn 多条经验后 context_filter 只返回成功经验"""
        node = integration_node

        # learn 2 条成功 + 1 条失败
        for i, success in enumerate([True, True, False]):
            ctx = {"task": {"success": success, "name": "push"}, "params": {"dist": 0.01 * (i + 1)}}
            req = _make_request(
                insight=f"FetchPush episode {i}: {'成功' if success else '失败'}，距离 {0.01 * (i+1):.2f}m",
                context=json.dumps(ctx),
                session_id="",
                collection="",
            )
            resp = node._learn_cb(req, _make_learn_response())
            assert resp.success is True

        # recall with context_filter
        req = _make_request(
            query="FetchPush episode",
            n=10,
            min_confidence=0.0,
            session_id="",
            context_filter='{"task.success": true}',
            spatial_sort="",
            collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        assert resp.total > 0, "context_filter 应返回至少 1 条成功经验"
        # 只有成功的经验
        for m in resp.memories:
            ctx = json.loads(m.context_json) if m.context_json else {}
            task = ctx.get("task", {})
            assert task.get("success") is True

    def test_learn_with_dict_context(self, integration_node):
        """learn context 为 JSON 字符串，recall 能正确解析"""
        node = integration_node
        ctx = {
            "params": {"approach_velocity": {"value": [0.1, 0.2, 0.3], "type": "vector"}},
            "spatial": {"grip_position": [1.0, 0.5, 0.4]},
            "task": {"success": True, "steps": 30},
        }
        req = _make_request(
            insight="机械臂推送成功: 速度 0.1m/s",
            context=json.dumps(ctx),
            session_id="",
            collection="",
        )
        resp = node._learn_cb(req, _make_learn_response())
        assert resp.success is True

        # recall 并验证 context 完整
        req = _make_request(
            query="机械臂推送", n=1, min_confidence=0.0,
            session_id="", context_filter="", spatial_sort="", collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        assert resp.total >= 1
        recalled_ctx = json.loads(resp.memories[0].context_json)
        assert "params" in recalled_ctx
        assert recalled_ctx["params"]["approach_velocity"]["value"] == [0.1, 0.2, 0.3]


class TestSavePerceptionIntegration:
    """save_perception 通过 Node 回调的完整性"""

    def test_save_perception_via_callback(self, integration_node):
        """save_perception 回调后数据完整存入 DB"""
        node = integration_node
        req = _make_request(
            description="左臂力矩传感器: [1.2, 0.8, 0.5]",
            perception_type="tactile",
            data='[1.2, 0.8, 0.5]',
            metadata='{"sensor": "left_arm", "unit": "Nm"}',
            session_id="",
            collection="",
        )
        resp = node._save_perception_cb(req, _make_perception_response())
        assert resp.success is True
        assert resp.memory_id > 0

    def test_save_perception_default_type(self, integration_node):
        """perception_type 为空时默认 visual"""
        node = integration_node
        req = _make_request(
            description="摄像头画面: 物体在桌面中央",
            perception_type="",
            data="",
            metadata="",
            session_id="",
            collection="",
        )
        resp = node._save_perception_cb(req, _make_perception_response())
        assert resp.success is True


class TestUpdateForgetIntegration:
    """update 和 forget 全链路"""

    def test_update_then_recall(self, integration_node):
        """update 后 recall 返回新内容"""
        node = integration_node

        # learn
        req = _make_request(
            insight="抓取力 0.5N 即可",
            context="", session_id="", collection="",
        )
        resp = node._learn_cb(req, _make_learn_response())
        mem_id = resp.memory_id

        # update
        req = _make_request(
            memory_id=mem_id,
            new_content="抓取力 0.8N 更稳定",
            context="",
        )
        resp = node._update_cb(req, _make_update_response())
        assert resp.success is True
        assert resp.old_content == "抓取力 0.5N 即可"
        assert resp.new_content_out == "抓取力 0.8N 更稳定"

        # recall 验证新内容
        req = _make_request(
            query="抓取力", n=5, min_confidence=0.0,
            session_id="", context_filter="", spatial_sort="", collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        assert resp.total >= 1
        assert any("0.8N" in m.content for m in resp.memories)

    def test_forget_then_recall_empty(self, integration_node):
        """forget 后 recall 不再返回该记忆"""
        node = integration_node

        # learn
        req = _make_request(
            insight="错误经验: 用力过猛导致物体飞出",
            context="", session_id="", collection="",
        )
        resp = node._learn_cb(req, _make_learn_response())
        mem_id = resp.memory_id

        # forget
        req = _make_request(memory_id=mem_id, reason="错误经验")
        resp = node._forget_cb(req, _make_forget_response())
        assert resp.success is True

        # recall 不再返回
        req = _make_request(
            query="用力过猛 飞出", n=5, min_confidence=0.0,
            session_id="", context_filter="", spatial_sort="", collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        for m in resp.memories:
            assert "飞出" not in m.content


class TestSessionIntegration:
    """session 生命周期全链路"""

    def test_start_end_session(self, integration_node):
        """start_session → learn → end_session 全流程"""
        node = integration_node

        # start_session
        req = _make_request(context='{"task": "push"}', collection="")
        resp = node._start_session_cb(req, _make_start_response())
        assert resp.success is True
        assert resp.session_id != ""
        sid = resp.session_id

        # learn with session
        req = _make_request(
            insight="session 内学习: 推送角度 45 度最佳",
            context="", session_id=sid, collection="",
        )
        resp = node._learn_cb(req, _make_learn_response())
        assert resp.success is True

        # end_session
        req = _make_request(
            session_id=sid,
            has_outcome_score=True,
            outcome_score=0.85,
        )
        resp = node._end_session_cb(req, _make_end_response())
        assert resp.success is True
        assert resp.summary_json != ""

    def test_end_session_without_score(self, integration_node):
        """has_outcome_score=False 时 outcome_score 不传"""
        node = integration_node

        req = _make_request(context="", collection="")
        resp = node._start_session_cb(req, _make_start_response())
        sid = resp.session_id

        req = _make_request(
            session_id=sid,
            has_outcome_score=False,
            outcome_score=0.0,
        )
        resp = node._end_session_cb(req, _make_end_response())
        assert resp.success is True


class TestEmptyFieldMapping:
    """ROS msg 空字符串 → SDK None 映射"""

    def test_empty_collection_uses_default(self, integration_node):
        """collection 空字符串使用 Node 默认 collection"""
        node = integration_node

        req = _make_request(
            insight="测试默认 collection",
            context="", session_id="", collection="",
        )
        resp = node._learn_cb(req, _make_learn_response())
        assert resp.success is True

        # recall 从默认 collection 检索
        req = _make_request(
            query="默认 collection", n=5, min_confidence=0.0,
            session_id="", context_filter="", spatial_sort="", collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        assert resp.total >= 1

    def test_custom_collection(self, integration_node):
        """自定义 collection 隔离"""
        node = integration_node

        # learn 到 custom collection
        req = _make_request(
            insight="自定义 collection 的经验",
            context="", session_id="", collection="custom_coll",
        )
        resp = node._learn_cb(req, _make_learn_response())
        assert resp.success is True

        # 从默认 collection recall，不应包含
        req = _make_request(
            query="自定义 collection", n=5, min_confidence=0.0,
            session_id="", context_filter="", spatial_sort="", collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        # 默认 collection 不包含 custom_coll 的记忆
        for m in resp.memories:
            assert "自定义 collection" not in m.content


class TestJsonParsing:
    """recall JSON 字段解析"""

    def test_valid_context_filter(self, integration_node):
        """有效 JSON context_filter 正常解析"""
        node = integration_node

        req = _make_request(
            insight="JSON 测试成功经验",
            context=json.dumps({"task": {"success": True}}),
            session_id="", collection="",
        )
        node._learn_cb(req, _make_learn_response())

        req = _make_request(
            query="JSON 测试", n=5, min_confidence=0.0,
            session_id="",
            context_filter='{"task.success": true}',
            spatial_sort="",
            collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        # 不应报错
        assert resp.error == ""

    def test_invalid_context_filter_json(self, integration_node):
        """无效 JSON context_filter → error 响应"""
        node = integration_node

        req = _make_request(
            query="test", n=5, min_confidence=0.0,
            session_id="",
            context_filter="{invalid json}",
            spatial_sort="",
            collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        assert resp.success is False
        assert "JSON" in resp.error

    def test_invalid_spatial_sort_json(self, integration_node):
        """无效 JSON spatial_sort → error 响应"""
        node = integration_node

        req = _make_request(
            query="test", n=5, min_confidence=0.0,
            session_id="",
            context_filter="",
            spatial_sort="not valid json",
            collection="",
        )
        resp = node._recall_cb(req, _make_recall_response())
        assert resp.success is False
        assert "JSON" in resp.error


class TestPerceptionBufferIntegration:
    """PerceptionBuffer 通过 Node 的真实 SDK 写入"""

    def test_perception_buffer_real_write(self, integration_node):
        """PerceptionBuffer 批量写入真实 SDK"""
        node = integration_node

        # 发送 5 条 perception
        for i in range(5):
            msg = _FakePerceptionData(
                seq=i + 1,
                description=f"观测数据 step {i}",
                perception_type="visual",
                data=json.dumps({"grip": [0.1 * i, 0.2, 0.3]}),
                metadata=json.dumps({"step": i}),
                session_id="",
                collection="",
            )
            node._perception_cb(msg)

        # flush 强制写入
        node._perc_buffer.flush()

        stats = node._perc_buffer.get_stats()
        assert stats["received"] == 5
        assert stats["written"] == 5
        assert stats["failed"] == 0
        assert stats["dropped"] == 0
