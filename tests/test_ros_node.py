"""ROS 2 Node 单元测试 — mock rclpy + mock SDK

Mock 策略（对标 test_mcp_unit.py）：
- rclpy / robotmem_msgs 在 import 前 mock（避免 ROS 依赖）
- RobotMemory SDK 用 MagicMock
- 直接调用 Service 回调 / PerceptionBuffer 测试业务逻辑
"""

from __future__ import annotations

import json
import sys
import threading
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ── Mock ROS 2 模块（必须在 import node.py 之前）──

_ros_mocks = {}


def _mock_module(name, attrs=None):
    """创建 mock 模块并注册到 sys.modules"""
    mod = ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    _ros_mocks[name] = mod
    return mod


# rclpy 及子模块
_mock_module("rclpy")
_mock_module("rclpy.callback_group", {
    "MutuallyExclusiveCallbackGroup": MagicMock,
    "ReentrantCallbackGroup": MagicMock,
})
_mock_module("rclpy.executors", {"MultiThreadedExecutor": MagicMock})

# rclpy.node — 需要一个可继承的 Node 类
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

_mock_module("rclpy.node", {"Node": _FakeNode})
_mock_module("rclpy.qos", {
    "DurabilityPolicy": MagicMock(),
    "HistoryPolicy": MagicMock(),
    "QoSProfile": MagicMock,
    "ReliabilityPolicy": MagicMock(),
})

# robotmem_msgs — 简单 mock msg/srv 类
class _FakeMsg:
    """通用 fake ROS msg，支持任意属性赋值"""
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

# 现在可以安全 import node.py
from robotmem.exceptions import DatabaseError, ValidationError
from robotmem_ros.node import (
    PerceptionBuffer,
    RobotMemNode,
    _dict_to_memory_msg,
    ros_error_boundary,
)


# ── Fixtures ──

@pytest.fixture
def mock_sdk():
    """Mock RobotMemory SDK（对标 test_mcp_unit.py 模式）"""
    sdk = MagicMock()
    sdk._closed = False
    sdk.learn.return_value = {
        "status": "created", "memory_id": 42,
        "auto_inferred": {"type": "fact", "tags": ["test"]},
    }
    sdk.recall.return_value = [
        {"id": 1, "content": "test memory", "type": "fact",
         "confidence": 0.9, "_rrf_score": 1.5, "context": '{"k":"v"}',
         "session_id": "s1", "created_at": "2026-03-11"},
    ]
    sdk.save_perception.return_value = {"memory_id": 99}
    sdk.forget.return_value = {"status": "forgotten"}
    sdk.update.return_value = {"old_content": "old", "new_content": "new"}
    sdk.start_session.return_value = "session-abc-123"
    sdk.end_session.return_value = {
        "summary": {"total_actions": 5},
        "decayed_count": 2,
        "consolidated": {"merged_groups": 1, "superseded_count": 3},
        "related_memories": [],
    }
    sdk.close.return_value = None
    # mock _db.conn for _count_active_memories
    sdk._db.conn.execute.return_value.fetchone.return_value = (10,)
    return sdk


@pytest.fixture
def mock_perception_sdk():
    """Perception 专用 mock SDK"""
    sdk = MagicMock()
    sdk.save_perception.return_value = {"memory_id": 100}
    sdk.close.return_value = None
    return sdk


@pytest.fixture
def perc_buffer(mock_perception_sdk):
    """PerceptionBuffer with mock SDK, batch_size=3"""
    return PerceptionBuffer(mock_perception_sdk, batch_size=3, flush_interval=1.0)


@pytest.fixture
def node(mock_sdk, mock_perception_sdk):
    """RobotMemNode with mock SDK（绕过真实初始化）"""
    with patch.object(RobotMemNode, "__init__", lambda self: None):
        n = RobotMemNode.__new__(RobotMemNode)
    # 手动设置 Node 需要的属性
    n._name = "robotmem_test"
    n._logger = MagicMock()
    n._mem = mock_sdk
    n._perception_mem = mock_perception_sdk
    n._collection = "test"
    n._db_path_str = "/tmp/test.db"
    n._embed_backend = "none"
    n._ready_pub = MagicMock()
    n._perc_buffer = PerceptionBuffer(mock_perception_sdk, batch_size=50)
    return n


# ══════════════════════════════════════════════════════════════
# 1. _dict_to_memory_msg
# ══════════════════════════════════════════════════════════════

class TestDictToMemoryMsg:
    """SDK dict → Memory.msg 字段映射"""

    def test_full_fields(self):
        d = {
            "id": 42, "content": "hello", "type": "fact",
            "perception_type": "visual", "confidence": 0.95,
            "_rrf_score": 1.23, "context": '{"k":"v"}',
            "session_id": "s1", "created_at": "2026-03-11",
        }
        m = _dict_to_memory_msg(d)
        assert m.id == 42
        assert m.content == "hello"
        assert m.type == "fact"
        assert m.perception_type == "visual"
        assert m.confidence == 0.95
        assert m.rrf_score == 1.23  # _rrf_score → rrf_score
        assert m.context_json == '{"k":"v"}'  # context → context_json
        assert m.session_id == "s1"
        assert m.created_at == "2026-03-11"

    def test_missing_fields_defaults(self):
        m = _dict_to_memory_msg({})
        assert m.id == 0
        assert m.content == ""
        assert m.type == ""
        assert m.perception_type == ""
        assert m.confidence == 0.0
        assert m.rrf_score == 0.0
        assert m.context_json == ""
        assert m.session_id == ""
        assert m.created_at == ""

    def test_none_values_become_empty_string(self):
        d = {"perception_type": None, "context": None, "session_id": None, "created_at": None}
        m = _dict_to_memory_msg(d)
        assert m.perception_type == ""
        assert m.context_json == ""
        assert m.session_id == ""
        assert m.created_at == ""


# ══════════════════════════════════════════════════════════════
# 2. ros_error_boundary
# ══════════════════════════════════════════════════════════════

class TestRosErrorBoundary:
    """三层错误边界"""

    def _make_self(self):
        """创建 fake self（模拟 Node）"""
        obj = MagicMock()
        obj.get_logger.return_value = MagicMock()
        obj._publish_ready = MagicMock()
        return obj

    def test_normal_passthrough(self):
        """正常返回透传"""
        self_obj = self._make_self()
        @ros_error_boundary
        def handler(self, req, resp):
            resp.success = True
            resp.data = "ok"
            return resp
        resp = MagicMock()
        result = handler(self_obj, MagicMock(), resp)
        assert result.success is True

    def test_validation_error_l1(self):
        """ValidationError → L1 warning"""
        self_obj = self._make_self()
        @ros_error_boundary
        def handler(self, req, resp):
            raise ValidationError("bad param")
        resp = MagicMock()
        result = handler(self_obj, MagicMock(), resp)
        assert result.success is False
        assert "参数校验失败" in result.error
        self_obj.get_logger().warning.assert_called_once()

    def test_database_error_l2(self):
        """DatabaseError → L2 error + ready=false"""
        self_obj = self._make_self()
        @ros_error_boundary
        def handler(self, req, resp):
            raise DatabaseError("db gone")
        resp = MagicMock()
        result = handler(self_obj, MagicMock(), resp)
        assert result.success is False
        assert "数据库错误" in result.error
        self_obj.get_logger().error.assert_called_once()
        self_obj._publish_ready.assert_called_once_with(False)

    def test_unexpected_error_l3(self):
        """未知异常 → L3 error"""
        self_obj = self._make_self()
        @ros_error_boundary
        def handler(self, req, resp):
            raise RuntimeError("boom")
        resp = MagicMock()
        result = handler(self_obj, MagicMock(), resp)
        assert result.success is False
        assert "内部错误" in result.error
        self_obj.get_logger().error.assert_called_once()


# ══════════════════════════════════════════════════════════════
# 3. PerceptionBuffer
# ══════════════════════════════════════════════════════════════

class TestPerceptionBuffer:
    """批量写入 + seq 检测 + 线程安全"""

    def _make_msg(self, seq=0, desc="test", ptype="visual"):
        return _FakePerceptionData(seq=seq, description=desc, perception_type=ptype)

    def test_add_below_batch_size_no_flush(self, perc_buffer, mock_perception_sdk):
        """未达 batch_size 不写 DB"""
        perc_buffer.add(self._make_msg(seq=1))
        perc_buffer.add(self._make_msg(seq=2))
        mock_perception_sdk.save_perception.assert_not_called()

    def test_add_reaches_batch_size_triggers_flush(self, perc_buffer, mock_perception_sdk):
        """达到 batch_size 自动 flush"""
        for i in range(3):  # batch_size=3
            perc_buffer.add(self._make_msg(seq=i + 1))
        assert mock_perception_sdk.save_perception.call_count == 3

    def test_manual_flush(self, perc_buffer, mock_perception_sdk):
        """手动 flush 清空 buffer"""
        perc_buffer.add(self._make_msg(seq=1))
        perc_buffer.add(self._make_msg(seq=2))
        perc_buffer.flush()
        assert mock_perception_sdk.save_perception.call_count == 2
        # 再 flush 不会重复写
        perc_buffer.flush()
        assert mock_perception_sdk.save_perception.call_count == 2

    def test_seq_gap_detection(self, perc_buffer):
        """seq 丢失检测"""
        perc_buffer.add(self._make_msg(seq=1))
        perc_buffer.add(self._make_msg(seq=5))  # gap=3 (2,3,4 丢失)
        s = perc_buffer.get_stats()
        assert s["received"] == 2
        assert s["dropped"] == 3

    def test_seq_zero_skips_gap_check(self, perc_buffer):
        """seq=0 跳过丢失检测"""
        perc_buffer.add(self._make_msg(seq=0))
        perc_buffer.add(self._make_msg(seq=0))
        s = perc_buffer.get_stats()
        assert s["dropped"] == 0

    def test_write_failure_counted(self, perc_buffer, mock_perception_sdk):
        """写入失败计入 stats"""
        mock_perception_sdk.save_perception.side_effect = RuntimeError("db locked")
        for i in range(3):  # 触发 flush
            perc_buffer.add(self._make_msg(seq=i + 1))
        s = perc_buffer.get_stats()
        assert s["written"] == 0
        assert s["failed"] == 3

    def test_partial_write_failure(self, perc_buffer, mock_perception_sdk):
        """部分写入失败"""
        call_count = [0]
        def _side_effect(**kw):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("fail on 2nd")
            return {"memory_id": 1}
        mock_perception_sdk.save_perception.side_effect = _side_effect
        for i in range(3):
            perc_buffer.add(self._make_msg(seq=i + 1))
        s = perc_buffer.get_stats()
        assert s["written"] == 2
        assert s["failed"] == 1

    def test_get_stats_thread_safe(self, perc_buffer):
        """get_stats 返回快照"""
        perc_buffer.add(self._make_msg(seq=1))
        s1 = perc_buffer.get_stats()
        perc_buffer.add(self._make_msg(seq=2))
        s2 = perc_buffer.get_stats()
        assert s1["received"] == 1
        assert s2["received"] == 2

    def test_concurrent_add(self, mock_perception_sdk):
        """多线程并发 add 不丢数据"""
        buf = PerceptionBuffer(mock_perception_sdk, batch_size=1000)
        errors = []
        n_threads = 4
        n_per_thread = 50

        def _add_batch(start):
            try:
                for i in range(n_per_thread):
                    buf.add(self._make_msg(seq=0, desc=f"t{start}-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_add_batch, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        s = buf.get_stats()
        assert s["received"] == n_threads * n_per_thread

    def test_buffer_bounded_by_batch_size(self, mock_perception_sdk):
        """锁外 I/O 后 buffer 大小始终 ≤ batch_size（背压告警为防御性守卫）"""
        buf = PerceptionBuffer(mock_perception_sdk, batch_size=5)
        for _ in range(25):  # 5 个 batch 周期
            buf.add(self._make_msg(seq=0))
        # 最后一个 batch 周期的尾巴（25 % 5 = 0，刚好清空）
        with buf._lock:
            assert len(buf._buffer) <= 5
        s = buf.get_stats()
        assert s["received"] == 25
        assert s["written"] == 25  # 5 个 batch × 5 items

    def test_flush_data_correctness(self, perc_buffer, mock_perception_sdk):
        """flush 传递正确参数"""
        msg = _FakePerceptionData(
            seq=1, description="saw wall", perception_type="visual",
            data='{"dist": 0.5}', metadata='{"sensor": "lidar"}',
            session_id="s1", collection="robot1",
        )
        perc_buffer.add(msg)
        perc_buffer.flush()
        mock_perception_sdk.save_perception.assert_called_once_with(
            description="saw wall",
            perception_type="visual",
            data='{"dist": 0.5}',
            metadata='{"sensor": "lidar"}',
            session_id="s1",
            collection="robot1",
        )

    def test_empty_optional_fields_become_none(self, perc_buffer, mock_perception_sdk):
        """空字符串可选字段转 None"""
        msg = _FakePerceptionData(
            seq=1, description="test",
            data="", metadata="", session_id="", collection="",
        )
        perc_buffer.add(msg)
        perc_buffer.flush()
        call_kwargs = mock_perception_sdk.save_perception.call_args[1]
        assert call_kwargs["data"] is None
        assert call_kwargs["metadata"] is None
        assert call_kwargs["session_id"] is None
        assert call_kwargs["collection"] is None


# ══════════════════════════════════════════════════════════════
# 4. Service 回调
# ══════════════════════════════════════════════════════════════

class _FakeRequest:
    """通用 fake Service Request"""
    def __getattr__(self, name):
        return ""

class _FakeResponse:
    """通用 fake Service Response"""
    def __init__(self):
        self.success = False
        self.error = ""

    def __getattr__(self, name):
        if name in ("success", "error"):
            return super().__getattribute__(name)
        return ""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)


class TestLearnCallback:
    def test_learn_success(self, node, mock_sdk):
        req = MagicMock()
        req.insight = "grasp force 5N works for soft objects"
        req.context = '{"env": "lab"}'
        req.session_id = "s1"
        req.collection = "robot1"
        resp = _FakeResponse()
        result = node._learn_cb(req, resp)
        assert result.success is True
        assert result.memory_id == 42
        mock_sdk.learn.assert_called_once_with(
            insight="grasp force 5N works for soft objects",
            context='{"env": "lab"}',
            session_id="s1",
            collection="robot1",
        )

    def test_learn_empty_optionals(self, node, mock_sdk):
        req = MagicMock()
        req.insight = "test"
        req.context = ""
        req.session_id = ""
        req.collection = ""
        resp = _FakeResponse()
        node._learn_cb(req, resp)
        mock_sdk.learn.assert_called_once_with(
            insight="test", context="", session_id=None, collection=None,
        )


class TestRecallCallback:
    def test_recall_success(self, node, mock_sdk):
        req = MagicMock()
        req.query = "grasp force"
        req.n = 5
        req.min_confidence = 0.3
        req.session_id = ""
        req.context_filter = ""
        req.spatial_sort = ""
        req.collection = ""
        resp = _FakeResponse()
        resp.memories = []
        resp.total = 0
        resp.mode = ""
        result = node._recall_cb(req, resp)
        assert result.total == 1
        assert result.mode == "bm25_only"  # embed_backend=none
        mock_sdk.recall.assert_called_once()

    def test_recall_with_json_params(self, node, mock_sdk):
        req = MagicMock()
        req.query = "test"
        req.n = 3
        req.min_confidence = 0.5
        req.session_id = ""
        req.context_filter = '{"env": "lab"}'
        req.spatial_sort = '{"target": [1,2,3]}'
        req.collection = ""
        resp = _FakeResponse()
        resp.memories = []
        resp.total = 0
        resp.mode = ""
        node._recall_cb(req, resp)
        call_kwargs = mock_sdk.recall.call_args[1]
        assert call_kwargs["context_filter"] == {"env": "lab"}
        assert call_kwargs["spatial_sort"] == {"target": [1, 2, 3]}

    def test_recall_invalid_json_raises_validation_error(self, node, mock_sdk):
        """无效 JSON → ValidationError → L1 分支"""
        req = MagicMock()
        req.query = "test"
        req.n = 5
        req.min_confidence = 0.3
        req.session_id = ""
        req.context_filter = "not-json{{"
        req.spatial_sort = ""
        req.collection = ""
        resp = _FakeResponse()
        # ros_error_boundary 会捕获 ValidationError
        result = node._recall_cb(req, resp)
        assert result.success is False
        assert "参数校验失败" in result.error

    def test_recall_invalid_spatial_sort_json(self, node, mock_sdk):
        req = MagicMock()
        req.query = "test"
        req.n = 5
        req.min_confidence = 0.3
        req.session_id = ""
        req.context_filter = ""
        req.spatial_sort = "bad json"
        req.collection = ""
        resp = _FakeResponse()
        result = node._recall_cb(req, resp)
        assert result.success is False
        assert "参数校验失败" in result.error

    def test_recall_mode_hybrid_when_embed_enabled(self, node):
        """embed_backend != none → mode=hybrid"""
        node._embed_backend = "onnx"
        req = MagicMock()
        req.query = "test"
        req.n = 5
        req.min_confidence = 0.3
        req.session_id = ""
        req.context_filter = ""
        req.spatial_sort = ""
        req.collection = ""
        resp = _FakeResponse()
        resp.memories = []
        resp.total = 0
        resp.mode = ""
        result = node._recall_cb(req, resp)
        assert result.mode == "hybrid"


class TestSavePerceptionCallback:
    def test_save_perception_success(self, node, mock_sdk):
        req = MagicMock()
        req.description = "red cube at 0.5m"
        req.perception_type = "visual"
        req.data = '{"color": "red"}'
        req.metadata = ""
        req.session_id = "s1"
        req.collection = ""
        resp = _FakeResponse()
        resp.memory_id = 0
        result = node._save_perception_cb(req, resp)
        assert result.success is True
        assert result.memory_id == 99

    def test_save_perception_default_type(self, node, mock_sdk):
        """perception_type 空 → 默认 visual"""
        req = MagicMock()
        req.description = "test"
        req.perception_type = ""
        req.data = ""
        req.metadata = ""
        req.session_id = ""
        req.collection = ""
        resp = _FakeResponse()
        resp.memory_id = 0
        node._save_perception_cb(req, resp)
        call_kwargs = mock_sdk.save_perception.call_args[1]
        assert call_kwargs["perception_type"] == "visual"


class TestForgetCallback:
    def test_forget_success(self, node, mock_sdk):
        req = MagicMock()
        req.memory_id = 42
        req.reason = "incorrect data"
        resp = _FakeResponse()
        result = node._forget_cb(req, resp)
        assert result.success is True
        mock_sdk.forget.assert_called_once_with(memory_id=42, reason="incorrect data")


class TestUpdateCallback:
    def test_update_success(self, node, mock_sdk):
        req = MagicMock()
        req.memory_id = 42
        req.new_content = "corrected content"
        req.context = "updated context"
        resp = _FakeResponse()
        resp.old_content = ""
        resp.new_content_out = ""
        result = node._update_cb(req, resp)
        assert result.success is True
        assert result.old_content == "old"
        assert result.new_content_out == "new"


class TestStartSessionCallback:
    def test_start_session_success(self, node, mock_sdk):
        req = MagicMock()
        req.context = '{"task": "pick"}'
        req.collection = ""
        resp = _FakeResponse()
        resp.session_id = ""
        resp.collection = ""
        resp.active_memories_count = 0
        result = node._start_session_cb(req, resp)
        assert result.success is True
        assert result.session_id == "session-abc-123"
        assert result.collection == "test"  # 默认 collection
        assert result.active_memories_count == 10  # mock 返回 10

    def test_start_session_custom_collection(self, node, mock_sdk):
        req = MagicMock()
        req.context = ""
        req.collection = "robot2"
        resp = _FakeResponse()
        resp.session_id = ""
        resp.collection = ""
        resp.active_memories_count = 0
        result = node._start_session_cb(req, resp)
        assert result.collection == "robot2"


class TestEndSessionCallback:
    def test_end_session_with_score(self, node, mock_sdk):
        req = MagicMock()
        req.session_id = "s1"
        req.has_outcome_score = True
        req.outcome_score = 0.85
        resp = _FakeResponse()
        resp.summary_json = ""
        resp.decayed_count = 0
        resp.consolidated_json = ""
        resp.related_memories = []
        result = node._end_session_cb(req, resp)
        assert result.success is True
        assert result.decayed_count == 2
        mock_sdk.end_session.assert_called_once_with(
            session_id="s1", outcome_score=0.85,
        )

    def test_end_session_without_score(self, node, mock_sdk):
        """has_outcome_score=false → score=None"""
        req = MagicMock()
        req.session_id = "s1"
        req.has_outcome_score = False
        req.outcome_score = 0.0
        resp = _FakeResponse()
        resp.summary_json = ""
        resp.decayed_count = 0
        resp.consolidated_json = ""
        resp.related_memories = []
        node._end_session_cb(req, resp)
        mock_sdk.end_session.assert_called_once_with(
            session_id="s1", outcome_score=None,
        )

    def test_end_session_zero_score_preserved(self, node, mock_sdk):
        """评分为 0（完全失败）不等于未提供（P1 #9 修复验证）"""
        req = MagicMock()
        req.session_id = "s1"
        req.has_outcome_score = True
        req.outcome_score = 0.0
        resp = _FakeResponse()
        resp.summary_json = ""
        resp.decayed_count = 0
        resp.consolidated_json = ""
        resp.related_memories = []
        node._end_session_cb(req, resp)
        mock_sdk.end_session.assert_called_once_with(
            session_id="s1", outcome_score=0.0,
        )


# ══════════════════════════════════════════════════════════════
# 5. _count_active_memories
# ══════════════════════════════════════════════════════════════

class TestCountActiveMemories:
    def test_success(self, node, mock_sdk):
        count = node._count_active_memories("test")
        assert count == 10

    def test_db_error_returns_zero_with_warning(self, node, mock_sdk):
        mock_sdk._db.conn.execute.side_effect = RuntimeError("db closed")
        count = node._count_active_memories("test")
        assert count == 0


# ══════════════════════════════════════════════════════════════
# 6. _publish_ready
# ══════════════════════════════════════════════════════════════

class TestPublishReady:
    def test_publish_ready_true(self, node):
        node._publish_ready(True)
        node._ready_pub.publish.assert_called_once()

    def test_publish_ready_false(self, node):
        node._publish_ready(False)
        node._ready_pub.publish.assert_called_once()


# ══════════════════════════════════════════════════════════════
# 7. destroy_node
# ══════════════════════════════════════════════════════════════

class TestDestroyNode:
    def test_shutdown_sequence(self, node, mock_sdk, mock_perception_sdk):
        """关闭顺序: ready=false → flush → close perception → close main"""
        call_order = []
        node._ready_pub.publish.side_effect = lambda s: call_order.append("ready_false")
        mock_perception_sdk.close.side_effect = lambda: call_order.append("close_perception")
        mock_sdk.close.side_effect = lambda: call_order.append("close_main")
        node.destroy_node()
        assert call_order == ["ready_false", "close_perception", "close_main"]

    def test_shutdown_publish_failure_continues(self, node, mock_sdk, mock_perception_sdk):
        """publish 失败不阻塞关闭（P1 #11 修复验证）"""
        node._ready_pub.publish.side_effect = RuntimeError("publisher dead")
        # 不应抛异常
        node.destroy_node()
        mock_perception_sdk.close.assert_called_once()
        mock_sdk.close.assert_called_once()


# ══════════════════════════════════════════════════════════════
# 8. 错误边界集成
# ══════════════════════════════════════════════════════════════

class TestErrorBoundaryIntegration:
    """Service 回调 + ros_error_boundary 集成"""

    def test_learn_validation_error(self, node, mock_sdk):
        """learn SDK 抛 ValidationError → L1 response"""
        mock_sdk.learn.side_effect = ValidationError("insight too short")
        req = MagicMock()
        req.insight = ""
        req.context = ""
        req.session_id = ""
        req.collection = ""
        resp = _FakeResponse()
        result = node._learn_cb(req, resp)
        assert result.success is False
        assert "参数校验失败" in result.error

    def test_learn_database_error(self, node, mock_sdk):
        """learn SDK 抛 DatabaseError → L2 response + ready=false"""
        mock_sdk.learn.side_effect = DatabaseError("WAL checkpoint failed")
        req = MagicMock()
        req.insight = "test"
        req.context = ""
        req.session_id = ""
        req.collection = ""
        resp = _FakeResponse()
        result = node._learn_cb(req, resp)
        assert result.success is False
        assert "数据库错误" in result.error

    def test_recall_unexpected_error(self, node, mock_sdk):
        """recall SDK 抛未知异常 → L3 response"""
        mock_sdk.recall.side_effect = RuntimeError("unexpected")
        req = MagicMock()
        req.query = "test"
        req.n = 5
        req.min_confidence = 0.3
        req.session_id = ""
        req.context_filter = ""
        req.spatial_sort = ""
        req.collection = ""
        resp = _FakeResponse()
        resp.memories = []
        resp.total = 0
        resp.mode = ""
        result = node._recall_cb(req, resp)
        assert result.success is False
        assert "内部错误" in result.error
