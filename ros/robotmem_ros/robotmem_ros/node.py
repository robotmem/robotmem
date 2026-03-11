"""robotmem ROS 2 Node — SDK 薄包装

7 Service + 1 Topic（perception 高频流）+ Ready 信号
三层防御核心在 SDK，ROS 层只做 @ros_error_boundary 薄壳。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from functools import wraps

import rclpy
from rclpy.callback_group import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from robotmem_msgs.msg import Memory, NodeStatus, PerceptionData
from robotmem_msgs.srv import (
    EndSession,
    Forget,
    Learn,
    Recall,
    SavePerception,
    StartSession,
    Update,
)

from robotmem.exceptions import DatabaseError, ValidationError
from robotmem.sdk import RobotMemory

logger = logging.getLogger(__name__)

# QoS: ready 信号（后到的 subscriber 也能收到）
QOS_READY = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)

# QoS: perception reliable（仿真，KEEP_LAST 让 depth 生效）
QOS_PERCEPTION_RELIABLE = QoSProfile(
    depth=100,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

# QoS: perception best_effort（实时，低延迟）
QOS_PERCEPTION_BEST_EFFORT = QoSProfile(
    depth=100,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


# ── 装饰器 ──

def ros_error_boundary(func):
    """ROS Service 回调错误边界 — 对标 @mcp_error_boundary"""
    @wraps(func)
    def wrapper(self, request, response):
        try:
            return func(self, request, response)
        except ValidationError as e:
            response.success = False
            response.error = f"参数校验失败: {e}"
            self.get_logger().warning("L1 %s: %s", func.__name__, e)
        except DatabaseError as e:
            response.success = False
            response.error = f"数据库错误: {e}"
            self.get_logger().error("L2 %s: %s", func.__name__, e)
            self._publish_ready(False)
        except Exception as e:
            response.success = False
            response.error = f"内部错误: {e}"
            self.get_logger().error("L3 %s: %s", func.__name__, e)
        return response
    return wrapper


# ── Perception Buffer ──

class PerceptionBuffer:
    """批量写入 + seq 丢失检测 + 背压告警（线程安全）"""

    def __init__(self, mem: RobotMemory, batch_size: int = 50, flush_interval: float = 1.0):
        self._mem = mem
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._last_seq: int = 0
        self._stats = {"received": 0, "written": 0, "dropped": 0, "failed": 0}
        self._last_stats_time = time.monotonic()

    def add(self, msg: PerceptionData):
        """L1: seq 丢失检测 + 入 buffer（共享状态全部在锁内）"""
        item = {
            "description": msg.description,
            "perception_type": msg.perception_type,
            "data": msg.data or None,
            "metadata": msg.metadata or None,
            "session_id": msg.session_id or None,
            "collection": msg.collection or None,
        }

        batch_to_flush = None
        with self._lock:
            self._stats["received"] += 1

            # seq 丢失检测
            if msg.seq > 0 and self._last_seq > 0:
                gap = msg.seq - self._last_seq - 1
                if gap > 0:
                    self._stats["dropped"] += gap
            self._last_seq = msg.seq

            self._buffer.append(item)
            # 背压检测
            if len(self._buffer) > 2 * self._batch_size:
                logger.warning(
                    "perception 背压: buffer=%d > 2×batch_size=%d",
                    len(self._buffer), 2 * self._batch_size,
                )
            if len(self._buffer) >= self._batch_size:
                batch_to_flush = self._buffer[:]
                self._buffer.clear()

        # DB I/O 在锁外执行，不阻塞 add()
        if batch_to_flush:
            self._write_batch(batch_to_flush)

    def flush(self):
        """外部调用 flush（Timer / shutdown）"""
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()
        # 锁外写入
        self._write_batch(batch)

    def _write_batch(self, batch: list[dict]):
        """DB I/O — 在锁外调用"""
        t0 = time.monotonic()
        written = 0
        failed = 0
        for item in batch:
            try:
                self._mem.save_perception(**item)
                written += 1
            except Exception as e:
                failed += 1
                logger.warning("perception 写入失败: %s", e)
        # stats 更新回锁内
        with self._lock:
            self._stats["written"] += written
            self._stats["failed"] += failed
        elapsed = (time.monotonic() - t0) * 1000
        logger.debug("perception flush: batch=%d, elapsed=%.1fms", len(batch), elapsed)

    def get_stats(self) -> dict:
        """线程安全的 stats 快照"""
        with self._lock:
            return dict(self._stats)

    def log_stats(self):
        """周期 stats 日志（Node Timer 调用）"""
        now = time.monotonic()
        if now - self._last_stats_time < 60:
            return
        self._last_stats_time = now
        s = self.get_stats()
        logger.info(
            "perception stats: received=%d written=%d dropped=%d failed=%d",
            s["received"], s["written"], s["dropped"], s["failed"],
        )


# ── 辅助函数 ──

def _dict_to_memory_msg(d: dict) -> Memory:
    """SDK recall dict → Memory.msg"""
    m = Memory()
    m.id = d.get("id", 0)
    m.content = d.get("content", "")
    m.type = d.get("type", "")
    m.perception_type = d.get("perception_type") or ""
    m.confidence = d.get("confidence", 0.0)
    m.rrf_score = d.get("_rrf_score", 0.0)
    m.context_json = d.get("context") or ""
    m.session_id = d.get("session_id") or ""
    m.created_at = d.get("created_at") or ""
    return m


# ── Node ──

class RobotMemNode(Node):
    """robotmem ROS 2 Node"""

    def __init__(self):
        super().__init__('robotmem')

        # ── L1 参数 ──
        self.declare_parameter('db_path', '')
        self.declare_parameter('collection', 'default')
        self.declare_parameter('embed_backend', 'onnx')
        self.declare_parameter('perception_qos', 'reliable')
        self.declare_parameter('perception_batch_size', 50)
        self.declare_parameter('perception_flush_interval', 1.0)

        db_path = (self.get_parameter('db_path').value or '').strip() or None
        self._collection = self.get_parameter('collection').value
        embed_backend = self.get_parameter('embed_backend').value
        perception_qos = self.get_parameter('perception_qos').value
        batch_size = self.get_parameter('perception_batch_size').value
        flush_interval = self.get_parameter('perception_flush_interval').value

        # L1: embed_backend 白名单
        if embed_backend not in ('onnx', 'ollama', 'none'):
            self.get_logger().warning("embed_backend=%r 非法，使用 onnx", embed_backend)
            embed_backend = 'onnx'

        # 缓存构造期参数，避免后续访问 SDK 私有属性
        self._db_path_str = db_path or "~/.robotmem/memory.db"
        self._embed_backend = embed_backend

        # ── L2 SDK 初始化 ──
        try:
            self._mem = RobotMemory(
                db_path=db_path, collection=self._collection, embed_backend=embed_backend,
            )
        except Exception as e:
            self.get_logger().fatal("robotmem 启动失败: %s", e)
            raise SystemExit(1)

        try:
            self._perception_mem = RobotMemory(
                db_path=db_path, collection=self._collection, embed_backend='none',
            )
        except Exception as e:
            self.get_logger().fatal("perception SDK 初始化失败: %s", e)
            self._mem.close()
            raise SystemExit(1)

        # ── CallbackGroup（读写分离）──
        self._write_group = MutuallyExclusiveCallbackGroup()
        self._read_group = ReentrantCallbackGroup()
        self._perc_group = ReentrantCallbackGroup()

        # ── 7 Service（recall 读操作独立 group）──
        for name, srv_type, cb, group in [
            ('learn', Learn, self._learn_cb, self._write_group),
            ('recall', Recall, self._recall_cb, self._read_group),
            ('save_perception', SavePerception, self._save_perception_cb, self._write_group),
            ('forget', Forget, self._forget_cb, self._write_group),
            ('update', Update, self._update_cb, self._write_group),
            ('start_session', StartSession, self._start_session_cb, self._write_group),
            ('end_session', EndSession, self._end_session_cb, self._write_group),
        ]:
            self.create_service(srv_type, name, cb, callback_group=group)

        # ── Perception Topic ──
        qos = QOS_PERCEPTION_RELIABLE if perception_qos == 'reliable' else QOS_PERCEPTION_BEST_EFFORT
        if perception_qos not in ('reliable', 'best_effort'):
            self.get_logger().warning("perception_qos=%r 非法，使用 reliable", perception_qos)
            qos = QOS_PERCEPTION_RELIABLE

        self._perc_buffer = PerceptionBuffer(self._perception_mem, batch_size, flush_interval)
        self.create_subscription(
            PerceptionData, 'perception', self._perception_cb, qos,
            callback_group=self._perc_group,
        )

        # Flush Timer
        self.create_timer(flush_interval, self._flush_timer_cb, callback_group=self._perc_group)
        # Stats Timer（每 60s）
        self.create_timer(60.0, self._stats_timer_cb, callback_group=self._perc_group)

        # ── L3 Ready 信号 ──
        self._ready_pub = self.create_publisher(NodeStatus, 'ready', QOS_READY)
        self._publish_ready(True)
        self.get_logger().info(
            "robotmem node ready: db=%s embed=%s collection=%s",
            self._db_path_str, self._embed_backend, self._collection,
        )

    # ── 辅助方法 ──

    def _count_active_memories(self, collection: str) -> int:
        """查询 active 记忆数量（集中 DB 访问 + 降级日志）"""
        try:
            return self._mem._db.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE collection=? AND status='active'",
                (collection,),
            ).fetchone()[0]
        except Exception as e:
            self.get_logger().warning("查询 active 记忆数失败: %s", e)
            return 0

    # ── Service 回调 ──

    @ros_error_boundary
    def _learn_cb(self, request, response):
        result = self._mem.learn(
            insight=request.insight,
            context=request.context or "",
            session_id=request.session_id or None,
            collection=request.collection or None,
        )
        response.success = True
        response.memory_id = result.get("memory_id", 0)
        response.status = result.get("status", "")
        response.auto_inferred_json = json.dumps(result.get("auto_inferred", {}))
        return response

    @ros_error_boundary
    def _recall_cb(self, request, response):
        # JSON 解析校验 — JSONDecodeError → ValidationError → L1 分支
        try:
            cf = json.loads(request.context_filter) if request.context_filter else None
        except json.JSONDecodeError as e:
            raise ValidationError(f"context_filter JSON 无效: {e}")
        try:
            ss = json.loads(request.spatial_sort) if request.spatial_sort else None
        except json.JSONDecodeError as e:
            raise ValidationError(f"spatial_sort JSON 无效: {e}")

        memories = self._mem.recall(
            query=request.query,
            n=request.n or 5,
            min_confidence=request.min_confidence or 0.3,
            session_id=request.session_id or None,
            context_filter=cf,
            spatial_sort=ss,
            collection=request.collection or None,
        )
        response.memories = [_dict_to_memory_msg(m) for m in memories]
        response.total = len(memories)
        response.mode = "hybrid" if self._embed_backend != 'none' else "bm25_only"
        return response

    @ros_error_boundary
    def _save_perception_cb(self, request, response):
        result = self._mem.save_perception(
            description=request.description,
            perception_type=request.perception_type or "visual",
            data=request.data or None,
            metadata=request.metadata or None,
            session_id=request.session_id or None,
            collection=request.collection or None,
        )
        response.success = True
        response.memory_id = result.get("memory_id", 0)
        return response

    @ros_error_boundary
    def _forget_cb(self, request, response):
        self._mem.forget(memory_id=request.memory_id, reason=request.reason)
        response.success = True
        return response

    @ros_error_boundary
    def _update_cb(self, request, response):
        result = self._mem.update(
            memory_id=request.memory_id,
            new_content=request.new_content,
            context=request.context or "",
        )
        response.success = True
        response.old_content = result.get("old_content", "")
        response.new_content_out = result.get("new_content", "")
        return response

    @ros_error_boundary
    def _start_session_cb(self, request, response):
        sid = self._mem.start_session(
            context=request.context or None,
            collection=request.collection or None,
        )
        coll = request.collection or self._collection
        response.success = True
        response.session_id = sid
        response.collection = coll
        response.active_memories_count = self._count_active_memories(coll)
        return response

    @ros_error_boundary
    def _end_session_cb(self, request, response):
        score = request.outcome_score if request.has_outcome_score else None
        result = self._mem.end_session(
            session_id=request.session_id,
            outcome_score=score,
        )
        response.success = True
        response.summary_json = json.dumps(result.get("summary", {}))
        response.decayed_count = result.get("decayed_count", 0)
        response.consolidated_json = json.dumps(result.get("consolidated", {}))
        response.related_memories = [
            _dict_to_memory_msg(m) for m in result.get("related_memories", [])
        ]
        return response

    # ── Perception 回调 ──

    def _perception_cb(self, msg: PerceptionData):
        self._perc_buffer.add(msg)

    def _flush_timer_cb(self):
        self._perc_buffer.flush()

    def _stats_timer_cb(self):
        self._perc_buffer.log_stats()

    # ── Ready 信号 ──

    def _publish_ready(self, ready: bool):
        status = NodeStatus()
        status.ready = ready
        status.db_path = self._db_path_str
        status.embed_backend = self._embed_backend
        status.collection = self._collection
        status.active_memories_count = self._count_active_memories(self._collection)
        self._ready_pub.publish(status)

    # ── 关闭 ──

    def destroy_node(self):
        """关闭顺序: ready=false → flush → close perception → close main → stats"""
        try:
            self._publish_ready(False)
        except Exception as e:
            self.get_logger().warning("shutdown 发布 ready=false 失败: %s", e)
        self._perc_buffer.flush()
        self._perception_mem.close()
        self._mem.close()
        s = self._perc_buffer.get_stats()
        self.get_logger().info(
            "robotmem shutdown: received=%d written=%d dropped=%d failed=%d",
            s["received"], s["written"], s["dropped"], s["failed"],
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotMemNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
