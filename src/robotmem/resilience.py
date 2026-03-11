"""容错原语 — safe_db_write / safe_db_transaction / mcp_error_boundary / ServiceCooldown

robotmem 精简版：去掉 CircuitBreaker / FileStateCircuitBreaker / ProcessManager /
graceful_kill / create_http_client（index1 hooks/observer 专用）。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────
# ServiceCooldown — 外部服务指数退避
# ────────────────────────────────────────────────────


class ServiceCooldown:
    """外部服务冷却器 — 指数退避，单进程内存状态

    使用者：embed.py Ollama 连接失败退避
    """

    def __init__(
        self,
        name: str,
        base_cooldown: float = 60.0,
        max_cooldown: float = 300.0,
        backoff_factor: float = 2.0,
    ):
        self.name = name
        self._base = base_cooldown
        self._max = max_cooldown
        self._factor = backoff_factor
        self._last_failure: float = 0.0
        self._consecutive_failures: int = 0

    @property
    def is_cooling(self) -> bool:
        """冷却中返回 True"""
        if self._last_failure == 0:
            return False
        return (time.monotonic() - self._last_failure) < self.current_backoff

    @property
    def current_backoff(self) -> float:
        """当前冷却时间（秒）"""
        if self._consecutive_failures == 0:
            return 0.0
        return min(
            self._base * (self._factor ** (self._consecutive_failures - 1)),
            self._max,
        )

    def record_failure(self):
        self._last_failure = time.monotonic()
        self._consecutive_failures += 1
        logger.warning(
            "%s 失败 (连续第 %d 次)，冷却 %.0fs",
            self.name,
            self._consecutive_failures,
            self.current_backoff,
        )

    def record_success(self):
        self._consecutive_failures = 0
        self._last_failure = 0.0

    def reset(self):
        self._consecutive_failures = 0
        self._last_failure = 0.0


# ────────────────────────────────────────────────────
# safe_db_write — 单条 DB 写保护
# ────────────────────────────────────────────────────


def safe_db_write(
    conn: sqlite3.Connection,
    sql: str,
    params: list | tuple | None = None,
) -> int | None:
    """安全 DB 写入 — 锁超时/磁盘满/DB损坏返回 None 而非崩溃

    返回值：
    - int: lastrowid（写入成功）
    - None: 写入失败（锁超时=可重试，磁盘满/损坏=不可恢复，均已记日志）

    调用方必须检查 None 并根据场景决定是否重试。
    conn 必须以默认 isolation_level 打开（非 autocommit）。
    """
    try:
        with conn:
            cursor = conn.execute(sql, params or [])
            return cursor.lastrowid
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "database is locked" in msg:
            logger.warning("DB 锁超时，写入跳过: %s", sql[:80])
            return None
        if "disk i/o error" in msg or "disk is full" in msg:
            logger.error("DB 磁盘错误: %s", e)
            return None
        raise
    except sqlite3.DatabaseError as e:
        msg = str(e).lower()
        if "malformed" in msg or "not a database" in msg:
            logger.error("DB 损坏: %s", e)
            return None
        raise


# ────────────────────────────────────────────────────
# safe_db_transaction — 原子批量写入
# ────────────────────────────────────────────────────


def safe_db_transaction(
    conn: sqlite3.Connection,
    fn: Callable[[sqlite3.Connection], Any],
) -> tuple[bool, Any]:
    """安全批量写入 — 在单个事务中执行 fn(conn)

    返回 (success, result):
    - (True, fn的返回值) — 成功
    - (False, None) — 锁超时/磁盘满/DB损坏
    """
    try:
        with conn:
            result = fn(conn)
        return True, result
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "database is locked" in msg:
            logger.warning("DB 锁超时，事务回滚")
            return False, None
        if "disk i/o error" in msg or "disk is full" in msg:
            logger.error("DB 磁盘错误: %s", e)
            return False, None
        raise
    except sqlite3.DatabaseError as e:
        msg = str(e).lower()
        if "malformed" in msg or "not a database" in msg:
            logger.error("DB 损坏: %s", e)
            return False, None
        raise


# ────────────────────────────────────────────────────
# mcp_error_boundary — MCP tool 错误边界
# ────────────────────────────────────────────────────


def mcp_error_boundary(func: Callable) -> Callable:
    """MCP tool 错误边界装饰器 — 异常不传播到客户端"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except sqlite3.DatabaseError as e:
            logger.error("MCP tool %s DB 错误: %s", func.__name__, e)
            return {"error": "数据库异常，请查看服务端日志"}
        except Exception as e:
            logger.exception("MCP tool %s 未知错误", func.__name__)
            return {"error": "内部错误，请查看服务端日志"}

    return wrapper
