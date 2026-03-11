"""会话领域 — Session 生命周期

从 index1 精简：去掉 maintenance_state / orient_feedback / brief /
raw_events / detect_expired / stats 等。

每个函数接收 conn: sqlite3.Connection 作为第一参数。
"""

from __future__ import annotations

import json
import logging
import sqlite3

from ..resilience import safe_db_transaction, safe_db_write

logger = logging.getLogger(__name__)


def get_or_create_session(
    conn: sqlite3.Connection,
    external_id: str | None,
    collection: str,
) -> dict | None:
    """获取或创建会话。失败返回 None。

    三层防御：
    - L1 事前：collection 非空
    - L2 事中：safe_db_transaction 原子操作
    - L3 事后：返回 session dict
    """
    if not collection:
        logger.error("get_or_create_session: collection 为空")
        return None

    def _do(c: sqlite3.Connection) -> dict:
        if external_id:
            cur = c.cursor()
            cur.row_factory = sqlite3.Row
            existing = cur.execute(
                "SELECT * FROM sessions WHERE external_id=?",
                [external_id],
            ).fetchone()
            if existing:
                c.execute("""
                    UPDATE sessions
                    SET session_count=session_count+1,
                        updated_at=strftime('%Y-%m-%dT%H:%M:%f','now'),
                        status='active'
                    WHERE id=?
                """, [existing["id"]])
                # UPDATE 后重新读取确保返回值与 DB 一致
                row = cur.execute(
                    "SELECT * FROM sessions WHERE id=?",
                    [existing["id"]],
                ).fetchone()
                return dict(row)

        cursor = c.execute("""
            INSERT INTO sessions (external_id, collection, session_count, status)
            VALUES (?, ?, 1, 'active')
        """, [external_id, collection])
        return {
            "id": cursor.lastrowid,
            "external_id": external_id,
            "collection": collection,
            "session_count": 1,
        }

    ok, result = safe_db_transaction(conn, _do)
    return result if ok else None


def update_session_context(
    conn: sqlite3.Connection,
    external_id: str,
    context: str,
) -> None:
    """更新会话上下文 — 通用 context JSON 字段

    圆桌决策（Abbeel）：支持通用 context 更新，不限制字段。
    context 为 JSON 字符串，可包含 env/task/robot_id 等任意字段。
    上限 64KB 防止意外写入大数据（点云/图像特征）。
    """
    if not external_id or not context:
        return
    if len(context) > 65536:
        logger.warning("update_session_context: context 超过 64KB，截断")
        context = context[:65536]
    safe_db_write(conn, """
        UPDATE sessions SET context=?, updated_at=strftime('%Y-%m-%dT%H:%M:%f','now')
        WHERE external_id=?
    """, [context, external_id])


def get_session_context(
    conn: sqlite3.Connection, external_id: str,
) -> dict | None:
    """获取会话上下文 JSON"""
    if not external_id:
        return None
    try:
        row = conn.execute(
            "SELECT context FROM sessions WHERE external_id=?",
            [external_id],
        ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None
    except Exception as e:
        logger.warning("get_session_context 失败: %s", e)
        return None


def mark_session_ended(conn: sqlite3.Connection, external_id: str) -> bool:
    """标记 session 正常结束"""
    if not external_id:
        return False
    return safe_db_write(conn, """
        UPDATE sessions SET status='ended', updated_at=strftime('%Y-%m-%dT%H:%M:%f','now')
        WHERE external_id=?
    """, [external_id]) is not None


def insert_session_outcome(
    conn: sqlite3.Connection,
    external_id: str,
    score: float | None = None,
) -> bool:
    """记录 session 结果评分

    参数 external_id 对应 sessions.external_id（字符串标识）。
    内部查询 sessions 表获取主键 id 后写入 session_outcomes。

    三层防御：
    - L1 事前：external_id 非空
    - L2 事中：safe_db_write
    - L3 事后：返回 bool
    """
    if not external_id:
        return False
    return safe_db_write(conn, """
        INSERT INTO session_outcomes (session_id, score)
        VALUES (?, ?)
    """, [external_id, score]) is not None


def get_session_summary(
    conn: sqlite3.Connection,
    session_id: str,
    collection: str,
) -> dict:
    """获取 session 摘要 — end_session 返回用

    统计本次会话中记录的记忆数和类型分布。
    """
    if not session_id:
        return {"memory_count": 0, "by_type": {}, "by_category": {}}

    def _do(c: sqlite3.Connection) -> dict:
        total = c.execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE session_id=? AND collection=?",
            [session_id, collection],
        ).fetchone()[0]

        by_type: dict[str, int] = {}
        for row in c.execute(
            "SELECT type, COUNT(*) FROM memories "
            "WHERE session_id=? AND collection=? GROUP BY type",
            [session_id, collection],
        ).fetchall():
            by_type[row[0]] = row[1]

        by_category: dict[str, int] = {}
        for row in c.execute(
            "SELECT category, COUNT(*) FROM memories "
            "WHERE session_id=? AND collection=? GROUP BY category",
            [session_id, collection],
        ).fetchall():
            by_category[row[0]] = row[1]

        return {
            "memory_count": total,
            "by_type": by_type,
            "by_category": by_category,
        }

    ok, result = safe_db_transaction(conn, _do)
    if ok and result:
        return result
    return {"memory_count": 0, "by_type": {}, "by_category": {}}
