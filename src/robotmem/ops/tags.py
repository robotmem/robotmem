"""标签领域 — memory_tags CRUD

从 index1 精简：去掉 symbol_link / centroid / rename / tag_tree 等。
每个函数接收 conn: sqlite3.Connection 作为第一参数。
"""

from __future__ import annotations

import logging
import re
import sqlite3

from ..resilience import safe_db_transaction

logger = logging.getLogger(__name__)


def _normalize_tag(tag: str) -> str:
    """标签标准化 — 英文 lowercase + snake_case，中文 trim 空格

    纯计算无副作用，失败时 fallback 到 strip()。
    """
    try:
        tag = tag.strip()
        if not tag:
            return ""
        # 中文字符保留原文
        if re.search(r"[\u4e00-\u9fff]", tag):
            return tag
        # 英文: lowercase + 空格/连字符→下划线 + 连续下划线合并
        tag = tag.lower().replace(" ", "_").replace("-", "_")
        tag = re.sub(r"_+", "_", tag).strip("_")
        return tag
    except Exception:
        return tag.strip() if isinstance(tag, str) else ""


def add_tags(
    conn: sqlite3.Connection,
    memory_id: int,
    tags: list[str],
    source: str = "auto",
) -> int:
    """为记忆添加标签 — 幂等 INSERT OR IGNORE

    三层防御：
    - L1 事前：memory_id 正整数 + tags 非空 + normalize
    - L2 事中：safe_db_transaction 幂等写入
    - L3 事后：requested/written 日志
    """
    # L1: 事前校验
    if not isinstance(memory_id, int) or memory_id <= 0:
        logger.error("add_tags: 非法 memory_id=%r", memory_id)
        return 0
    if not isinstance(tags, (list, tuple)) or not tags:
        return 0
    clean_tags: list[str] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        nt = _normalize_tag(t)
        if nt:
            clean_tags.append(nt)
    if not clean_tags:
        return 0

    # L2: 幂等写入
    def _op(c: sqlite3.Connection) -> int:
        before = c.total_changes
        c.executemany(
            "INSERT OR IGNORE INTO memory_tags(memory_id, tag, source) "
            "VALUES(?, ?, ?)",
            [(memory_id, tag, source) for tag in clean_tags],
        )
        return c.total_changes - before

    ok, count = safe_db_transaction(conn, _op)

    # L3: 事后日志
    logger.debug(
        "add_tags memory=%d requested=%d written=%s",
        memory_id, len(clean_tags), count if ok else 0,
    )
    return count if ok else 0


def get_tags(conn: sqlite3.Connection, memory_id: int) -> list[str]:
    """获取记忆的所有标签"""
    if not isinstance(memory_id, int) or memory_id <= 0:
        return []
    try:
        rows = conn.execute(
            "SELECT tag FROM memory_tags WHERE memory_id = ? ORDER BY tag",
            (memory_id,),
        ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logger.warning("get_tags 失败 memory_id=%d: %s", memory_id, e)
        return []


def get_tag_stats(
    conn: sqlite3.Connection, collection: str | None = None,
) -> dict[str, int]:
    """标签使用统计 — JOIN memories WHERE status='active'"""
    try:
        if collection:
            rows = conn.execute("""
                SELECT mt.tag, COUNT(*) as cnt
                FROM memory_tags mt
                JOIN memories m ON mt.memory_id = m.id
                WHERE m.status = 'active' AND m.collection = ?
                GROUP BY mt.tag ORDER BY cnt DESC
            """, (collection,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT mt.tag, COUNT(*) as cnt
                FROM memory_tags mt
                JOIN memories m ON mt.memory_id = m.id
                WHERE m.status = 'active'
                GROUP BY mt.tag ORDER BY cnt DESC
            """).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception as e:
        logger.warning("get_tag_stats 失败: %s", e)
        return {}
