"""搜索领域 — FTS5 / 向量搜索（recall 用）

从 index1 精简：去掉 orient / universal / applicable_when 等。
返回完整记忆字段供 recall 使用（区别于 db_cog.py 的 dedup 最小字段）。

每个函数接收 conn: sqlite3.Connection 作为第一参数。
"""

from __future__ import annotations

import logging
import re
import sqlite3

from ..db import floats_to_blob, tokenize_for_fts5

logger = logging.getLogger(__name__)


def fts_search_memories(
    conn: sqlite3.Connection,
    query: str,
    collection: str,
    limit: int = 10,
) -> list[dict]:
    """BM25 全文搜索 memories — recall 用

    返回完整字段: id, content, type, perception_type, session_id,
    category, confidence, context, scope_files, created_at, bm25_score

    三层防御：
    - L1 事前：query 非空 + FTS5 语法清理
    - L2 事中：try-except 捕获 FTS5 异常
    - L3 事后：返回空列表不崩溃
    """
    if not query or not query.strip():
        return []
    limit = min(max(1, limit), 100)

    # FTS5 语法清理 — 只保留字母数字和 CJK，其余替换为空格
    cleaned = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', " ", tokenize_for_fts5(query))
    cleaned = re.sub(r"\b(AND|OR|NOT|NEAR)\b", " ", cleaned, flags=re.IGNORECASE)
    tokens = cleaned.split()
    if not tokens:
        return []
    # 过滤单字符非 CJK token（噪音太大）
    tokens = [t for t in tokens if len(t) > 1 or re.search(r'[\u4e00-\u9fff]', t)]
    if not tokens:
        return []
    fts5_query = " OR ".join(f'"{t}"' for t in tokens if t.strip())
    if not fts5_query:
        return []

    try:
        rows = conn.execute("""
            SELECT m.id, m.content, m.human_summary, m.type, m.perception_type,
                   m.session_id, m.category, m.confidence,
                   m.context, m.scope_files, m.scope_entities,
                   m.created_at,
                   bm25(memories_fts) as bm25_score
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.rowid
            WHERE memories_fts MATCH ?
              AND m.collection = ?
              AND m.status = 'active'
            ORDER BY bm25(memories_fts)
            LIMIT ?
        """, [fts5_query, collection, limit]).fetchall()

        return [
            {
                "id": r[0], "content": r[1], "human_summary": r[2],
                "type": r[3], "perception_type": r[4],
                "session_id": r[5], "category": r[6],
                "confidence": r[7], "context": r[8],
                "scope_files": r[9], "scope_entities": r[10],
                "created_at": r[11], "bm25_score": r[12],
            }
            for r in rows
        ]
    except sqlite3.OperationalError as e:
        logger.warning("FTS5 搜索失败: %s | query=%r", e, fts5_query)
        return []


def vec_search_memories(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    collection: str,
    limit: int = 10,
    vec_loaded: bool = False,
) -> list[dict]:
    """向量 KNN 搜索 memories — recall 用

    返回完整字段: id, content, type, perception_type, session_id,
    category, confidence, context, scope_files, created_at, distance

    三层防御：
    - L1 事前：embedding 非空 + vec_loaded 检查
    - L2 事中：try-except 捕获 vec0 异常
    - L3 事后：返回空列表不崩溃
    """
    if not vec_loaded or not query_embedding:
        return []
    limit = min(max(1, limit), 100)
    if not isinstance(query_embedding[0], (int, float)):
        logger.warning("vec_search_memories: embedding 元素类型错误: %s", type(query_embedding[0]))
        return []

    try:
        blob = floats_to_blob(query_embedding)
        rows = conn.execute("""
            SELECT m.id, m.content, m.human_summary, m.type, m.perception_type,
                   m.session_id, m.category, m.confidence,
                   m.context, m.scope_files, m.scope_entities,
                   m.created_at,
                   v.distance
            FROM memories_vec v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ?
              AND m.collection = ?
              AND m.status = 'active'
              AND k = ?
        """, [blob, collection, limit]).fetchall()

        return [
            {
                "id": r[0], "content": r[1], "human_summary": r[2],
                "type": r[3], "perception_type": r[4],
                "session_id": r[5], "category": r[6],
                "confidence": r[7], "context": r[8],
                "scope_files": r[9], "scope_entities": r[10],
                "created_at": r[11], "distance": r[12],
            }
            for r in rows
        ]
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        logger.warning("向量搜索失败: %s", e)
        return []
