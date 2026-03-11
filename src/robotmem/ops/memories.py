"""记忆领域 — 统一 memories 表 CRUD + FTS5/vec0 同步

合并 index1 的 facts + perceptions 到单一 memories 表。
type='fact' = declarative memory（命题记忆）
type='perception' = procedural memory（程序性记忆）

每个函数接收 conn: sqlite3.Connection 作为第一参数。
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3

from ..db import floats_to_blob, tokenize_for_fts5
from ..dedup import jaccard_similarity
from ..resilience import safe_db_transaction, safe_db_write
from ..validators import positive_int, validate_args
from .tags import _normalize_tag

logger = logging.getLogger(__name__)

# perception_type 白名单
VALID_PERCEPTION_TYPES = frozenset({
    "visual", "tactile", "auditory", "proprioceptive", "procedural",
})


def insert_memory(
    conn: sqlite3.Connection,
    memory: dict,
    vec_loaded: bool = False,
) -> int | None:
    """插入一条记忆 — 原子写 memories + vec0

    type='fact': content 为命题文本（learn 工具）
    type='perception': content 为描述，perception_type/data/metadata 为感知数据

    三层防御：
    - L1 事前：content 非空 + 截断 300 字 + content_hash 去重
    - L2 事中：safe_db_transaction 原子写（memories + vec0）
    - L3 事后：logger 记录写入成功/失败

    FTS5 同步由 schema.py 中的触发器自动处理。
    Vec0 需手动同步（无触发器）。

    Args:
        memory: 记忆字段 dict（必须包含 content + collection）
        vec_loaded: sqlite-vec 扩展是否已加载

    Returns:
        新记忆 id，失败返回 None
    """
    # L1: content 非空 + 截断
    content = (memory.get("content") or "").strip()
    if not content:
        logger.error("insert_memory: content 为空，拒绝写入")
        return None
    content = content[:300]

    collection = memory.get("collection", "default")
    if not collection:
        logger.error("insert_memory: collection 为空，拒绝写入")
        return None

    mem_type = memory.get("type", "fact")
    if mem_type not in ("fact", "perception"):
        logger.error("insert_memory: 非法 type=%r", mem_type)
        return None

    # L1: perception_type 白名单（仅 perception 类型）
    perception_type = memory.get("perception_type")
    if mem_type == "perception" and perception_type:
        if perception_type not in VALID_PERCEPTION_TYPES:
            logger.error(
                "insert_memory: 非法 perception_type=%r，白名单=%s",
                perception_type, VALID_PERCEPTION_TYPES,
            )
            return None

    # L1: content_hash O(1) 去重
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    try:
        existing = conn.execute(
            "SELECT 1 FROM memories WHERE content_hash = ? AND collection = ? "
            "AND status = 'active' LIMIT 1",
            (content_hash, collection),
        ).fetchone()
        if existing:
            logger.debug("insert_memory: content_hash 重复，跳过写入")
            return None
    except sqlite3.OperationalError as e:
        if "no such column" not in str(e).lower():
            raise
        # content_hash 列不存在（旧 schema）— 不阻塞写入

    # 从 memory dict 读取 tags（不修改原始 dict）
    tags = memory.get("tags")
    tag_source = memory.get("tag_source", "auto")

    category = memory.get("category", "observation")
    decay_rate = memory.get("decay_rate", 0.01)

    # L2: safe_db_transaction 原子写
    def _do(c: sqlite3.Connection) -> int:
        # human_summary: 用户传入 > content 前 200 字
        human_summary = memory.get("human_summary") or content[:200]

        cursor = c.execute("""
            INSERT INTO memories
                (session_id, collection, type, content, human_summary, context,
                 perception_type, perception_data, perception_metadata, concept,
                 category, confidence, decay_rate, source, scope, status,
                 content_hash, media_hash,
                 embedding, scope_files, scope_entities)
            VALUES (?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, 'active',
                    ?, ?,
                    ?, ?, ?)
        """, [
            memory.get("session_id"),
            collection,
            mem_type,
            content,
            human_summary,
            memory.get("context"),
            perception_type,
            memory.get("perception_data"),
            memory.get("perception_metadata"),
            memory.get("concept"),
            category,
            memory.get("confidence", 0.9),
            decay_rate,
            memory.get("source", "tool"),
            memory.get("scope", "project"),
            content_hash,
            memory.get("media_hash"),
            memory.get("embedding"),
            memory.get("scope_files", "[]"),
            memory.get("scope_entities", "[]"),
        ])
        memory_id = cursor.lastrowid

        # FTS5 手动同步（jieba 分词 CJK 文本）
        try:
            fts_content = tokenize_for_fts5(content)
            fts_summary = tokenize_for_fts5(human_summary) if human_summary else ""
            c.execute(
                "INSERT INTO memories_fts(rowid, content, human_summary, scope_files, scope_entities) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, fts_content, fts_summary,
                 memory.get("scope_files", "[]"), memory.get("scope_entities", "[]")),
            )
        except Exception as e:
            logger.warning("FTS5 索引失败 (memory_id=%d): %s", memory_id, e)

        # Vec0 手动同步
        emb = memory.get("embedding")
        if emb and vec_loaded:
            emb_blob = emb if isinstance(emb, bytes) else floats_to_blob(emb)
            try:
                c.execute(
                    "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                    (memory_id, emb_blob),
                )
            except Exception as e:
                logger.warning("Vec 索引失败 (memory_id=%d): %s", memory_id, e)

        # 事务内写入 tags
        if tags:
            clean = [_normalize_tag(t) for t in tags if isinstance(t, str)]
            clean = [t for t in clean if t]
            if clean:
                c.executemany(
                    "INSERT OR IGNORE INTO memory_tags(memory_id, tag, source) "
                    "VALUES(?, ?, ?)",
                    [(memory_id, t, tag_source) for t in clean],
                )

        return memory_id

    ok, result = safe_db_transaction(conn, _do)

    # L3: 事后日志
    if ok and result:
        logger.debug(
            "insert_memory: id=%d type=%s category=%s collection=%s",
            result, mem_type, category, collection,
        )
    else:
        logger.warning(
            "insert_memory 失败: type=%s collection=%s content=%.50s",
            mem_type, collection, content,
        )

    return result if ok else None


@validate_args(memory_id=positive_int)
def get_memory(conn: sqlite3.Connection, memory_id: int) -> dict | None:
    """根据 ID 获取单条记忆，不存在返回 None"""
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        row = cur.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("get_memory(%d) 失败: %s", memory_id, e)
        return None


@validate_args(memory_id=positive_int)
def update_memory(conn: sqlite3.Connection, memory_id: int, **updates) -> None:
    """更新记忆字段

    FTS5 手动同步（content/human_summary/scope_files/scope_entities 变更时重建）。

    三层防御：
    - L1 事前：@validate_args(memory_id) + 白名单过滤
    - L2 事中：safe_db_transaction 原子写
    - L3 事后：logger 记录
    """
    if not updates:
        return

    allowed = {
        "content", "human_summary", "category", "confidence", "decay_rate",
        "scope_files", "scope_entities", "scope",
        "status", "context", "perception_metadata",
    }
    sets = ["updated_at=strftime('%Y-%m-%dT%H:%M:%f','now')"]
    values: list = []
    for key, val in updates.items():
        if key in allowed:
            sets.append(f"{key}=?")
            values.append(val)
    if len(sets) == 1:
        return
    values.append(memory_id)

    # 检查是否需要重建 FTS5 索引
    fts_fields = {"content", "human_summary", "scope_files", "scope_entities"}
    need_fts_rebuild = bool(fts_fields & updates.keys())

    sql = f"UPDATE memories SET {', '.join(sets)} WHERE id=?"

    def _do(c: sqlite3.Connection) -> None:
        c.execute(sql, values)

        # FTS5 手动同步（jieba 分词 CJK）
        if need_fts_rebuild:
            try:
                row = c.execute(
                    "SELECT content, human_summary, scope_files, scope_entities "
                    "FROM memories WHERE id=?", [memory_id],
                ).fetchone()
                if row:
                    from ..db import SUPPORTS_CONTENTLESS_DELETE
                    # 删除旧 FTS5 行
                    if SUPPORTS_CONTENTLESS_DELETE:
                        c.execute("DELETE FROM memories_fts WHERE rowid=?", [memory_id])
                    else:
                        # contentless FTS5 无法直接 DELETE，需要用 'delete' 命令
                        # 但我们没有旧值 — 跳过删除，FTS5 会有残留（可接受）
                        pass
                    # 插入新 FTS5 行（分词后）
                    c.execute(
                        "INSERT INTO memories_fts(rowid, content, human_summary, scope_files, scope_entities) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (memory_id,
                         tokenize_for_fts5(row[0] or ""),
                         tokenize_for_fts5(row[1] or ""),
                         row[2] or "[]", row[3] or "[]"),
                    )
            except Exception as e:
                logger.warning("FTS5 索引更新失败 (memory_id=%d): %s", memory_id, e)

    safe_db_transaction(conn, _do)


@validate_args(memory_id=positive_int)
def invalidate_memory(
    conn: sqlite3.Connection, memory_id: int, reason: str = "",
) -> None:
    """将记忆标记为失效（forget 工具）

    三层防御：
    - L1 事前：@validate_args(memory_id)
    - L2 事中：safe_db_write
    - L3 事后：logger 记录
    """

    safe_db_write(conn, """
        UPDATE memories
        SET status='invalidated', invalidated_reason=?,
            invalidated_at=strftime('%Y-%m-%dT%H:%M:%f','now'),
            updated_at=strftime('%Y-%m-%dT%H:%M:%f','now')
        WHERE id=?
    """, [reason, memory_id])


@validate_args(memory_id=positive_int)
def touch_memory(conn: sqlite3.Connection, memory_id: int) -> None:
    """更新访问计数和最后访问时间 — recall 命中时调用"""
    safe_db_write(conn, """
        UPDATE memories
        SET access_count=access_count+1,
            return_count=return_count+1,
            last_accessed=strftime('%Y-%m-%dT%H:%M:%f','now')
        WHERE id=?
    """, [memory_id])


def batch_touch_memories(conn: sqlite3.Connection, memory_ids: list[int]) -> None:
    """批量更新 access_count + return_count — 单次事务，减少锁竞争"""
    if not memory_ids:
        return
    valid = [mid for mid in memory_ids if isinstance(mid, int) and mid > 0]
    if not valid:
        return

    def _do(c: sqlite3.Connection) -> None:
        for mid in valid:
            c.execute("""
                UPDATE memories
                SET access_count=access_count+1,
                    return_count=return_count+1,
                    last_accessed=strftime('%Y-%m-%dT%H:%M:%f','now')
                WHERE id=?
            """, [mid])

    safe_db_transaction(conn, _do)


def get_session_memories(
    conn: sqlite3.Connection, session_id: str, collection: str,
) -> list[dict]:
    """获取指定会话中所有记忆，按创建时间排序 — end_session 摘要用"""
    if not session_id:
        return []
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        rows = cur.execute("""
            SELECT * FROM memories
            WHERE session_id=? AND collection=?
            ORDER BY created_at
        """, [session_id, collection]).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("get_session_memories 失败: %s", e)
        return []


def get_memories_missing_embedding(
    conn: sqlite3.Connection, collection: str, limit: int = 50,
) -> list[tuple[int, str]]:
    """获取缺失 embedding 的记忆 — (id, content)

    供后台 embedding 任务使用。
    """
    try:
        rows = conn.execute("""
            SELECT id, content FROM memories
            WHERE collection=?
              AND embedding IS NULL
              AND status='active'
            ORDER BY created_at DESC
            LIMIT ?
        """, [collection, limit]).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception as e:
        logger.warning("get_memories_missing_embedding 失败: %s", e)
        return []


@validate_args(memory_id=positive_int)
def update_memory_embedding(
    conn: sqlite3.Connection,
    memory_id: int,
    embedding: list[float],
    vec_loaded: bool = False,
) -> None:
    """更新单条记忆的 embedding + vec0 索引"""
    emb_blob = floats_to_blob(embedding)

    def _do(c: sqlite3.Connection) -> None:
        c.execute(
            "UPDATE memories SET embedding=?, updated_at=strftime('%Y-%m-%dT%H:%M:%f','now') WHERE id=?",
            [emb_blob, memory_id],
        )
        if vec_loaded:
            try:
                c.execute(
                    "INSERT OR REPLACE INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                    (memory_id, emb_blob),
                )
            except Exception as e:
                logger.warning("Vec 索引更新失败 (memory_id=%d): %s", memory_id, e)

    safe_db_transaction(conn, _do)


def apply_time_decay(
    conn: sqlite3.Connection,
    min_interval_days: float = 1.0,
) -> int:
    """批量时间衰减 — end_session 时调用

    公式：confidence_new = confidence × (1 - decay_rate) ^ days_since_last_access
    基准：last_accessed（recall 命中时刷新）— 越用越记得，不用就忘
    间隔：至少 min_interval_days 天才衰减（防高频 session 过度衰减）
    下限：confidence > 0.05 时才衰减

    三层防御：
    - L1 事前：min_interval_days > 0
    - L2 事中：safe_db_write
    - L3 事后：返回衰减行数

    Returns:
        衰减的记忆数量
    """
    # L1: 参数校验
    if min_interval_days <= 0:
        min_interval_days = 1.0

    def _do(c: sqlite3.Connection) -> int:
        c.execute("""
            UPDATE memories
            SET confidence = confidence * power(1 - decay_rate,
                    julianday('now') - julianday(COALESCE(last_accessed, created_at))),
                updated_at = strftime('%Y-%m-%dT%H:%M:%f','now')
            WHERE status = 'active'
              AND julianday('now') - julianday(COALESCE(last_accessed, created_at)) > ?
              AND confidence > 0.05
        """, [min_interval_days])
        return c.execute("SELECT changes()").fetchone()[0]

    ok, count = safe_db_transaction(conn, _do)

    if ok and count and count > 0:
        logger.info("apply_time_decay: %d 条记忆已衰减", count)
    return count if ok else 0


# --- 记忆巩固 ---

CONSOLIDATION_JACCARD_THRESHOLD = 0.50


def consolidate_session(
    conn: sqlite3.Connection,
    session_id: str,
    collection: str = "default",
) -> dict:
    """Session 级记忆巩固 — 贪心 Jaccard 聚类 + supersede

    算法：
    1. 查询可巩固记忆（排除保护类别、高 confidence、perception）
    2. 不足 3 条 → 跳过
    3. 按 category 分组，组内两两 Jaccard
    4. > 阈值 → 贪心聚类（簇内两两 > 阈值约束）
    5. 每簇选代表（confidence↑ → access_count↑ → created_at DESC）
    6. 非代表 → superseded

    Returns:
        {"merged_groups": int, "superseded_count": int,
         "compression_ratio": float, "avg_similarity": float}
    """
    empty_result = {
        "merged_groups": 0, "superseded_count": 0,
        "compression_ratio": 0.0, "avg_similarity": 0.0,
    }

    if not session_id:
        return empty_result

    # 1. 查询可巩固记忆
    try:
        cur = conn.cursor()
        cur.row_factory = sqlite3.Row
        rows = cur.execute("""
            SELECT id, content, category, confidence, access_count, created_at
            FROM memories
            WHERE session_id = ? AND collection = ? AND status = 'active'
              AND category NOT IN ('constraint', 'postmortem', 'gotcha')
              AND confidence < 0.95
              AND perception_type IS NULL
        """, [session_id, collection]).fetchall()
    except sqlite3.Error as e:
        logger.warning("consolidate_session 查询失败: %s", e)
        return empty_result

    memories = [dict(r) for r in rows]

    # 2. 不足 3 条 → 跳过
    if len(memories) < 3:
        return empty_result

    # 3. 按 category 分组
    groups: dict[str, list[dict]] = {}
    for m in memories:
        cat = m.get("category", "observation")
        groups.setdefault(cat, []).append(m)

    # 4+5. 组内贪心聚类
    all_clusters: list[list[dict]] = []
    all_similarities: list[float] = []

    for cat, mems in groups.items():
        if len(mems) < 2:
            continue

        clustered: set[int] = set()
        for i, a in enumerate(mems):
            if a["id"] in clustered:
                continue
            cluster = [a]
            for j in range(i + 1, len(mems)):
                b = mems[j]
                if b["id"] in clustered:
                    continue
                # 簇内两两约束：新成员必须与簇内所有成员相似度 > 阈值
                sim_ok = True
                for c in cluster:
                    sim = jaccard_similarity(b["content"], c["content"])
                    if sim <= CONSOLIDATION_JACCARD_THRESHOLD:
                        sim_ok = False
                        break
                    all_similarities.append(sim)
                if sim_ok:
                    cluster.append(b)
                    clustered.add(b["id"])

            if len(cluster) >= 2:
                clustered.add(a["id"])
                all_clusters.append(cluster)

    if not all_clusters:
        return empty_result

    # 6+7. 每簇选代表，非代表 → superseded
    superseded_count = 0

    def _do_supersede(c: sqlite3.Connection) -> int:
        count = 0
        for cluster in all_clusters:
            # 代表选择：confidence DESC → access_count DESC → created_at DESC
            sorted_cluster = sorted(
                cluster,
                key=lambda m: (
                    m.get("confidence", 0),
                    m.get("access_count", 0),
                    m.get("created_at", ""),
                ),
                reverse=True,
            )
            representative = sorted_cluster[0]
            for m in sorted_cluster[1:]:
                c.execute("""
                    UPDATE memories
                    SET status = 'superseded',
                        superseded_by = ?,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%f','now')
                    WHERE id = ?
                """, [representative["id"], m["id"]])
                count += 1
        return count

    ok, result = safe_db_transaction(conn, _do_supersede)
    if ok:
        superseded_count = result or 0

    total_original = len(memories)
    compression_ratio = round(
        superseded_count / total_original, 2,
    ) if total_original > 0 else 0.0
    avg_sim = round(
        sum(all_similarities) / len(all_similarities), 3,
    ) if all_similarities else 0.0

    logger.info(
        "consolidate_session: session=%s, merged=%d groups, superseded=%d, ratio=%.2f",
        session_id, len(all_clusters), superseded_count, compression_ratio,
    )

    return {
        "merged_groups": len(all_clusters),
        "superseded_count": superseded_count,
        "compression_ratio": compression_ratio,
        "avg_similarity": avg_sim,
    }
