"""robotmem Python SDK — 实例化接口

RobotMemory 类：实例持有 config/db/embedder，全 sync API。
和 api.py（全局单例）的区别：实例化、可多实例共存、异常可预测。

用法:
    from robotmem.sdk import RobotMemory

    mem = RobotMemory()                      # 零配置，ONNX 默认
    mem.learn("force 0.8N works for push")
    tips = mem.recall("push technique", n=5)
    mem.close()

    # 上下文管理器
    with RobotMemory(db_path=":memory:") as mem:
        mem.learn("test insight")
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

from .db import floats_to_blob
from .auto_classify import (
    build_context_json,
    classify_category,
    classify_tags,
    estimate_confidence,
    extract_scope,
    normalize_scope_files,
)
from .config import Config
from .db_cog import CogDatabase
from .dedup import check_duplicate
from .embed import Embedder, create_embedder
from .exceptions import DatabaseError, ValidationError
from .ops.memories import (
    apply_time_decay,
    consolidate_session as do_consolidate,
    get_memory,
    insert_memory,
    invalidate_memory,
    update_memory,
    update_memory_embedding,
)
from .ops.sessions import (
    get_or_create_session,
    get_session_summary,
    insert_session_outcome,
    mark_session_ended,
    update_session_context,
)
from .ops.tags import add_tags
from .search import recall_sync
from .validators import (
    EndSessionParams,
    ForgetParams,
    LearnParams,
    RecallParams,
    SavePerceptionParams,
    StartSessionParams,
    UpdateParams,
    parse_params,
)

logger = logging.getLogger(__name__)


class RobotMemory:
    """robotmem Python SDK — 实例化同步接口

    三层防御策略（和 MCP Server 共用 L1/L2 原语，L3 不同）：
    - L1 事前：parse_params() Pydantic 校验 → 失败抛 ValidationError
    - L2 事中：safe_db_write / dedup / embed → DB 失败抛 DatabaseError
    - L2 事中：embedding 失败 → 静默降级（增强不是必须，宪法第 4 条）
    - L2 事中：重复 → 返回 {"status": "duplicate"}（不是异常）
    - L3 事后：无兜底装饰器，异常直接传播给用户
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        collection: str = "default",
        embed_backend: str = "onnx",
    ):
        """初始化 RobotMemory 实例

        Args:
            db_path: 数据库路径。None = ~/.robotmem/memory.db，":memory:" = 内存库
            collection: 默认 collection 名
            embed_backend: "onnx"（默认，本地 ~5ms）| "ollama" | "none"（纯 BM25）
        """
        self._collection = collection
        self._closed = False

        # 构建 Config
        self._config = self._build_config(db_path, embed_backend)

        # 初始化 DB（触发连接 + schema）
        self._db = CogDatabase(self._config)
        try:
            _ = self._db.conn
        except Exception as e:
            raise DatabaseError(f"数据库初始化失败: {e}") from e

        # 初始化 Embedder（延迟到首次 learn/recall 时才加载模型）
        if embed_backend == "none":
            self._embedder: Embedder | None = None
        else:
            self._embedder = create_embedder(self._config)

        self._owns_resources = True  # 自己创建的 db/embedder，close 时释放

    @classmethod
    def _from_components(
        cls,
        db: CogDatabase,
        embedder: Embedder | None,
        collection: str = "default",
    ) -> RobotMemory:
        """内部工厂方法 — MCP Server 复用已有 db/embedder

        MCP Server 在 lifespan 中创建 db_cog 和 embedder，
        SDK 实例复用这些组件，不创建新连接。close() 不释放共享资源。
        """
        instance = object.__new__(cls)
        instance._collection = collection
        instance._closed = False
        instance._config = db._config
        instance._db = db
        instance._embedder = embedder
        instance._owns_resources = False  # MCP 管理生命周期
        return instance

    @staticmethod
    def _build_config(
        db_path: str | Path | None,
        embed_backend: str,
    ) -> Config:
        """从构造参数生成 Config"""
        kwargs: dict = {}

        if db_path is not None:
            kwargs["db_path"] = str(db_path)

        if embed_backend == "none":
            # Config 校验只接受 "onnx"/"ollama"，"none" 用 "onnx" 占位
            # 实际后端由构造函数中 self._embedder = None 控制
            kwargs["embed_backend"] = "onnx"
        else:
            kwargs["embed_backend"] = embed_backend

        return Config(**kwargs)

    def _ensure_open(self) -> None:
        """检查实例未关闭"""
        if self._closed:
            raise DatabaseError("RobotMemory 已关闭")

    def _ensure_embedder(self) -> None:
        """首次使用时检测 embedder 可用性（sync，不阻塞构造函数）"""
        if self._embedder is None:
            return
        if self._embedder.available:
            return
        # 首次调用：尝试初始化 ONNX 模型
        if hasattr(self._embedder, "_ensure_encoder"):
            try:
                self._embedder._ensure_encoder()
            except Exception as e:
                logger.warning("SDK embedder 初始化失败: %s，降级为 BM25-only", e)

    # ── 核心 API ──

    def learn(
        self,
        insight: str,
        context: str | dict = "",
        session_id: str | None = None,
        collection: str | None = None,
    ) -> dict:
        """记录物理经验

        三层防御：
        - L1 事前：insight 非空 + Pydantic 校验
        - L2 事中：auto_classify + dedup + embed + 原子写入
        - L3 事后：返回 {"status": "created", "memory_id": ...}

        Args:
            insight: 学到的经验（必填）
            context: 上下文（str 或 dict，dict 会自动 JSON 序列化）
            session_id: 关联的 session ID
            collection: 覆盖实例默认 collection（MCP Server 用）

        Returns:
            {"status": "created", "memory_id": ..., "auto_inferred": {...}}
            {"status": "duplicate", "method": ..., "existing_id": ..., "similarity": ...}

        Raises:
            ValidationError: 参数校验失败
            DatabaseError: 数据库写入失败
        """
        self._ensure_open()

        coll = collection if collection and collection.strip() else self._collection

        # context dict → str
        if isinstance(context, dict):
            context = json.dumps(context, ensure_ascii=False)

        # L1: Pydantic 校验
        result = parse_params(
            LearnParams, insight=insight, context=context,
            collection=coll, session_id=session_id,
        )
        if isinstance(result, dict):
            raise ValidationError(result.get("error", "参数校验失败"))
        params = result

        # L2: auto_classify — 每步 try-except 降级
        try:
            category = classify_category(params.insight)
        except Exception as e:
            logger.debug("learn classify_category 降级: %s", e)
            category = "observation"

        try:
            confidence = estimate_confidence(params.insight, params.context)
        except Exception as e:
            logger.debug("learn estimate_confidence 降级: %s", e)
            confidence = 0.9

        try:
            scope = extract_scope(params.insight)
            scope_files = normalize_scope_files(scope.get("scope_files", []))
            scope_entities = scope.get("scope_entities", [])
        except Exception as e:
            logger.debug("learn extract_scope 降级: %s", e)
            scope_files, scope_entities = [], []

        try:
            inferred_tags = classify_tags(params.insight, params.context)
        except Exception as e:
            logger.debug("learn classify_tags 降级: %s", e)
            inferred_tags = []

        try:
            ctx_json = build_context_json(params.insight, params.context)
        except Exception as e:
            logger.debug("learn build_context_json 降级: %s", e)
            ctx_json = params.context

        # L2: 去重
        try:
            dedup_result = check_duplicate(
                params.insight, coll, params.session_id,
                self._db, self._embedder if self._embedder and self._embedder.available else None,
            )
            if dedup_result.is_dup:
                existing_id = (
                    dedup_result.similar_facts[0].get("id")
                    if dedup_result.similar_facts else None
                )
                return {
                    "status": "duplicate",
                    "method": dedup_result.method,
                    "existing_id": existing_id,
                    "similarity": dedup_result.similarity,
                }
        except Exception as e:
            logger.warning("learn 去重检查异常: %s", e)

        # L2: embedding（降级为 None）— embed_one_sync 返回 list[float]，转为 blob
        self._ensure_embedder()
        embedding = None
        if self._embedder and self._embedder.available:
            try:
                emb_list = self._embedder.embed_one_sync(params.insight)
                embedding = floats_to_blob(emb_list) if emb_list else None
            except Exception as e:
                logger.warning("learn embedding 降级: %s", e)

        # L2: 原子写入
        memory_id = insert_memory(self._db.conn, {
            "session_id": params.session_id,
            "collection": coll,
            "type": "fact",
            "content": params.insight,
            "human_summary": params.insight[:200],
            "context": ctx_json if isinstance(ctx_json, str) else json.dumps(ctx_json),
            "category": category,
            "confidence": confidence,
            "source": "sdk",
            "scope": "project",
            "scope_files": json.dumps(scope_files),
            "scope_entities": json.dumps(scope_entities),
            "embedding": embedding,
            "tags": inferred_tags,
            "tag_source": "auto",
        }, vec_loaded=self._db.vec_loaded)

        if not memory_id:
            raise DatabaseError("写入失败")

        # L3: 返回
        return {
            "status": "created",
            "memory_id": memory_id,
            "auto_inferred": {
                "category": category,
                "confidence": confidence,
                "tags": inferred_tags,
                "scope_files": scope_files,
            },
        }

    def recall(
        self,
        query: str,
        n: int = 5,
        min_confidence: float = 0.3,
        session_id: str | None = None,
        context_filter: dict | None = None,
        spatial_sort: dict | None = None,
        collection: str | None = None,
    ) -> list[dict]:
        """检索经验 — BM25 + Vec 混合搜索

        Args:
            query: 搜索查询
            n: 返回条数（1~100）
            min_confidence: 最低置信度
            session_id: 限定 session
            context_filter: 结构化过滤，如 {"task.success": True}
            spatial_sort: 空间排序，如 {"field": "spatial.position", "target": [1.3, 0.7]}
            collection: 覆盖实例默认 collection（MCP Server 用）

        Returns:
            list[dict] — 每个 dict 包含 content, confidence, _rrf_score 等字段

        Raises:
            ValidationError: 参数校验失败
        """
        self._ensure_open()

        coll = collection if collection and collection.strip() else self._collection

        # L1: Pydantic 校验（SDK 接受 dict，MCP 接受 JSON 字符串）
        result = parse_params(
            RecallParams, query=query, collection=coll,
            n=n, min_confidence=min_confidence, session_id=session_id,
            context_filter=json.dumps(context_filter) if context_filter else None,
            spatial_sort=json.dumps(spatial_sort) if spatial_sort else None,
        )
        if isinstance(result, dict):
            raise ValidationError(result.get("error", "参数校验失败"))
        params = result

        # L2: context_filter 已经是 dict，直接传
        # （MCP 需要 JSON 解析，SDK 直接接受 dict）

        # L2: 搜索
        self._ensure_embedder()
        recall_result = recall_sync(
            query=params.query,
            db=self._db,
            embedder=self._embedder if self._embedder and self._embedder.available else None,
            collection=coll,
            top_k=params.n,
            min_confidence=params.min_confidence,
            session_id=params.session_id,
            context_filter=context_filter,
            spatial_sort=spatial_sort,
        )

        return recall_result.memories

    def save_perception(
        self,
        description: str,
        perception_type: str = "visual",
        data: str | None = None,
        metadata: str | None = None,
        session_id: str | None = None,
        collection: str | None = None,
    ) -> dict:
        """保存感知/轨迹/力矩（procedural memory）

        三层防御：
        - L1 事前：description 非空 + perception_type 白名单
        - L2 事中：embedding + 原子写入
        - L3 事后：返回 memory_id

        Args:
            description: 感知描述（至少 5 字符）
            perception_type: "visual"|"tactile"|"auditory"|"proprioceptive"|"procedural"
            data: 原始数据（JSON 字符串）
            metadata: 元数据（JSON 字符串）
            session_id: 关联的 session ID
            collection: 覆盖实例默认 collection

        Returns:
            {"memory_id": ..., "perception_type": ..., "collection": ..., "has_embedding": ...}

        Raises:
            ValidationError: 参数校验失败
            DatabaseError: 数据库写入失败
        """
        self._ensure_open()

        coll = collection if collection and collection.strip() else self._collection

        # L1: Pydantic 校验
        result = parse_params(
            SavePerceptionParams, description=description,
            perception_type=perception_type, data=data, metadata=metadata,
            collection=coll, session_id=session_id,
        )
        if isinstance(result, dict):
            raise ValidationError(result.get("error", "参数校验失败"))
        params = result

        # L2: embedding（降级为 None）— embed_one_sync 返回 list[float]，转为 blob
        self._ensure_embedder()
        embedding = None
        if self._embedder and self._embedder.available:
            try:
                emb_list = self._embedder.embed_one_sync(params.description)
                embedding = floats_to_blob(emb_list) if emb_list else None
            except Exception as e:
                logger.warning("save_perception embedding 降级: %s", e)

        # L2: 原子写入
        memory_id = insert_memory(self._db.conn, {
            "session_id": params.session_id,
            "collection": coll,
            "type": "perception",
            "content": params.description,
            "human_summary": params.description[:200],
            "perception_type": params.perception_type,
            "perception_data": params.data,
            "perception_metadata": params.metadata,
            "category": "observation",
            "confidence": 0.9,
            "source": "sdk",
            "scope": "project",
            "embedding": embedding,
        }, vec_loaded=self._db.vec_loaded)

        if not memory_id:
            raise DatabaseError("写入失败")

        # L3: 返回
        return {
            "memory_id": memory_id,
            "perception_type": params.perception_type,
            "collection": coll,
            "has_embedding": embedding is not None,
        }

    def forget(self, memory_id: int, reason: str) -> dict:
        """删除错误记忆（软删除）

        三层防御：
        - L1 事前：memory_id 正整数 + reason 非空
        - L2 事中：归属校验 + invalidate
        - L3 事后：返回确认

        Args:
            memory_id: 要删除的记忆 ID
            reason: 删除原因

        Returns:
            {"status": "forgotten", "memory_id": ..., "content": ..., "reason": ...}

        Raises:
            ValidationError: 参数校验失败 / 记忆不存在 / 状态不允许
        """
        self._ensure_open()

        # L1: Pydantic 校验
        result = parse_params(ForgetParams, memory_id=memory_id, reason=reason)
        if isinstance(result, dict):
            raise ValidationError(result.get("error", "参数校验失败"))
        params = result

        # L2: 归属校验
        mem = get_memory(self._db.conn, params.memory_id)
        if not mem:
            raise ValidationError(f"记忆 #{params.memory_id} 不存在")
        if mem.get("status") != "active":
            raise ValidationError(
                f"记忆 #{params.memory_id} 状态为 {mem.get('status')}，无法删除"
            )

        # L2: 软删除
        try:
            invalidate_memory(self._db.conn, params.memory_id, params.reason)
        except Exception as e:
            raise DatabaseError(f"删除记忆 #{params.memory_id} 失败: {e}") from e

        # L3: 返回
        return {
            "status": "forgotten",
            "memory_id": params.memory_id,
            "content": (mem.get("content") or "")[:100],
            "reason": params.reason,
        }

    def update(
        self,
        memory_id: int,
        new_content: str,
        context: str = "",
    ) -> dict:
        """修正记忆内容

        三层防御：
        - L1 事前：memory_id 正整数 + new_content 非空
        - L2 事中：归属校验 + auto_classify + 原子更新 + 重建 embedding/tags
        - L3 事后：返回 old/new 对照

        Args:
            memory_id: 要更新的记忆 ID
            new_content: 新内容
            context: 上下文（用于重新分类）

        Returns:
            {"status": "updated", "memory_id": ..., "old_content": ..., "new_content": ..., "auto_inferred": {...}}

        Raises:
            ValidationError: 参数校验失败 / 记忆不存在 / 状态不允许
        """
        self._ensure_open()

        # L1: Pydantic 校验
        result = parse_params(
            UpdateParams, memory_id=memory_id,
            new_content=new_content, context=context,
        )
        if isinstance(result, dict):
            raise ValidationError(result.get("error", "参数校验失败"))
        params = result

        # L2: 归属校验
        mem = get_memory(self._db.conn, params.memory_id)
        if not mem:
            raise ValidationError(f"记忆 #{params.memory_id} 不存在")
        if mem.get("status") != "active":
            raise ValidationError(
                f"记忆 #{params.memory_id} 状态为 {mem.get('status')}，无法更新"
            )

        old_content = mem.get("content", "")

        # L2: 重新分类
        try:
            category = classify_category(params.new_content)
            confidence = estimate_confidence(params.new_content, params.context)
        except Exception:
            category = mem.get("category", "observation")
            confidence = mem.get("confidence", 0.9)

        # L2: 更新
        update_memory(
            self._db.conn, params.memory_id,
            content=params.new_content,
            category=category,
            confidence=confidence,
        )

        # L2: 重建 embedding（降级跳过）
        self._ensure_embedder()
        if self._embedder and self._embedder.available:
            try:
                new_emb = self._embedder.embed_one_sync(params.new_content)
                update_memory_embedding(
                    self._db.conn, params.memory_id, new_emb,
                    vec_loaded=self._db.vec_loaded,
                )
            except Exception as e:
                logger.warning("update embedding 重建失败: %s", e)

        # L2: 重建 tags
        try:
            inferred_tags = classify_tags(params.new_content, params.context)
            if inferred_tags:
                add_tags(self._db.conn, params.memory_id, inferred_tags, source="auto")
        except Exception as e:
            logger.warning("update tags 重建失败: %s", e)

        # L3: 返回
        return {
            "status": "updated",
            "memory_id": params.memory_id,
            "old_content": old_content[:100],
            "new_content": params.new_content[:100],
            "auto_inferred": {
                "category": category,
                "confidence": confidence,
            },
        }

    def batch_learn(self, insights: list[str | dict]) -> list[dict]:
        """批量 learn — 逐条调用 learn()，单条失败不影响其他

        Args:
            insights: 每个元素为 str（纯 insight）或 dict（learn() 参数）

        Returns:
            list[dict] — 每条对应一个 learn() 返回值或 {"status": "error", "error": "..."}
        """
        self._ensure_open()
        results: list[dict] = []
        for item in insights:
            if isinstance(item, str):
                item = {"insight": item}
            try:
                result = self.learn(**item)
                results.append(result)
            except Exception as e:
                results.append({"status": "error", "error": str(e)})
        return results

    # ── 上下文管理器 ──

    @contextmanager
    def session(self, context: str | dict | None = None):
        """Session 上下文管理器 — 自动 start/end

        用法:
            with mem.session(context={"task": "push"}) as sid:
                mem.learn("...", session_id=sid)

        Yields:
            str: session_id
        """
        sid = self.start_session(context=context)
        try:
            yield sid
        finally:
            try:
                self.end_session(session_id=sid)
            except Exception as e:
                logger.warning("session %s 自动关闭失败: %s", sid, e)

    def start_session(
        self,
        context: str | dict | None = None,
        collection: str | None = None,
    ) -> str:
        """开始新 session — 返回 session_id

        Args:
            context: session 上下文（str 或 dict）
            collection: 覆盖实例默认 collection
        """
        self._ensure_open()

        import uuid

        coll = collection if collection and collection.strip() else self._collection

        # context dict → str
        ctx_str = None
        if isinstance(context, dict):
            ctx_str = json.dumps(context, ensure_ascii=False)
        elif isinstance(context, str):
            ctx_str = context

        result = parse_params(
            StartSessionParams, collection=coll, context=ctx_str,
        )
        if isinstance(result, dict):
            raise ValidationError(result.get("error", "参数校验失败"))
        params = result

        ext_id = str(uuid.uuid4())
        session = get_or_create_session(self._db.conn, ext_id, coll)
        if not session:
            raise DatabaseError("创建 session 失败")

        if params.context:
            update_session_context(self._db.conn, ext_id, params.context)

        return ext_id

    def end_session(
        self,
        session_id: str,
        outcome_score: float | None = None,
    ) -> dict:
        """结束 session — 标记结束 + 时间衰减 + 巩固 + 评分"""
        self._ensure_open()

        result = parse_params(
            EndSessionParams, session_id=session_id, outcome_score=outcome_score,
        )
        if isinstance(result, dict):
            raise ValidationError(result.get("error", "参数校验失败"))
        params = result

        # 查询 session 关联的 collection
        try:
            row = self._db.conn.execute(
                "SELECT collection FROM sessions WHERE external_id=?",
                (params.session_id,),
            ).fetchone()
            coll = row[0] if row else self._collection
        except Exception as e:
            logger.warning("end_session 查 collection 失败: %s，fallback 到 %s", e, self._collection)
            coll = self._collection

        # 标记结束
        mark_session_ended(self._db.conn, params.session_id)

        # 时间衰减
        decayed = 0
        try:
            decayed = apply_time_decay(self._db.conn)
        except Exception as e:
            logger.warning("end_session time_decay 失败: %s", e)

        # 记忆巩固
        consolidated = {"merged_groups": 0, "superseded_count": 0}
        try:
            consolidated = do_consolidate(self._db.conn, params.session_id, coll)
        except Exception as e:
            logger.warning("consolidate_session 失败: %s", e)

        # proactive recall（sync 版）
        related: list[dict] = []
        try:
            top_row = self._db.conn.execute(
                "SELECT content FROM memories "
                "WHERE session_id=? AND collection=? AND status='active' "
                "ORDER BY created_at DESC LIMIT 1",
                [params.session_id, coll],
            ).fetchone()
            top_content = top_row[0] if top_row else ""
            if top_content:
                pr_result = recall_sync(
                    query=top_content,
                    db=self._db,
                    embedder=self._embedder if self._embedder and self._embedder.available else None,
                    collection=coll,
                    top_k=5,
                )
                related = [
                    m for m in pr_result.memories
                    if m.get("session_id") != params.session_id
                ][:5]
        except Exception as e:
            logger.warning("proactive recall 失败: %s", e)

        # 记录评分
        if params.outcome_score is not None:
            try:
                insert_session_outcome(self._db.conn, params.session_id, params.outcome_score)
            except Exception as e:
                logger.warning("end_session outcome 写入失败: %s", e)

        # 返回摘要
        summary = get_session_summary(self._db.conn, params.session_id, coll)

        return {
            "status": "ended",
            "session_id": params.session_id,
            "summary": summary,
            "decayed_count": decayed,
            "consolidated": consolidated,
            "related_memories": related,
        }

    # ── 生命周期 ──

    def close(self) -> None:
        """释放 DB 连接和 embedder 资源

        _from_components() 创建的实例不释放共享资源（MCP 管理生命周期）。
        """
        if self._closed:
            return
        self._closed = True
        if self._owns_resources:
            self._db.close()

    def __enter__(self) -> RobotMemory:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"RobotMemory(collection={self._collection!r}, "
            f"embed={'none' if self._embedder is None else self._config.embed_backend}, "
            f"closed={self._closed})"
        )
