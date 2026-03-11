"""ONNX Embedding Client — fastembed 本地推理

零外部服务依赖，~5ms/query，自动下载模型到 ~/.cache/fastembed/。
默认模型：BAAI/bge-small-en-v1.5（384 维，67MB，MTEB retrieval 51.68）。
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class FastEmbedEmbedder:
    """基于 fastembed 的本地 ONNX embedding 后端

    fastembed（by Qdrant）：108KB wheel，无 PyTorch 依赖，30+ 预训练模型。
    模型首次使用时自动下载到 ~/.cache/fastembed/。
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384, cache_dir: str = ""):
        self._model_name = model
        self._dim = dim
        self._cache_dir = cache_dir
        self._available: bool | None = None
        self._unavailable_reason: str = ""
        self._encoder = None  # lazy init
        self._init_lock = threading.Lock()

    def _ensure_encoder(self):
        """延迟初始化 encoder（首次调用时加载模型），线程安全"""
        if self._encoder is not None:
            return
        with self._init_lock:
            if self._encoder is not None:
                return
            try:
                from fastembed import TextEmbedding
                kwargs: dict = {"model_name": self._model_name}
                if self._cache_dir:
                    kwargs["cache_dir"] = self._cache_dir
                self._encoder = TextEmbedding(**kwargs)
                self._available = True
                self._unavailable_reason = ""
                logger.info("FastEmbed 模型已加载: %s (%dd)", self._model_name, self._dim)
            except ImportError:
                self._available = False
                self._unavailable_reason = "fastembed 未安装，请运行: pip install fastembed"
                raise
            except Exception as e:
                self._available = False
                self._unavailable_reason = f"FastEmbed 模型加载失败: {e}"
                raise

    # ── sync 方法（SDK 用，原生同步，不需要 asyncio 桥接）──

    def embed_one_sync(self, text: str) -> list[float]:
        """单条文本 → 向量（同步，SDK 用）"""
        self._ensure_encoder()
        try:
            embeddings = list(self._encoder.embed([text]))
            return embeddings[0].tolist()
        except Exception as e:
            logger.error("FastEmbed embed_one_sync 失败: %s", e)
            raise RuntimeError("ONNX embedding 失败") from e

    def embed_batch_sync(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float] | None]:
        """批量文本 → 向量列表（同步，SDK 用）"""
        if not texts:
            return []
        try:
            self._ensure_encoder()
            embeddings = list(self._encoder.embed(texts, batch_size=batch_size))
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.warning("FastEmbed embed_batch_sync 失败: %s，返回 None 填充", e)
            return [None] * len(texts)

    # ── async 方法（MCP Server 用）──

    async def embed_one(self, text: str) -> list[float]:
        """单条文本 → 向量"""
        import asyncio

        self._ensure_encoder()
        loop = asyncio.get_running_loop()
        try:
            embeddings = await loop.run_in_executor(
                None, lambda: list(self._encoder.embed([text]))
            )
            return embeddings[0].tolist()
        except Exception as e:
            logger.error("FastEmbed embed_one 失败: %s", e)
            raise RuntimeError("ONNX embedding 失败") from e

    async def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float] | None]:
        """批量文本 → 向量列表"""
        if not texts:
            return []
        try:
            import asyncio

            self._ensure_encoder()
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: list(self._encoder.embed(texts, batch_size=batch_size))
            )
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.warning("FastEmbed embed_batch 失败: %s，返回 None 填充", e)
            return [None] * len(texts)

    async def check_availability(self) -> bool:
        """检测 ONNX 后端可用性"""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            self._ensure_encoder()
            test_result = await loop.run_in_executor(
                None, lambda: list(self._encoder.embed(["ping"]))
            )
            if test_result and len(test_result[0]) == self._dim:
                self._available = True
                self._unavailable_reason = ""
                return True
            elif test_result:
                actual_dim = len(test_result[0])
                self._available = False
                self._unavailable_reason = (
                    f"维度不匹配: 配置 {self._dim}, 模型实际 {actual_dim}。"
                    "请更新 onnx_dim 配置。"
                )
                return False
            else:
                self._available = False
                self._unavailable_reason = "FastEmbed 返回空向量"
                return False
        except ImportError:
            self._available = False
            self._unavailable_reason = "fastembed 未安装，请运行: pip install fastembed"
            return False
        except Exception as e:
            self._available = False
            self._unavailable_reason = f"FastEmbed 不可用: {e}"
            return False

    @property
    def available(self) -> bool:
        return self._available is True

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    async def close(self) -> None:
        self._encoder = None
