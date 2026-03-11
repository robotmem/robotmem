"""Embedding Client — Ollama / OpenAI 兼容 / ONNX

Embedder Protocol 定义通用接口，create_embedder() 工厂函数按配置选择后端。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Embedding 后端协议"""

    @property
    def available(self) -> bool: ...

    @property
    def unavailable_reason(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed_one(self, text: str) -> list[float]: ...

    async def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float] | None]: ...

    def embed_one_sync(self, text: str) -> list[float]: ...

    def embed_batch_sync(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float] | None]: ...

    async def check_availability(self) -> bool: ...

    async def close(self) -> None: ...


def create_embedder(config) -> Embedder:
    """工厂函数 — 根据 config.embed_backend 创建 embedder

    "onnx": FastEmbedEmbedder（本地 ONNX，~5ms/query）
    "ollama": OllamaEmbedder（Ollama HTTP API）
    """
    backend = getattr(config, "embed_backend", "ollama")

    if backend == "onnx":
        from .embed_onnx import FastEmbedEmbedder
        return FastEmbedEmbedder(
            model=getattr(config, "onnx_model", "BAAI/bge-small-en-v1.5"),
            dim=getattr(config, "onnx_dim", 384),
            cache_dir=getattr(config, "fastembed_cache_dir", ""),
        )

    return OllamaEmbedder(
        model=config.embedding_model,
        ollama_url=config.ollama_url,
        dim=config.embedding_dim,
        api=config.embed_api,
    )


class OllamaEmbedder:
    """Embedding 客户端 — Ollama + OpenAI 兼容 API"""

    _MAX_RETRIES = 3
    _BACKOFF_BASE = 1.0
    _TOTAL_TIMEOUT = 30.0
    _CONCURRENT_BATCHES = 4

    def __init__(self, model: str, ollama_url: str, dim: int = 768, api: str = "ollama"):
        self._model = model
        self._ollama_url = ollama_url.rstrip("/")
        self._dim = dim
        self._api = api
        self._client = None
        self._client_lock = asyncio.Lock()
        self._available: bool | None = None
        self._unavailable_reason: str = ""

        from .resilience import ServiceCooldown
        self._cooldown = ServiceCooldown(f"ollama_embed_{model}")

    async def _get_client(self):
        """获取或懒创建 httpx 客户端"""
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self._ollama_url,
                    timeout=httpx.Timeout(connect=3.0, read=10.0, write=10.0, pool=10.0),
                    transport=httpx.AsyncHTTPTransport(
                        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                    ),
                )
            return self._client

    # ── API 适配 ──

    def _embed_endpoint(self) -> str:
        return "/v1/embeddings" if self._api == "openai_compat" else "/api/embed"

    def _embed_payload(self, input_data: str | list[str]) -> dict:
        return {"model": self._model, "input": input_data}

    def _parse_embeddings(self, data: dict) -> list[list[float]]:
        if self._api == "openai_compat":
            items = data.get("data")
            if not items or not isinstance(items, list):
                raise ValueError(f"响应缺少有效 data 数组: {str(data)[:200]}")
            if all("index" in item for item in items):
                items.sort(key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        return data["embeddings"]

    # ── 核心方法 ──

    async def embed_one(self, text: str) -> list[float]:
        """单条文本 → 向量，带指数退避重试 + 总超时"""
        import httpx

        try:
            return await asyncio.wait_for(
                self._embed_one_inner(text), timeout=self._TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            self._set_unavailable(f"embedding 总超时（{self._TOTAL_TIMEOUT}s）")
            raise httpx.ReadTimeout(f"embed_one 总超时 {self._TOTAL_TIMEOUT}s")

    async def _embed_one_inner(self, text: str) -> list[float]:
        import httpx

        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            client = await self._get_client()
            try:
                resp = await client.post(self._embed_endpoint(), json=self._embed_payload(text))
                resp.raise_for_status()
                return self._parse_embeddings(resp.json())[0]
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                wait = self._BACKOFF_BASE * (2 ** attempt)
                logger.warning("embed_one 失败 (%d/%d): %s，%.1fs 后重试",
                               attempt + 1, self._MAX_RETRIES, e, wait)
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError:
                raise
            except (KeyError, IndexError) as e:
                raise ValueError(f"embed 响应格式错误: {e}") from e
        if last_exc:
            raise last_exc
        raise RuntimeError("重试循环结束但无异常")

    async def _embed_single_batch(self, batch: list[str], batch_num: int) -> list[list[float]]:
        import httpx

        try:
            return await asyncio.wait_for(
                self._embed_single_batch_inner(batch, batch_num),
                timeout=self._TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            self._set_unavailable(f"embedding 总超时（{self._TOTAL_TIMEOUT}s）")
            raise httpx.ReadTimeout(f"embed_batch 第 {batch_num} 批总超时")

    async def _embed_single_batch_inner(self, batch: list[str], batch_num: int) -> list[list[float]]:
        import httpx

        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            client = await self._get_client()
            try:
                resp = await client.post(self._embed_endpoint(), json=self._embed_payload(batch))
                resp.raise_for_status()
                return self._parse_embeddings(resp.json())
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                wait = self._BACKOFF_BASE * (2 ** attempt)
                logger.warning("embed_batch 第 %d 批失败 (%d/%d): %s",
                               batch_num, attempt + 1, self._MAX_RETRIES, e)
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError:
                raise
            except (KeyError, IndexError) as e:
                raise ValueError(f"embed 响应格式错误: {e}") from e
        if last_exc:
            raise last_exc
        raise RuntimeError("重试循环结束但无异常")

    async def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float] | None]:
        """批量文本 → 向量列表，失败位置填 None"""
        if not texts:
            return []

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        if len(batches) == 1:
            try:
                return await self._embed_single_batch(batches[0], 0)
            except Exception as e:
                logger.error("embed_batch 全部失败（重试耗尽）: %s", e)
                return [None] * len(batches[0])

        sem = asyncio.Semaphore(self._CONCURRENT_BATCHES)

        async def _limited(batch: list[str], num: int) -> list[list[float]]:
            async with sem:
                return await self._embed_single_batch(batch, num)

        results = await asyncio.gather(
            *[_limited(b, i) for i, b in enumerate(batches)],
            return_exceptions=True,
        )

        all_embeddings: list[list[float] | None] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("embed_batch 第 %d 批失败: %s", i, result)
                all_embeddings.extend([None] * len(batches[i]))
            else:
                all_embeddings.extend(result)
        return all_embeddings

    # ── sync 桥接（SDK 用）──

    def embed_one_sync(self, text: str) -> list[float]:
        """单条文本 → 向量（同步，SDK 用）

        使用 httpx.Client 同步发请求，避免 asyncio event loop 冲突。
        """
        import httpx

        with httpx.Client(
            base_url=self._ollama_url,
            timeout=httpx.Timeout(connect=3.0, read=10.0, write=10.0, pool=10.0),
        ) as client:
            last_exc: Exception | None = None
            for attempt in range(self._MAX_RETRIES):
                try:
                    resp = client.post(self._embed_endpoint(), json=self._embed_payload(text))
                    resp.raise_for_status()
                    return self._parse_embeddings(resp.json())[0]
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    last_exc = e
                    import time
                    wait = self._BACKOFF_BASE * (2 ** attempt)
                    logger.warning("embed_one_sync 失败 (%d/%d): %s", attempt + 1, self._MAX_RETRIES, e)
                    time.sleep(wait)
                except httpx.HTTPStatusError:
                    raise
                except (KeyError, IndexError) as e:
                    raise ValueError(f"embed 响应格式错误: {e}") from e
            if last_exc:
                raise last_exc
            raise RuntimeError("重试循环结束但无异常")

    def embed_batch_sync(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float] | None]:
        """批量文本 → 向量列表（同步，SDK 用）"""
        if not texts:
            return []
        import httpx

        all_embeddings: list[list[float] | None] = []
        with httpx.Client(
            base_url=self._ollama_url,
            timeout=httpx.Timeout(connect=3.0, read=10.0, write=10.0, pool=10.0),
        ) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    resp = client.post(self._embed_endpoint(), json=self._embed_payload(batch))
                    resp.raise_for_status()
                    all_embeddings.extend(self._parse_embeddings(resp.json()))
                except Exception as e:
                    logger.warning("embed_batch_sync 第 %d 批失败: %s", i // batch_size, e)
                    all_embeddings.extend([None] * len(batch))
        return all_embeddings

    # ── 可用性检测（合并 ollama_check.py）──

    async def check_availability(self) -> bool:
        """检测 embedding 服务是否可用"""
        if self._cooldown.is_cooling:
            return False

        client = await self._get_client()

        if self._api == "openai_compat":
            return await self._check_openai_compat(client)
        return await self._check_ollama(client)

    async def _check_ollama(self, client) -> bool:
        """Ollama 模式：身份验证 → 模型检查 → embed 测试"""
        import httpx

        # 身份验证
        try:
            resp = await client.get("/api/version", timeout=3.0)
            resp.raise_for_status()
            if "version" not in resp.json():
                self._set_unavailable(f"端口 {self._ollama_url} 非 Ollama 服务，请检查端口冲突")
                return False
        except (httpx.ConnectError, httpx.TimeoutException):
            self._set_unavailable("Ollama 未启动，请运行: ollama serve")
            return False
        except (httpx.HTTPError, ValueError):
            self._set_unavailable(f"端口 {self._ollama_url} 疑似非 Ollama 服务")
            return False

        # 模型检查
        try:
            resp = await client.get("/api/tags", timeout=3.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            names = [m.get("name", "") for m in models]
            found = any(n == self._model or n.startswith(f"{self._model}:") for n in names)
            if not found:
                self._set_unavailable(f"模型 {self._model} 未下载，请运行: ollama pull {self._model}")
                return False
        except httpx.HTTPError as e:
            self._set_unavailable(f"Ollama 检测异常: {e}")
            return False

        # Embed 测试
        try:
            resp = await client.post("/api/embed", json={"model": self._model, "input": "ping"}, timeout=15.0)
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings", [])
            if not embeddings or not embeddings[0]:
                self._set_unavailable(f"模型 {self._model} 返回空向量")
                return False
        except httpx.TimeoutException:
            self._set_unavailable(f"模型 {self._model} 加载超时，可能内存不足")
            return False
        except httpx.HTTPError as e:
            self._set_unavailable(f"Ollama embed 测试失败: {e}")
            return False

        self._set_available()
        return True

    async def _check_openai_compat(self, client) -> bool:
        """OpenAI 兼容模式：直接 embed 短文本测试"""
        import httpx

        try:
            resp = await client.post("/v1/embeddings", json={"model": self._model, "input": "ping"}, timeout=15.0)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data or not data[0].get("embedding"):
                self._set_unavailable(f"模型 {self._model} 返回空向量")
                return False
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            self._set_unavailable(f"Embedding 服务不可用 ({self._ollama_url}): {e}")
            return False
        except httpx.HTTPError as e:
            self._set_unavailable(f"Embedding 服务异常: {e}")
            return False

        self._set_available()
        return True

    def _set_unavailable(self, reason: str) -> None:
        self._available = False
        self._unavailable_reason = reason
        self._cooldown.record_failure()

    def _set_available(self) -> None:
        self._available = True
        self._unavailable_reason = ""
        self._cooldown.record_success()

    def reset_cooldown(self) -> None:
        self._available = None
        self._cooldown.reset()

    @property
    def available(self) -> bool:
        return self._available is True

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def close(self) -> None:
        async with self._client_lock:
            if self._client:
                try:
                    await self._client.aclose()
                except Exception as e:
                    logger.warning("httpx client 关闭异常: %s", e)
                finally:
                    self._client = None
