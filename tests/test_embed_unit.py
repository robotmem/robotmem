"""embed.py + embed_onnx.py 单元测试 — mock 外部依赖"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ── OllamaEmbedder 测试 ──

class TestOllamaEmbedder:
    """OllamaEmbedder 单元测试（mock httpx）"""

    def _make_embedder(self, api="ollama"):
        from robotmem.embed import OllamaEmbedder
        return OllamaEmbedder(model="bge-m3", ollama_url="http://localhost:11434", dim=768, api=api)

    def test_init(self):
        e = self._make_embedder()
        assert e.model == "bge-m3"
        assert e.dim == 768
        assert e.available is False  # 未检查前
        assert e.unavailable_reason == ""

    def test_properties(self):
        e = self._make_embedder()
        assert e.model == "bge-m3"
        assert e.dim == 768
        assert e.available is False
        assert e.unavailable_reason == ""

    def test_embed_endpoint_ollama(self):
        e = self._make_embedder(api="ollama")
        assert e._embed_endpoint() == "/api/embed"

    def test_embed_endpoint_openai(self):
        e = self._make_embedder(api="openai_compat")
        assert e._embed_endpoint() == "/v1/embeddings"

    def test_embed_payload(self):
        e = self._make_embedder()
        payload = e._embed_payload("hello")
        assert payload == {"model": "bge-m3", "input": "hello"}

    def test_embed_payload_batch(self):
        e = self._make_embedder()
        payload = e._embed_payload(["a", "b"])
        assert payload == {"model": "bge-m3", "input": ["a", "b"]}

    def test_parse_embeddings_ollama(self):
        e = self._make_embedder(api="ollama")
        result = e._parse_embeddings({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_parse_embeddings_openai(self):
        e = self._make_embedder(api="openai_compat")
        data = {"data": [{"index": 1, "embedding": [0.3]}, {"index": 0, "embedding": [0.1]}]}
        result = e._parse_embeddings(data)
        assert result == [[0.1], [0.3]]  # sorted by index

    def test_parse_embeddings_openai_no_index(self):
        e = self._make_embedder(api="openai_compat")
        data = {"data": [{"embedding": [0.1]}, {"embedding": [0.2]}]}
        result = e._parse_embeddings(data)
        assert result == [[0.1], [0.2]]

    def test_parse_embeddings_openai_invalid(self):
        e = self._make_embedder(api="openai_compat")
        with pytest.raises(ValueError, match="响应缺少有效"):
            e._parse_embeddings({"data": None})

    @pytest.mark.asyncio
    async def test_embed_one_success(self):
        e = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        e._client = mock_client

        result = await e.embed_one("test")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_one_retry_then_success(self):
        import httpx
        e = self._make_embedder()

        good_resp = MagicMock()
        good_resp.json.return_value = {"embeddings": [[0.5]]}
        good_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[
            httpx.ConnectError("fail"),
            good_resp,
        ])
        e._client = mock_client
        e._BACKOFF_BASE = 0.01  # 加速测试

        result = await e.embed_one("test")
        assert result == [0.5]

    @pytest.mark.asyncio
    async def test_embed_one_key_error(self):
        e = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"wrong_key": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        e._client = mock_client

        with pytest.raises(ValueError, match="响应格式错误"):
            await e.embed_one("test")

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        e = self._make_embedder()
        result = await e.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_single(self):
        e = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1], [0.2]]}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        e._client = mock_client

        result = await e.embed_batch(["a", "b"], batch_size=32)
        assert result == [[0.1], [0.2]]

    @pytest.mark.asyncio
    async def test_embed_batch_single_failure(self):
        e = self._make_embedder()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("fail"))
        e._client = mock_client
        e._BACKOFF_BASE = 0.01

        result = await e.embed_batch(["a", "b"], batch_size=32)
        assert result == [None, None]

    @pytest.mark.asyncio
    async def test_embed_batch_multi_partial_failure(self):
        e = self._make_embedder()

        good_resp = MagicMock()
        good_resp.json.return_value = {"embeddings": [[0.1]]}
        good_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        # batch_size=1 会产生 2 个 batch
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # 第一批 3 次重试全失败
                raise Exception("fail")
            return good_resp

        mock_client.post = mock_post
        e._client = mock_client
        e._BACKOFF_BASE = 0.01

        result = await e.embed_batch(["a", "b"], batch_size=1)
        assert len(result) == 2
        # 至少一个是 None（失败的 batch）
        assert None in result or [0.1] in result

    @pytest.mark.asyncio
    async def test_check_availability_ollama_success(self):
        e = self._make_embedder(api="ollama")

        mock_client = AsyncMock()

        # /api/version
        version_resp = MagicMock()
        version_resp.json.return_value = {"version": "0.1.0"}
        version_resp.raise_for_status = MagicMock()

        # /api/tags
        tags_resp = MagicMock()
        tags_resp.json.return_value = {"models": [{"name": "bge-m3:latest"}]}
        tags_resp.raise_for_status = MagicMock()

        # /api/embed
        embed_resp = MagicMock()
        embed_resp.json.return_value = {"embeddings": [[0.1, 0.2]]}
        embed_resp.raise_for_status = MagicMock()

        mock_client.get = AsyncMock(side_effect=[version_resp, tags_resp])
        mock_client.post = AsyncMock(return_value=embed_resp)
        e._client = mock_client

        result = await e.check_availability()
        assert result is True
        assert e.available is True

    @pytest.mark.asyncio
    async def test_check_availability_ollama_not_running(self):
        import httpx
        e = self._make_embedder(api="ollama")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        e._client = mock_client

        result = await e.check_availability()
        assert result is False
        assert "Ollama 未启动" in e.unavailable_reason

    @pytest.mark.asyncio
    async def test_check_availability_ollama_wrong_service(self):
        e = self._make_embedder(api="ollama")

        mock_client = AsyncMock()
        resp = MagicMock()
        resp.json.return_value = {"status": "ok"}  # 无 version 字段
        resp.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=resp)
        e._client = mock_client

        result = await e.check_availability()
        assert result is False
        assert "非 Ollama 服务" in e.unavailable_reason

    @pytest.mark.asyncio
    async def test_check_availability_ollama_model_missing(self):
        e = self._make_embedder(api="ollama")

        mock_client = AsyncMock()
        version_resp = MagicMock()
        version_resp.json.return_value = {"version": "0.1.0"}
        version_resp.raise_for_status = MagicMock()

        tags_resp = MagicMock()
        tags_resp.json.return_value = {"models": [{"name": "llama3:latest"}]}
        tags_resp.raise_for_status = MagicMock()

        mock_client.get = AsyncMock(side_effect=[version_resp, tags_resp])
        e._client = mock_client

        result = await e.check_availability()
        assert result is False
        assert "未下载" in e.unavailable_reason

    @pytest.mark.asyncio
    async def test_check_availability_ollama_embed_empty(self):
        e = self._make_embedder(api="ollama")

        mock_client = AsyncMock()
        version_resp = MagicMock()
        version_resp.json.return_value = {"version": "0.1.0"}
        version_resp.raise_for_status = MagicMock()

        tags_resp = MagicMock()
        tags_resp.json.return_value = {"models": [{"name": "bge-m3"}]}
        tags_resp.raise_for_status = MagicMock()

        embed_resp = MagicMock()
        embed_resp.json.return_value = {"embeddings": []}
        embed_resp.raise_for_status = MagicMock()

        mock_client.get = AsyncMock(side_effect=[version_resp, tags_resp])
        mock_client.post = AsyncMock(return_value=embed_resp)
        e._client = mock_client

        result = await e.check_availability()
        assert result is False
        assert "空向量" in e.unavailable_reason

    @pytest.mark.asyncio
    async def test_check_availability_ollama_embed_timeout(self):
        import httpx
        e = self._make_embedder(api="ollama")

        mock_client = AsyncMock()
        version_resp = MagicMock()
        version_resp.json.return_value = {"version": "0.1.0"}
        version_resp.raise_for_status = MagicMock()

        tags_resp = MagicMock()
        tags_resp.json.return_value = {"models": [{"name": "bge-m3"}]}
        tags_resp.raise_for_status = MagicMock()

        mock_client.get = AsyncMock(side_effect=[version_resp, tags_resp])
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        e._client = mock_client

        result = await e.check_availability()
        assert result is False
        assert "超时" in e.unavailable_reason

    @pytest.mark.asyncio
    async def test_check_availability_openai_success(self):
        e = self._make_embedder(api="openai_compat")

        mock_client = AsyncMock()
        resp = MagicMock()
        resp.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=resp)
        e._client = mock_client

        result = await e.check_availability()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_availability_openai_empty(self):
        e = self._make_embedder(api="openai_compat")

        mock_client = AsyncMock()
        resp = MagicMock()
        resp.json.return_value = {"data": []}
        resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=resp)
        e._client = mock_client

        result = await e.check_availability()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_availability_openai_connect_error(self):
        import httpx
        e = self._make_embedder(api="openai_compat")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        e._client = mock_client

        result = await e.check_availability()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_availability_cooldown(self):
        e = self._make_embedder()
        # 模拟冷却中状态：record_failure 后 is_cooling 为 True
        e._cooldown.record_failure()

        result = await e.check_availability()
        assert result is False

    def test_reset_cooldown(self):
        e = self._make_embedder()
        e._available = True
        e.reset_cooldown()
        assert e._available is None

    @pytest.mark.asyncio
    async def test_close(self):
        e = self._make_embedder()
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        e._client = mock_client

        await e.close()
        assert e._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        e = self._make_embedder()
        await e.close()  # 不应报错

    @pytest.mark.asyncio
    async def test_close_error(self):
        e = self._make_embedder()
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock(side_effect=Exception("close error"))
        e._client = mock_client

        await e.close()
        assert e._client is None  # 即使出错也清理

    @pytest.mark.asyncio
    async def test_check_availability_ollama_tags_http_error(self):
        import httpx
        e = self._make_embedder(api="ollama")

        mock_client = AsyncMock()
        version_resp = MagicMock()
        version_resp.json.return_value = {"version": "0.1.0"}
        version_resp.raise_for_status = MagicMock()

        mock_client.get = AsyncMock(side_effect=[
            version_resp,
            httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock()),
        ])
        e._client = mock_client

        result = await e.check_availability()
        assert result is False


    # ── embed_one_sync 测试 ──

    def test_embed_one_sync_success(self):
        e = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            MockClient.return_value = mock_client

            result = e.embed_one_sync("test")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_one_sync_key_error(self):
        e = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"wrong_key": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            MockClient.return_value = mock_client

            with pytest.raises(ValueError, match="响应格式错误"):
                e.embed_one_sync("test")

    # ── embed_batch_sync 测试 ──

    def test_embed_batch_sync_empty(self):
        e = self._make_embedder()
        result = e.embed_batch_sync([])
        assert result == []

    def test_embed_batch_sync_success(self):
        e = self._make_embedder()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1], [0.2]]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            MockClient.return_value = mock_client

            result = e.embed_batch_sync(["a", "b"], batch_size=32)
            assert result == [[0.1], [0.2]]

    def test_embed_batch_sync_partial_failure(self):
        e = self._make_embedder()
        good_resp = MagicMock()
        good_resp.json.return_value = {"embeddings": [[0.1]]}
        good_resp.raise_for_status = MagicMock()

        call_count = 0
        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("fail")
            return good_resp

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post = mock_post
            MockClient.return_value = mock_client

            result = e.embed_batch_sync(["a", "b"], batch_size=1)
            assert len(result) == 2
            assert result[0] is None  # 第一批失败
            assert result[1] == [0.1]  # 第二批成功


# ── create_embedder 工厂测试 ──

class TestCreateEmbedder:

    def test_create_ollama(self):
        from robotmem.embed import create_embedder, OllamaEmbedder
        config = MagicMock()
        config.embed_backend = "ollama"
        config.embedding_model = "bge-m3"
        config.ollama_url = "http://localhost:11434"
        config.embedding_dim = 768
        config.embed_api = "ollama"

        embedder = create_embedder(config)
        assert isinstance(embedder, OllamaEmbedder)

    def test_create_onnx(self):
        from robotmem.embed import create_embedder
        config = MagicMock()
        config.embed_backend = "onnx"
        config.onnx_model = "BAAI/bge-small-en-v1.5"
        config.onnx_dim = 384
        config.fastembed_cache_dir = ""

        embedder = create_embedder(config)
        from robotmem.embed_onnx import FastEmbedEmbedder
        assert isinstance(embedder, FastEmbedEmbedder)

    def test_create_default_ollama(self):
        """没有 embed_backend 属性时默认 ollama"""
        from robotmem.embed import create_embedder, OllamaEmbedder
        config = MagicMock(spec=[])  # 没有任何属性
        config.embedding_model = "bge-m3"
        config.ollama_url = "http://localhost:11434"
        config.embedding_dim = 768
        config.embed_api = "ollama"

        embedder = create_embedder(config)
        assert isinstance(embedder, OllamaEmbedder)


# ── FastEmbedEmbedder 测试 ──

class TestFastEmbedEmbedder:

    def _make_embedder(self):
        from robotmem.embed_onnx import FastEmbedEmbedder
        return FastEmbedEmbedder(model="test-model", dim=4)

    def test_init(self):
        e = self._make_embedder()
        assert e.model == "test-model"
        assert e.dim == 4
        assert e.available is False

    @pytest.mark.asyncio
    async def test_close(self):
        e = self._make_embedder()
        e._encoder = MagicMock()
        await e.close()
        assert e._encoder is None

    @pytest.mark.asyncio
    async def test_embed_one_success(self):
        import numpy as np
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.return_value = iter([np.array([0.1, 0.2, 0.3, 0.4])])
        e._encoder = mock_encoder

        result = await e.embed_one("test")
        assert len(result) == 4
        assert abs(result[0] - 0.1) < 1e-6

    @pytest.mark.asyncio
    async def test_embed_one_failure(self):
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.side_effect = Exception("model error")
        e._encoder = mock_encoder

        with pytest.raises(RuntimeError, match="ONNX embedding 失败"):
            await e.embed_one("test")

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        e = self._make_embedder()
        result = await e.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        import numpy as np
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.return_value = iter([
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
        ])
        e._encoder = mock_encoder

        result = await e.embed_batch(["a", "b"])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_embed_batch_failure(self):
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.side_effect = Exception("fail")
        e._encoder = mock_encoder

        result = await e.embed_batch(["a", "b"])
        assert result == [None, None]

    @pytest.mark.asyncio
    async def test_check_availability_success(self):
        import numpy as np
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.return_value = iter([np.array([0.1, 0.2, 0.3, 0.4])])
        e._encoder = mock_encoder

        result = await e.check_availability()
        assert result is True
        assert e.available is True

    @pytest.mark.asyncio
    async def test_check_availability_dim_mismatch(self):
        import numpy as np
        e = self._make_embedder()  # dim=4
        mock_encoder = MagicMock()
        mock_encoder.embed.return_value = iter([np.array([0.1, 0.2])])  # dim=2
        e._encoder = mock_encoder

        result = await e.check_availability()
        assert result is False
        assert "维度不匹配" in e.unavailable_reason

    @pytest.mark.asyncio
    async def test_check_availability_import_error(self):
        e = self._make_embedder()
        e._encoder = None

        with patch.dict("sys.modules", {"fastembed": None}):
            # _ensure_encoder will try to import fastembed
            e._init_lock = __import__("threading").Lock()
            # Simulate ImportError in _ensure_encoder
            original = e._ensure_encoder

            def mock_ensure():
                raise ImportError("no fastembed")

            e._ensure_encoder = mock_ensure
            result = await e.check_availability()
            assert result is False

    @pytest.mark.asyncio
    async def test_check_availability_general_error(self):
        e = self._make_embedder()

        def mock_ensure():
            raise RuntimeError("some error")

        e._ensure_encoder = mock_ensure
        result = await e.check_availability()
        assert result is False
        assert "不可用" in e.unavailable_reason

    def test_cache_dir_stored(self):
        from robotmem.embed_onnx import FastEmbedEmbedder
        e = FastEmbedEmbedder(model="test", dim=4, cache_dir="/tmp/cache")
        assert e._cache_dir == "/tmp/cache"

    # ── embed_one_sync 测试 ──

    def test_embed_one_sync_success(self):
        import numpy as np
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.return_value = iter([np.array([0.1, 0.2, 0.3, 0.4])])
        e._encoder = mock_encoder

        result = e.embed_one_sync("test")
        assert len(result) == 4
        assert abs(result[0] - 0.1) < 1e-6

    def test_embed_one_sync_failure(self):
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.side_effect = Exception("model error")
        e._encoder = mock_encoder

        with pytest.raises(RuntimeError, match="ONNX embedding 失败"):
            e.embed_one_sync("test")

    # ── embed_batch_sync 测试 ──

    def test_embed_batch_sync_empty(self):
        e = self._make_embedder()
        result = e.embed_batch_sync([])
        assert result == []

    def test_embed_batch_sync_success(self):
        import numpy as np
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.return_value = iter([
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
        ])
        e._encoder = mock_encoder

        result = e.embed_batch_sync(["a", "b"])
        assert len(result) == 2

    def test_embed_batch_sync_failure(self):
        e = self._make_embedder()
        mock_encoder = MagicMock()
        mock_encoder.embed.side_effect = Exception("fail")
        e._encoder = mock_encoder

        result = e.embed_batch_sync(["a", "b"])
        assert result == [None, None]
