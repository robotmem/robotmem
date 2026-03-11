"""SQLite 工具函数 — FTS5 分词 + 向量序列化

从 index1 db.py 抽取的公共工具，供 db_cog.py / ops/ / search.py 使用。
"""

from __future__ import annotations

import re
import sqlite3
import struct

# CJK 字符检测（中日韩）
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")

# SQLite 版本检测：contentless_delete 需要 3.43.0+
_SQLITE_VERSION = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
SUPPORTS_CONTENTLESS_DELETE = _SQLITE_VERSION >= (3, 43, 0)


def tokenize_for_fts5(text: str) -> str:
    """对含 CJK 字符的文本做分词，供 FTS5 索引和查询使用。

    - 纯英文文本原样返回（零开销）
    - 含 CJK + jieba 已装 → jieba 分词
    - 含 CJK + jieba 未装 → 原样返回（降级，英文用户不受影响）
    """
    if not text or not _CJK_RE.search(text):
        return text
    try:
        import jieba
    except ImportError:
        # jieba 未装 — 降级为原样返回，不阻塞写入
        return text
    return " ".join(jieba.cut(text))


def floats_to_blob(floats: list[float]) -> bytes:
    """float 列表转 sqlite-vec 需要的 bytes"""
    return struct.pack(f"{len(floats)}f", *floats)


def blob_to_floats(blob: bytes, dim: int) -> list[float]:
    """bytes 转 float 列表"""
    expected = dim * 4
    if len(blob) != expected:
        raise ValueError(
            f"向量维度不匹配: 期望 {dim} 维 ({expected} 字节), 实际 {len(blob)} 字节"
        )
    return list(struct.unpack(f"{dim}f", blob))
