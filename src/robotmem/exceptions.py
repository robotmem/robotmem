"""robotmem 异常层级 — SDK 自定义异常

MCP Server 用 @mcp_error_boundary 吞异常返回 {"error": "..."};
SDK 让异常传播，但保证类型可预测。

层级：
    RobotMemError          基类
    ├── ValidationError    L1：参数校验失败
    ├── DatabaseError      L2：数据库写入失败
    └── EmbeddingError     L2：embedding 失败
"""


class RobotMemError(Exception):
    """robotmem 基类异常"""


class ValidationError(RobotMemError):
    """L1：参数校验失败（Pydantic 不过 / 类型错误 / 范围越界）"""


class DatabaseError(RobotMemError):
    """L2：数据库写入失败（锁超时 / 磁盘满 / 损坏）"""


class EmbeddingError(RobotMemError):
    """L2：embedding 失败（ONNX 崩溃 / Ollama 超时）"""
