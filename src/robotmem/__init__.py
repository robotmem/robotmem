"""robotmem — 机器人记忆系统

Python API:
    from robotmem import save_perception, recall

    save_perception(
        description="Grasped red cup: force=12.5N, 30 steps",
        perception_type="procedural",
        data='{"actions": [[0.1, -0.3, 0.05]], "force_peak": 12.5}',
    )

    memories = recall("how to grasp a cup")
    for m in memories["memories"]:
        print(m["content"], m["_rrf_score"])
"""

__version__ = "0.1.0"

from .api import (
    end_session,
    forget,
    learn,
    recall,
    save_perception,
    start_session,
    update,
)
from .exceptions import (
    DatabaseError,
    EmbeddingError,
    RobotMemError,
    ValidationError,
)
from .sdk import RobotMemory

__all__ = [
    "RobotMemory",
    "RobotMemError",
    "ValidationError",
    "DatabaseError",
    "EmbeddingError",
    "save_perception",
    "recall",
    "learn",
    "forget",
    "update",
    "start_session",
    "end_session",
    "__version__",
]
