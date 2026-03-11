#!/usr/bin/env python3
"""robotmem 迁移辅助脚本 — 批量替换 import 路径

用法：
    python scripts/migrate_imports.py src/robotmem/auto_classify.py
    python scripts/migrate_imports.py src/robotmem/ops/facts.py
    python scripts/migrate_imports.py src/robotmem/  # 批量处理目录

功能：
    - 自动替换 index1 → robotmem 的 import 路径
    - 输出每处修改的行号和内容，方便人工审查
    - --dry-run 模式只输出不修改
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# import 路径替换规则（顺序敏感，长匹配优先）
IMPORT_RULES: list[tuple[str, str]] = [
    # ops 内部：3层→2层
    ("from ...resilience import", "from ..resilience import"),
    ("from ...db import", "from ..db import"),

    # cognitive 子模块→顶层
    ("from .cognitive.auto_classify import", "from .auto_classify import"),
    ("from .cognitive.dedup import", "from .dedup import"),
    ("from .cognitive.search_cog import", "from .search import"),
    ("from .cognitive.db_cog import", "from .db_cog import"),
    ("from .cognitive.conflict import", "from .conflict import"),
    ("from .cognitive.capture import", "from .temporal import"),
    ("from .cognitive.tag_tree import", "from .tag_tree import"),
    ("from .cognitive.rules import", "from .rules import"),
    ("from .cognitive.rrf import", "from .rrf import"),
    ("from .cognitive.ops.facts import", "from .ops.facts import"),
    ("from .cognitive.ops.perceptions import", "from .ops.perceptions import"),
    ("from .cognitive.ops.sessions import", "from .ops.sessions import"),
    ("from .cognitive.ops.tags import", "from .ops.tags import"),
    ("from .cognitive.ops.search import", "from .ops.search import"),

    # 包名替换
    ("from index1.", "from robotmem."),
    ("import index1.", "import robotmem."),
]

# 需要删除的 import（robotmem 不需要的模块）
DELETE_PATTERNS: list[re.Pattern] = [
    re.compile(r"from .unified_search import"),
    re.compile(r"from .cognitive.hooks import"),
    re.compile(r"from .cognitive.crystallize import"),
    re.compile(r"from .cognitive.observer"),
    re.compile(r"from .cognitive.realtime import"),
    re.compile(r"from .cognitive.llm_utils import"),
    re.compile(r"from .cognitive.pearl"),
    re.compile(r"from .cognitive.ops.bundles import"),
    re.compile(r"from .cognitive.ops.edges import"),
    re.compile(r"from .cognitive.ops.models import"),
    re.compile(r"from .cognitive.ops.ecosystem import"),
    re.compile(r"from .cognitive.ops.observations import"),
    re.compile(r"from .cognitive.ops.rule_store import"),
    re.compile(r"from .cognitive.ops.checkpoints import"),
    re.compile(r"from .cognitive.ops.cross_edges import"),
    re.compile(r"from .cognitive.ops.pearls import"),
    re.compile(r"from .indexer import"),
    re.compile(r"from .db import Database"),  # corpus Database 类
]


def process_file(filepath: Path, dry_run: bool = False) -> list[str]:
    """处理单个文件，返回变更日志"""
    changes: list[str] = []
    lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines: list[str] = []

    for i, line in enumerate(lines, 1):
        original = line

        # 检查是否需要删除
        deleted = False
        for pattern in DELETE_PATTERNS:
            if pattern.search(line):
                changes.append(f"  L{i} 删除: {line.rstrip()}")
                deleted = True
                new_lines.append(f"# [migrated] {line}")
                break

        if deleted:
            continue

        # 检查是否需要替换
        for old, new in IMPORT_RULES:
            if old in line:
                line = line.replace(old, new)

        if line != original:
            changes.append(f"  L{i} 替换: {original.rstrip()}")
            changes.append(f"       → {line.rstrip()}")

        new_lines.append(line)

    if changes and not dry_run:
        filepath.write_text("".join(new_lines), encoding="utf-8")

    return changes


def main():
    parser = argparse.ArgumentParser(description="robotmem import 路径迁移工具")
    parser.add_argument("path", help="文件或目录路径")
    parser.add_argument("--dry-run", action="store_true", help="只输出不修改")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"路径不存在: {target}", file=sys.stderr)
        sys.exit(1)

    files = list(target.rglob("*.py")) if target.is_dir() else [target]
    total_changes = 0

    for f in sorted(files):
        changes = process_file(f, dry_run=args.dry_run)
        if changes:
            mode = "[DRY-RUN] " if args.dry_run else ""
            print(f"\n{mode}{f}:")
            for c in changes:
                print(c)
            total_changes += len([c for c in changes if c.startswith("  L")])

    print(f"\n{'=' * 40}")
    print(f"总计: {len(files)} 个文件, {total_changes} 处变更")
    if args.dry_run:
        print("（dry-run 模式，未实际修改文件）")


if __name__ == "__main__":
    main()
