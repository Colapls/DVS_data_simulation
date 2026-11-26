#!/usr/bin/env python3
"""把数据集 index_all.txt 重建为"磁盘上状态为 completed 的序列"。

修复历史遗留：旧版 register_sequence 只追加不清理，索引指向已删除的
序列（实测索引 13 条、磁盘 8 条，其中还包含唯一的夜间序列）。采集器
现在每次登记新序列都会自动重建；本脚本用于立即修复存量数据集，不用
等下一次采集。

同时清理 locations.csv 里的孤儿行——序列目录被删后，那里的位置记录
不会自动消失（实测同样残留 5 条）。

原文件先备份成 *.bak，再原子替换，出错可回滚。

    python tools/rebuild_index.py [数据集目录]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvs_collector.dataset import DatasetLayout  # noqa: E402


def _clean_locations(path: Path, valid: set[str]) -> list[str]:
    """删掉 locations.csv 里序列目录已不存在的行，返回被删的序列 ID。"""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    kept = [row for row in rows if row.get("sequence_id") in valid]
    removed = [
        row.get("sequence_id", "")
        for row in rows
        if row.get("sequence_id") not in valid
    ]
    if len(kept) == len(rows) or not fields:
        return []
    path.with_suffix(".csv.bak").write_text(
        path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    temp.replace(path)
    return removed


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "dvs_roadside_multicam")
    layout = DatasetLayout(root)
    completed = layout._completed_sequence_ids()
    index = layout.root / "index_all.txt"
    old = set(index.read_text(encoding="utf-8").splitlines()) if index.is_file() else set()
    new = set(completed)
    if old != new:
        # 先备份再原子替换：出错了可以直接把 .bak 改回来。
        if index.is_file():
            index.with_suffix(".txt.bak").write_text(
                index.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
            )
        temp = index.with_suffix(".txt.tmp")
        temp.write_text(
            "\n".join(completed) + ("\n" if completed else ""),
            encoding="utf-8",
            newline="\n",
        )
        temp.replace(index)
        print(f"索引已重建：{len(completed)} 条 completed 序列（原 {len(old)} 条）")
        print("原文件已备份为 index_all.txt.bak")
        for item in sorted(old - new):
            print(f"  移除（磁盘上不存在或未完成）：{item}")
        for item in sorted(new - old):
            print(f"  补入：{item}")
    else:
        print(f"索引已与磁盘一致（{len(completed)} 条），无需改动")

    orphan_rows = _clean_locations(layout.root / "locations.csv", new)
    if orphan_rows:
        print(f"locations.csv 已清理 {len(orphan_rows)} 条孤儿记录（原文件备份为 locations.csv.bak）")
        for item in orphan_rows:
            print(f"  移除：{item}")
    else:
        print("locations.csv 无孤儿记录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
