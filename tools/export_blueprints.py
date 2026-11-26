#!/usr/bin/env python3
"""导出 CARLA 实际蓝图清单，核对三类别映射表。

方案 §6.3 的第一步：连上 CARLA 跑一遍，导出实际蓝图清单并核对映射表，
确认没有静默归类之后再开始重采。

输出两个文件：

- ``blueprints.csv``：全部车辆/行人蓝图及其类别、category_id、子类
- ``unknown_blueprints.csv``：靠"其余 vehicle.* → car"兜底规则归类的蓝图，
  需要人工确认是不是真的属于 car

用法::

    python tools/export_blueprints.py --host 127.0.0.1 --port 2000 --out-dir .
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvs_collector.classes import (  # noqa: E402
    CLASS_MAP_VERSION,
    classify_blueprint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 CARLA 蓝图清单并核对类别映射表")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    try:
        import carla
    except ImportError as exc:
        print(
            "找不到 CARLA Python API：请在 dvs 环境运行本脚本。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    library = world.get_blueprint_library()

    rows = []
    for blueprint in list(library.filter("vehicle.*")) + list(
        library.filter("walker.pedestrian.*")
    ):
        info = classify_blueprint(blueprint.id)
        rows.append(
            {
                "blueprint_id": blueprint.id,
                "category": info["category"],
                "category_id": "" if info["category_id"] is None else info["category_id"],
                "subtype": info["subtype"] or "",
                "by_fallback": info["by_fallback"],
                "class_map_version": CLASS_MAP_VERSION,
            }
        )
    rows.sort(key=lambda row: row["blueprint_id"])

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "blueprint_id",
        "category",
        "category_id",
        "subtype",
        "by_fallback",
        "class_map_version",
    )
    with (out_dir / "blueprints.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    unknown = [row for row in rows if row["by_fallback"]]
    with (out_dir / "unknown_blueprints.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(unknown)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"地图 {world.get_map().name}：共 {len(rows)} 个蓝图")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    print(f"待人工确认（兜底归类）：{len(unknown)} 个 → {out_dir / 'unknown_blueprints.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
