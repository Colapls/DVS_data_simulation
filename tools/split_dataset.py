#!/usr/bin/env python3
"""按路口划分数据集，并统计三类目标的分布。

划分按**路口**而不是按序列或帧随机分：同一个路口的不同序列外观高度相似，
按帧随机分会让同一批目标同时出现在训练集和测试集里（方案 §7.4、§10）。
测试集另外保证至少含一条夜间序列。

用法::

    python tools/split_dataset.py --dataset dvs_roadside_multicam
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def _read_locations(dataset_root: Path) -> dict[str, dict[str, str]]:
    path = dataset_root / "locations.csv"
    if not path.is_file():
        raise SystemExit(f"缺少位置时间索引：{path}")
    with path.open("r", encoding="utf-8", newline="") as source:
        return {row["sequence_id"]: row for row in csv.DictReader(source)}


def _split_by_junction(
    locations: dict[str, dict[str, str]], seed: int = 20240501
) -> dict[str, list[str]]:
    """把路口（不是序列）分到 train/val/test。"""
    junctions: dict[str, list[str]] = defaultdict(list)
    for sequence_id, row in locations.items():
        junctions[row.get("junction_id") or sequence_id].append(sequence_id)
    names = sorted(junctions)
    random.Random(seed).shuffle(names)
    total = len(names)
    train_end = max(1, round(total * TRAIN_RATIO))
    val_end = max(train_end + 1, round(total * (TRAIN_RATIO + VAL_RATIO)))
    assignment = {
        "train": names[:train_end],
        "val": names[train_end:val_end],
        "test": names[val_end:],
    }
    return {
        split: sorted(
            sequence_id
            for name in group
            for sequence_id in junctions[name]
        )
        for split, group in assignment.items()
    }


def _ensure_night_in_test(
    splits: dict[str, list[str]], locations: dict[str, dict[str, str]]
) -> str | None:
    """保证测试集里至少有一条夜间序列；必要时把整个路口挪进测试集。"""
    def has_night(sequence_ids: list[str]) -> bool:
        return any(
            "night" in (locations.get(sequence_id, {}).get("weather") or "").lower()
            for sequence_id in sequence_ids
        )

    if has_night(splits["test"]):
        return None
    night_junction = None
    for sequence_id in splits["train"]:
        weather = (locations.get(sequence_id, {}).get("weather") or "").lower()
        if "night" not in weather:
            continue
        junction = locations.get(sequence_id, {}).get("junction_id")
        night_junction = junction or sequence_id
        break
    if night_junction is None:
        return None
    moved = [
        sequence_id
        for sequence_id in splits["train"]
        if (locations.get(sequence_id, {}).get("junction_id") or sequence_id)
        == night_junction
    ]
    if not moved:
        return None
    splits["train"] = [
        sequence_id
        for sequence_id in splits["train"]
        if sequence_id not in moved
    ]
    splits["test"].extend(moved)
    splits["test"].sort()
    return f"路口 {night_junction} 整组移入测试集，保证含夜间序列"


def _class_statistics(
    dataset_root: Path, sequence_ids: list[str]
) -> dict[str, Any]:
    """按类别统计实例数、轨迹数、框尺寸和零事件比例。"""
    instances: dict[str, int] = defaultdict(int)
    tracks: dict[str, set[int]] = defaultdict(set)
    low_event: dict[str, int] = defaultdict(int)
    sizes: dict[str, list[int]] = defaultdict(list)
    for sequence_id in sequence_ids:
        cameras = dataset_root / "sequences" / sequence_id / "cameras"
        if not cameras.is_dir():
            continue
        for camera_dir in sorted(cameras.iterdir()):
            annotation_file = camera_dir / "annotations.json"
            if not annotation_file.is_file():
                continue
            payload = json.loads(annotation_file.read_text(encoding="utf-8"))
            for frame in payload.get("frames", []):
                for annotation in frame.get("annotations", []):
                    category = annotation.get("category", "unknown")
                    instances[category] += 1
                    tracks[category].add(annotation.get("actor_id", 0))
                    if "low_event" in annotation.get("quality_flags", []):
                        low_event[category] += 1
                    box_w, box_h = annotation["bbox"][2], annotation["bbox"][3]
                    sizes[category].append(box_w * box_h)
    summary: dict[str, Any] = {}
    for category, count in sorted(instances.items()):
        areas = sorted(sizes[category])
        summary[category] = {
            "instances": count,
            "tracks": len(tracks[category]),
            "low_event_ratio": round(low_event[category] / count, 4) if count else 0.0,
            "median_box_area_px": areas[len(areas) // 2] if areas else 0,
            "min_box_area_px": areas[0] if areas else 0,
            "max_box_area_px": areas[-1] if areas else 0,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="按路口划分数据集并统计目标分布")
    parser.add_argument(
        "--dataset", type=Path, default=Path("dvs_roadside_multicam")
    )
    parser.add_argument(
        "--seed", type=int, default=20240501, help="路口划分随机种子"
    )
    args = parser.parse_args()
    dataset_root = args.dataset.expanduser().resolve()

    locations = _read_locations(dataset_root)
    splits = _split_by_junction(locations, args.seed)
    note = _ensure_night_in_test(splits, locations)

    splits_dir = dataset_root / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name, sequence_ids in splits.items():
        (splits_dir / f"{name}.txt").write_text(
            "\n".join(sequence_ids) + ("\n" if sequence_ids else ""),
            encoding="utf-8",
        )

    report = {
        "splits": {
            name: {
                "sequences": len(sequence_ids),
                "junctions": sorted(
                    {
                        locations.get(sequence_id, {}).get("junction_id")
                        or sequence_id
                        for sequence_id in sequence_ids
                    }
                ),
                "statistics": _class_statistics(dataset_root, sequence_ids),
            }
            for name, sequence_ids in splits.items()
        }
    }
    if note:
        report["note"] = note
    (dataset_root / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, sequence_ids in splits.items():
        print(f"{name}: {len(sequence_ids)} 条序列")
    if note:
        print(f"提示：{note}")
    print(f"划分结果已写入 {splits_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
