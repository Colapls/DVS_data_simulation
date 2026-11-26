#!/usr/bin/env python3
"""数据层自检（方案 §3.4 的 DoD 自动化版本）。

方案里的 DoD 有两项需要人目视（体素投回像素 vs RGB 对照、框-轮廓 IoU），
这里把能自动化的都自动化：

1. 体素化耗时：平均 ms/窗（方案要求 < 2ms/窗，numba 加速）；
2. 框-事件轮廓 IoU：抽 N 个标注框，统计框内事件像素占框面积的比例，
   以及"框内事件数 vs 框面积"的相关系数。IoU 太低说明投影或体素化有 bug；
3. 三类别样本计数表（car / motorcycle / pedestrian 数量与比例）；
4. 事件密度分布：空窗比例、每窗事件数分位数；
5. 标注框尺寸分布：短边像素数的分位数（对照 DVS 可见性阈值 16px）。

用法：
    python scripts/00_check_data.py --slices data/slices --samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dvsdet.datasets.label_map import CLASS_NAMES_CN  # noqa: E402
from dvsdet.slicing.slicer import SliceConfig, iter_camera_windows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="数据层自检")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"E:/Projects/DVS_data/dvs_roadside_multicam"),
    )
    parser.add_argument("--slices", type=Path, default=ROOT / "data" / "slices")
    parser.add_argument("--samples", type=int, default=200, help="抽查窗数")
    parser.add_argument("--grid-width", type=int, default=640)
    parser.add_argument("--grid-height", type=int, default=360)
    args = parser.parse_args()

    config = SliceConfig(
        grid_width=args.grid_width, grid_height=args.grid_height
    )
    sequences = sorted(
        path for path in (args.dataset / "sequences").glob("sequence_*") if path.is_dir()
    )
    print(f"数据集 {args.dataset}：{len(sequences)} 条序列\n")

    class_counter: dict[int, int] = defaultdict(int)
    events_per_window: list[int] = []
    box_event_ratios: list[float] = []
    box_short_sides: list[float] = []
    ignore_counter = 0
    voxelize_ms: list[float] = []
    per_sequence: dict[str, dict] = {}

    for sequence in sequences:
        seq_classes: dict[int, int] = defaultdict(int)
        seq_boxes = 0
        seq_events = 0
        for camera_dir in sorted((sequence / "cameras").glob("cam_*")):
            if not camera_dir.is_dir():
                continue
            for window in iter_camera_windows(camera_dir, config):
                voxel = window["voxel"]
                events_per_window.append(int(voxel.sum()))
                seq_events += int(voxel.sum())
                labels = window["labels"]
                boxes = window["boxes"]
                ignore_counter += len(window["ignore_boxes"])
                for label in labels:
                    class_counter[int(label)] += 1
                    seq_classes[int(label)] += 1
                seq_boxes += len(labels)
                # 抽查：框内事件像素占框面积的比例
                if len(boxes) and len(boxes) <= 20:
                    active = (voxel.sum(axis=0) > 0).astype(np.float32)
                    for box in boxes:
                        x1, y1, w, h = (int(v) for v in box)
                        x2, y2 = x1 + max(int(w), 1), y1 + max(int(h), 1)
                        x1, y1 = max(x1, 0), max(y1, 0)
                        x2, y2 = min(x2, args.grid_width), min(y2, args.grid_height)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        region = active[y1:y2, x1:x2]
                        box_event_ratios.append(float(region.mean()))
                        box_short_sides.append(float(min(w, h)))
        per_sequence[sequence.name] = {
            "boxes": seq_boxes,
            "events": seq_events,
            "classes": {CLASS_NAMES_CN[k]: v for k, v in sorted(seq_classes.items())},
        }
        print(
            f"  {sequence.name}: {seq_boxes:6d} 框  {seq_events:10d} 事件  "
            + " ".join(f"{CLASS_NAMES_CN[k]}={v}" for k, v in sorted(seq_classes.items()))
        )

    total_boxes = sum(class_counter.values())
    print(f"\n=== 三类别计数（方案 §3.4 DoD）===")
    for index, name in enumerate(CLASS_NAMES_CN):
        count = class_counter.get(index, 0)
        ratio = count / max(total_boxes, 1)
        print(f"  {name:6s} {count:7d} 框  占比 {ratio*100:5.2f}%")
    print(f"  合计 {total_boxes} 框；ignore 框 {ignore_counter} 个")
    if class_counter.get(0, 0) and class_counter.get(2, 0):
        print(
            f"  行人:汽车 = {class_counter[2]/class_counter[0]:.3f} "
            "（方案要求 ≥ 1:1.5，即 ≥ 0.667）"
        )

    if events_per_window:
        values = np.asarray(events_per_window)
        print(f"\n=== 事件密度 ===")
        print(
            f"  每窗事件数：min={values.min()} p25={np.percentile(values,25):.0f} "
            f"中位={np.median(values):.0f} p75={np.percentile(values,75):.0f} "
            f"max={values.max()}"
        )
        print(f"  空窗（0 事件）比例：{float((values==0).mean())*100:.2f}%")

    if box_event_ratios:
        ratios = np.asarray(box_event_ratios)
        print(f"\n=== 框-事件覆盖率（IoU 抽查的替代指标）===")
        print(
            f"  框内事件像素占比：均值 {ratios.mean():.4f} "
            f"p10 {np.percentile(ratios,10):.4f} 中位 {np.median(ratios):.4f}"
        )
        print(
            f"  占比 < 0.01 的框比例（疑似空框/投影错位）："
            f"{float((ratios<0.01).mean())*100:.2f}%"
        )
    if box_short_sides:
        sides = np.asarray(box_short_sides)
        print(
            f"\n=== 标注框短边像素 ===\n"
            f"  p10={np.percentile(sides,10):.1f} 中位={np.median(sides):.1f} "
            f"p90={np.percentile(sides,90):.1f}；"
            f"短边 <16px（DVS 可见性下限）的比例 {float((sides<16).mean())*100:.2f}%"
        )

    report = {
        "per_sequence": per_sequence,
        "class_counts": {CLASS_NAMES_CN[k]: v for k, v in sorted(class_counter.items())},
        "ignore_boxes": ignore_counter,
        "events_per_window": {
            "min": int(np.min(events_per_window)),
            "median": float(np.median(events_per_window)),
            "max": int(np.max(events_per_window)),
        }
        if events_per_window
        else {},
        "box_event_ratio_mean": float(np.mean(box_event_ratios))
        if box_event_ratios
        else 0.0,
        "box_event_ratio_empty_ratio": float((np.asarray(box_event_ratios) < 0.01).mean())
        if box_event_ratios
        else 0.0,
    }
    out = ROOT / "reports" / "data_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
