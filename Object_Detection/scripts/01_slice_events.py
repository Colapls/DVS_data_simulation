#!/usr/bin/env python3
"""切片：把 CARLA 采集产物切成训练窗并体素化。

    python scripts/01_slice_events.py --config configs/data/slice_carla.yaml

输出：data/slices/<sequence_id>/<camera_id>/<win_idx>.npz
每�� npz 含体素网格、标注、域随机化所需的 contrast_threshold/map/weather。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dvsdet.slicing.slicer import SliceConfig, slice_sequence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="事件切片 + 体素化")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="覆盖配置里的数据集根目录")
    parser.add_argument("--output", type=Path, help="覆盖配置里的切片输出目录")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_root = Path(args.dataset or config["dataset_root"])
    output_root = Path(args.output or config["output_root"])
    slice_config = SliceConfig(
        window_ms=float(config.get("window_ms", 50.0)),
        overlap_ms=float(config.get("overlap_ms", 25.0)),
        num_time_channels=int(config.get("num_time_channels", 9)),
        grid_width=int(config.get("grid_width", 640)),
        grid_height=int(config.get("grid_height", 360)),
        min_events_per_window=int(config.get("min_events_per_window", 10)),
    )

    sequences = sorted(
        path for path in (dataset_root / "sequences").glob("sequence_*") if path.is_dir()
    )
    print(f"数据集：{dataset_root}")
    print(f"序列 {len(sequences)} 条 → 切片输出 {output_root}")
    print(
        f"窗 {slice_config.window_ms} ms / 重叠 {slice_config.overlap_ms} ms / "
        f"时间通道 {slice_config.num_time_channels}（共 {2*slice_config.num_time_channels} 通道）/ "
        f"网格 {slice_config.grid_width}×{slice_config.grid_height}"
    )

    reports = []
    total = 0
    for sequence in sequences:
        report = slice_sequence(sequence, output_root, slice_config, args.overwrite)
        total += report["windows"]
        reports.append(report)
        print(
            f"  {report['sequence_id']}: {report['windows']} 窗，"
            f"{report['seconds']:.1f} s（{report['ms_per_window']} ms/窗）"
        )
    print(
        f"\n完成：{total} 个训练窗，"
        f"体素化 {'numba 加速' if reports and reports[0]['numba'] else '纯 numpy（慢）'}"
    )
    (output_root.parent / "slice_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
