#!/usr/bin/env python3
"""离线标注：从采集快照重投影，并统计每个目标框内的事件。

两步都能不连 CARLA 跑：

- ``project``：读 ``snapshot.json`` 重算 annotations.json。投影代码改了以后
  不必重采，只要重跑这个脚本。
- ``stats``：读 events.h5 统计框内事件数、密度、活跃像素比例和极性比，
  写回 ``quality_flags``（low_event / static）。

用法::

    python tools/annotate_offline.py project --sequence <序列目录>
    python tools/annotate_offline.py stats --dataset <数据集目录>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvs_collector.classes import CLASSES, CLASS_MAP_VERSION  # noqa: E402
from dvs_collector.labels import (  # noqa: E402
    MAX_DETECT_DISTANCE_M,
    MIN_DETECT_SHORT_SIDE_PX,
    project_records,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _integration_time_us(camera_dir: Path, frames: list[dict[str, Any]]) -> int:
    """事件积分时间：优先用 HDF5 里记录的值，其次用帧间隔。"""
    try:
        import h5py
    except ImportError:
        h5py = None  # type: ignore[assignment]
    if h5py is not None:
        try:
            with h5py.File(camera_dir / "events.h5", "r") as handle:
                if "meta/integration_time_us" in handle:
                    return int(handle["meta/integration_time_us"][()])
        except OSError:
            pass
    if len(frames) > 1:
        return int(frames[1]["timestamp_us"] - frames[0]["timestamp_us"])
    return 50_000


def project_sequence(
    sequence_root: Path,
    np: Any,
    min_detect_px: int = MIN_DETECT_SHORT_SIDE_PX,
    max_detect_distance: float = MAX_DETECT_DISTANCE_M,
) -> list[str]:
    """用 snapshot.json 重算该序列所有机位的 annotations.json。"""
    snapshot = _load_json(sequence_root / "snapshot.json")
    sequence_id = snapshot["sequence_id"]
    written: list[str] = []
    for camera_id, camera in snapshot["cameras"].items():
        # 优先用快照里存的 inverse_matrix（与在线投影同源，逐字段可复现）；
        # 旧快照没有这个字段时才回退到对 world_matrix 做数值求逆。
        if "inverse_matrix" in camera:
            inverse = np.asarray(camera["inverse_matrix"], dtype=np.float64)
        else:
            inverse = np.linalg.inv(
                np.asarray(camera["world_matrix"], dtype=np.float64)
            )
        frames: list[dict[str, Any]] = []
        for frame in snapshot["frames"]:
            visible = {
                int(actor_id): values
                for actor_id, values in frame.get("visibility", {})
                .get(camera_id, {})
                .items()
            }
            labels, ignored, _ = project_records(
                frame["actors"],
                inverse,
                int(camera["width"]),
                int(camera["height"]),
                float(camera["fov"]),
                np,
                visible_by_actor=visible,
                min_short_side_px=min_detect_px,
                max_distance_m=max_detect_distance,
            )
            frames.append(
                {
                    "frame_id": (
                        f"{sequence_id}_{camera_id}_{frame['frame_index']:06d}"
                    ),
                    "timestamp_us": frame["timestamp_us"],
                    "integration_time_us": 0,
                    "annotations": labels,
                    "ignored_annotations": ignored,
                }
            )
        camera_dir = sequence_root / "cameras" / camera_id
        integration = _integration_time_us(camera_dir, frames)
        for frame in frames:
            frame["integration_time_us"] = integration
        _write_json(
            camera_dir / "annotations.json",
            {
                "sequence_id": sequence_id,
                "camera_id": camera_id,
                "sensor_width": int(camera["width"]),
                "sensor_height": int(camera["height"]),
                "class_map_version": snapshot.get(
                    "class_map_version", CLASS_MAP_VERSION
                ),
                "classes": snapshot.get("classes", dict(CLASSES)),
                "frames": frames,
            },
        )
        written.append(camera_id)
    return written


def annotate_camera(
    annotations: dict[str, Any], events_path: Path, np: Any
) -> dict[str, int]:
    """统计每个标注框内的事件，并写回质量标记。"""
    import h5py

    counts = {"instances": 0, "low_event": 0, "static": 0, "occluded": 0}
    with h5py.File(events_path, "r") as handle:
        offsets = handle["frames/event_offsets"][:]
        width = int(handle["meta/sensor_width"][()])
        height = int(handle["meta/sensor_height"][()])
        event_x = handle["events/x"]
        event_y = handle["events/y"]
        event_p = handle["events/p"]
        for index, frame in enumerate(annotations.get("frames", [])):
            if index + 1 >= len(offsets):
                break
            start = int(offsets[index])
            end = int(offsets[index + 1])
            grid = None
            positive = None
            if end > start:
                xs = event_x[start:end].astype(np.int64)
                ys = event_y[start:end].astype(np.int64)
                polarity = event_p[start:end]
                pixels = ys * width + xs
                grid = np.bincount(
                    pixels, minlength=width * height
                ).reshape(height, width)
                positive = np.bincount(
                    pixels[polarity > 0], minlength=width * height
                ).reshape(height, width)
            for annotation in frame.get("annotations", []):
                x, y, box_w, box_h = annotation["bbox"]
                if grid is None or box_w <= 0 or box_h <= 0:
                    event_count = 0
                    active = 0
                    on_events = 0
                else:
                    region = grid[y : y + box_h, x : x + box_w]
                    event_count = int(region.sum())
                    active = int(np.count_nonzero(region))
                    on_events = int(positive[y : y + box_h, x : x + box_w].sum())
                area = max(1, box_w * box_h)
                annotation["event_count"] = event_count
                annotation["event_density"] = round(event_count / area, 4)
                annotation["active_pixel_ratio"] = round(active / area, 4)
                annotation["on_off_ratio"] = (
                    round(on_events / max(1, event_count - on_events), 4)
                    if event_count
                    else None
                )
                flags = [
                    flag
                    for flag in annotation.get("quality_flags", [])
                    if flag not in ("low_event", "static")
                ]
                if event_count == 0:
                    flags.append("low_event")
                if annotation.get("stationary"):
                    flags.append("static")
                annotation["quality_flags"] = flags
                counts["instances"] += 1
                counts["low_event"] += int(event_count == 0)
                counts["static"] += int(bool(annotation.get("stationary")))
                counts["occluded"] += int(bool(annotation.get("occluded")))
    return counts


def stats_dataset(dataset_root: Path, np: Any) -> dict[str, Any]:
    """给整个数据集的每个机位补上事件统计，并汇总成报告。"""
    report: dict[str, Any] = {"sequences": {}, "totals": {}}
    totals = {"instances": 0, "low_event": 0, "static": 0, "occluded": 0}
    for sequence_root in sorted(
        (dataset_root / "sequences").glob("sequence_*")
    ):
        annotations_path = sequence_root / "cameras"
        if not annotations_path.is_dir():
            continue
        sequence_report: dict[str, Any] = {}
        for camera_dir in sorted(annotations_path.iterdir()):
            annotation_file = camera_dir / "annotations.json"
            events_file = camera_dir / "events.h5"
            if not annotation_file.is_file() or not events_file.is_file():
                continue
            annotations = _load_json(annotation_file)
            counts = annotate_camera(annotations, events_file, np)
            _write_json(annotation_file, annotations)
            sequence_report[camera_dir.name] = counts
            for key in totals:
                totals[key] += counts.get(key, 0)
        if sequence_report:
            report["sequences"][sequence_root.name] = sequence_report
    report["totals"] = totals
    low_ratio = (
        round(totals["low_event"] / totals["instances"], 4)
        if totals["instances"]
        else 0.0
    )
    report["low_event_ratio"] = low_ratio
    _write_json(dataset_root / "annotation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="离线标注：重投影与事件统计")
    subparsers = parser.add_subparsers(dest="command", required=True)

    project_parser = subparsers.add_parser(
        "project", help="用 snapshot.json 重算 annotations.json"
    )
    project_parser.add_argument(
        "--sequence", type=Path, required=True, help="序列目录"
    )
    project_parser.add_argument(
        "--min-detect-px",
        type=int,
        default=MIN_DETECT_SHORT_SIDE_PX,
        help="可探测的最小投影短边（像素），须与采集时一致",
    )
    project_parser.add_argument(
        "--max-detect-distance",
        type=float,
        default=MAX_DETECT_DISTANCE_M,
        help="最大标注距离（米），须与采集时一致",
    )

    stats_parser = subparsers.add_parser(
        "stats", help="统计框内事件并写回 quality_flags"
    )
    stats_parser.add_argument(
        "--dataset", type=Path, default=Path("dvs_roadside_multicam")
    )

    args = parser.parse_args()
    import numpy as np

    if args.command == "project":
        cameras = project_sequence(
            args.sequence.resolve(),
            np,
            min_detect_px=args.min_detect_px,
            max_detect_distance=args.max_detect_distance,
        )
        print(f"已重投影：{args.sequence}（机位 {', '.join(cameras)}）")
        return 0
    report = stats_dataset(args.dataset.resolve(), np)
    totals = report["totals"]
    print(
        f"事件统计完成：{totals['instances']} 个实例，"
        f"零事件 {totals['low_event']} 个（{report['low_event_ratio']:.1%}），"
        f"静止 {totals['static']} 个，遮挡 {totals['occluded']} 个"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
