#!/usr/bin/env python3
"""Validate the roadside dataset layout and per-camera synchronization."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


SEQUENCE_PATTERN = re.compile(r"^sequence_\d{6}$")

#: 相邻帧之间框面积/尺寸突变超过这个倍数就报警（方案 §7.3 的轨迹突变检查）。
#: 阈值按实测定的：真实数据里 99.9% 的相邻帧面积变化 < 2×，超过 6× 基本是
#: actor id 复用或 3D 框朝向突变（车从正面变成侧面），需要人看一眼。
AREA_JUMP_RATIO = 6.0
SIZE_JUMP_RATIO = 2.5


def _read_timestamps(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_annotations(
    annotations: dict[str, Any],
    width: int,
    height: int,
    label: str,
) -> tuple[list[str], list[str]]:
    """检查标注本身：框合法、同帧不重复、ignore 标记一致、轨迹突变。

    返回 ``(errors, warnings)``：框非法、同 actor 重复、ignore 没有原因都是
    硬错误；相邻帧的框尺寸突变只是报警（远处目标本来就小，比例噪声大）。
    """
    errors: list[str] = []
    warnings: list[str] = []
    previous: dict[int, tuple[list[float], float]] = {}
    for index, frame in enumerate(annotations.get("frames", [])):
        seen: set[Any] = set()
        for annotation in frame.get("annotations", []):
            actor_id = annotation.get("actor_id")
            bbox = annotation.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                errors.append(f"{label} 第 {index} 帧的 bbox 格式不对：{bbox}")
                continue
            values = []
            for value in bbox:
                if not isinstance(value, (int, float)) or value != value:
                    errors.append(f"{label} 第 {index} 帧 actor {actor_id} 的 bbox 含非法值")
                    values = []
                    break
                values.append(float(value))
            if len(values) != 4:
                continue
            x, y, box_w, box_h = values
            if box_w <= 0 or box_h <= 0:
                errors.append(f"{label} 第 {index} 帧 actor {actor_id} 的框宽高非正")
            if x < 0 or y < 0 or x + box_w > width or y + box_h > height:
                errors.append(f"{label} 第 {index} 帧 actor {actor_id} 的框越界")
            if actor_id is not None:
                if actor_id in seen:
                    errors.append(f"{label} 第 {index} 帧 actor {actor_id} 重复标注")
                seen.add(actor_id)
            if annotation.get("ignore") and not (
                annotation.get("ignore_reason")
                or annotation.get("quality_flags")
            ):
                errors.append(
                    f"{label} 第 {index} 帧 actor {actor_id} 标了 ignore 却没有原因"
                )
            area = box_w * box_h
            if actor_id is not None:
                previous_box = previous.get(actor_id)
                if previous_box is not None:
                    last_box, last_area = previous_box
                    if last_area > 0 and area > 0:
                        ratio = max(area / last_area, last_area / area)
                        if ratio > AREA_JUMP_RATIO:
                            warnings.append(
                                f"{label} 第 {index} 帧 actor {actor_id} 的框面积突变 "
                                f"{ratio:.1f}×"
                            )
                    for side, last_value, value in (
                        ("宽", last_box[2], box_w),
                        ("高", last_box[3], box_h),
                    ):
                        if last_value > 0 and value > 0:
                            side_ratio = max(value / last_value, last_value / value)
                            if side_ratio > SIZE_JUMP_RATIO:
                                warnings.append(
                                    f"{label} 第 {index} 帧 actor {actor_id} 的框{side}突变 "
                                    f"{side_ratio:.1f}×"
                                )
                previous[actor_id] = (values, area)
    return errors, warnings


def _validate_camera(
    sequence_id: str,
    camera_dir: Path,
) -> tuple[list[int], list[str], list[str]]:
    import h5py
    import numpy as np

    errors: list[str] = []
    warnings: list[str] = []
    timestamps_path = camera_dir / "timestamps.txt"
    rgb_dir = camera_dir / "rgb"
    annotation_path = camera_dir / "annotations.json"
    event_path = camera_dir / "events.h5"
    for required in (timestamps_path, rgb_dir, annotation_path, event_path):
        if not required.exists():
            errors.append(f"缺少文件或目录：{required}")
    if errors:
        return [], errors, warnings

    try:
        timestamps = _read_timestamps(timestamps_path)
        if not timestamps:
            errors.append(f"{timestamps_path} 中没有时间戳")
        if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
            errors.append(f"{timestamps_path} 时间戳不是严格递增")
        rgb_frames = sorted(
            path for path in rgb_dir.iterdir() if path.suffix in (".png", ".jpg")
        )
        if len(rgb_frames) != len(timestamps):
            errors.append(
                f"{camera_dir.name} RGB 帧数 {len(rgb_frames)} "
                f"与时间戳数 {len(timestamps)} 不一致"
            )
        suffix = rgb_frames[0].suffix if rgb_frames else ".png"
        expected_names = [f"{index:06d}{suffix}" for index in range(len(rgb_frames))]
        if [path.name for path in rgb_frames] != expected_names:
            errors.append(f"{rgb_dir} 的 RGB 帧名不连续或格式错误")

        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(annotations, dict):
            errors.append(f"{annotation_path} 根节点必须是 JSON 对象")
            return timestamps, errors, warnings
        if annotations.get("sequence_id") != sequence_id:
            errors.append(f"{annotation_path} sequence_id 不匹配")
        if annotations.get("camera_id") != camera_dir.name:
            errors.append(f"{annotation_path} camera_id 不匹配")
        annotation_frames = annotations.get("frames")
        if not isinstance(annotation_frames, list) or len(annotation_frames) != len(
            timestamps
        ):
            errors.append(f"{annotation_path} 标注帧数与时间戳数不一致")
        elif any(not isinstance(frame, dict) for frame in annotation_frames):
            errors.append(f"{annotation_path} 的 frames 元素必须是 JSON 对象")
        elif [frame.get("timestamp_us") for frame in annotation_frames] != timestamps:
            errors.append(f"{annotation_path} 标注时间戳与 timestamps.txt 不一致")
        elif [
            frame.get("frame_id") for frame in annotation_frames
        ] != [
            f"{sequence_id}_{camera_dir.name}_{index:06d}"
            for index in range(len(timestamps))
        ]:
            errors.append(f"{annotation_path} 帧 ID 不连续或格式错误")
        else:
            annotation_errors, annotation_warnings = validate_annotations(
                annotations,
                int(annotations.get("sensor_width", 0)),
                int(annotations.get("sensor_height", 0)),
                f"{sequence_id}/{camera_dir.name}",
            )
            errors.extend(annotation_errors)
            warnings.extend(annotation_warnings)

        with h5py.File(event_path, "r") as handle:
            for dataset in ("events/t", "events/x", "events/y", "events/p"):
                if dataset not in handle:
                    errors.append(f"{event_path} 缺少 /{dataset}")
            for metadata in (
                "sensor_width",
                "sensor_height",
                "sequence_id",
                "camera_id",
                "synced_rgb",
                "town",
                "weather",
                "dvs_params",
            ):
                if f"meta/{metadata}" not in handle:
                    errors.append(f"{event_path} 缺少 /meta/{metadata}")
            if errors:
                return timestamps, errors, warnings

            width = int(handle["meta/sensor_width"][()])
            height = int(handle["meta/sensor_height"][()])
            expected_dtypes = {
                "events/t": ("u", 8),
                "events/x": ("u", 2),
                "events/y": ("u", 2),
                "events/p": ("i", 1),
                "meta/sensor_width": ("u", 2),
                "meta/sensor_height": ("u", 2),
            }
            for dataset, (kind, itemsize) in expected_dtypes.items():
                dtype = handle[dataset].dtype
                if dtype.kind != kind or dtype.itemsize != itemsize:
                    errors.append(
                        f"{event_path} /{dataset} dtype 应为 {kind}{itemsize * 8}"
                    )
            stored_sequence = handle["meta/sequence_id"][()]
            stored_camera = handle["meta/camera_id"][()]
            if isinstance(stored_sequence, bytes):
                stored_sequence = stored_sequence.decode("utf-8")
            if isinstance(stored_camera, bytes):
                stored_camera = stored_camera.decode("utf-8")
            if stored_sequence != sequence_id:
                errors.append(f"{event_path} 的 meta/sequence_id 不匹配")
            if stored_camera != camera_dir.name:
                errors.append(f"{event_path} 的 meta/camera_id 不匹配")
            count = len(handle["events/t"])
            previous_timestamp: int | None = None
            chunk_size = 1_000_000
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                times = handle["events/t"][start:end]
                if times.size:
                    if previous_timestamp is not None and int(times[0]) <= previous_timestamp:
                        errors.append(f"{event_path} 事件时间戳不是严格递增")
                    if np.any(times[1:] <= times[:-1]):
                        errors.append(f"{event_path} 事件时间戳不是严格递增")
                    previous_timestamp = int(times[-1])
                xs = handle["events/x"][start:end]
                ys = handle["events/y"][start:end]
                polarities = handle["events/p"][start:end]
                if np.any(xs >= width) or np.any(ys >= height):
                    errors.append(f"{event_path} 存在超出分辨率的事件坐标")
                if np.any((polarities != 1) & (polarities != -1)):
                    errors.append(f"{event_path} 存在非法事件极性")
            if "frames/event_offsets" not in handle:
                errors.append(f"{event_path} 缺少 /frames/event_offsets")
            else:
                offsets = handle["frames/event_offsets"]
                if len(offsets) != len(timestamps) + 1:
                    errors.append(f"{event_path} 帧事件偏移数量不正确")
                elif len(offsets) and int(offsets[-1]) != count:
                    errors.append(f"{event_path} 最终事件偏移与事件数不一致")
            errors.extend(_validate_event_time_unit(handle, event_path, timestamps))
        return timestamps, errors, warnings
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return [], [f"校验 {camera_dir} 时读取失败：{exc}"], warnings


def _validate_event_time_unit(
    handle: Any,
    event_path: Path,
    timestamps: list[int],
) -> list[str]:
    """检查事件时间戳的单位是否和 RGB 帧时间一致。

    事件时间轴必须落在对应帧的曝光窗口内；单位写错（纳秒当微秒存）会让
    事件时间比帧时间大三个数量级，这里直接把它挑出来。
    """
    errors: list[str] = []
    unit = "ns"
    if "meta/event_timestamp_unit" in handle:
        value = handle["meta/event_timestamp_unit"][()]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        unit = str(value)
    scale = 1000.0 if unit == "ns" else 1.0
    integration = 0
    if "meta/integration_time_us" in handle:
        integration = int(handle["meta/integration_time_us"][()])
    if "frames/event_offsets" not in handle or not timestamps:
        return errors
    offsets = handle["frames/event_offsets"]
    samples = sorted({0, len(timestamps) // 2, len(timestamps) - 1})
    for index in samples:
        if index + 1 >= len(offsets):
            continue
        start, end = int(offsets[index]), int(offsets[index + 1])
        if end <= start:
            continue
        last_event_us = int(handle["events/t"][end - 1]) / scale
        frame_us = timestamps[index]
        # 事件发生在该帧曝光窗口内：略早于帧时刻，最多提前一个积分时间。
        if not (
            frame_us - integration - 5000 <= last_event_us <= frame_us + 5000
        ):
            errors.append(
                f"{event_path} 事件时间戳与 RGB 帧时间不一致："
                f"第 {index} 帧事件时刻 {last_event_us:.0f} us，帧时刻 {frame_us} us，"
                f"meta 声明单位 {unit}；"
                "若实际是纳秒，请运行 tools/fix_event_timestamps.py 修复。"
            )
            break
    return errors


def _validate_location_index(root: Path, sequence_ids: list[str]) -> list[str]:
    """检查位置时间索引：每条序列都要有位置和时间标注。"""
    path = root / "locations.csv"
    if not path.is_file():
        return [f"缺少位置时间索引：{path}"]
    errors: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    indexed = {row.get("sequence_id"): row for row in rows}
    for sequence_id in sequence_ids:
        row = indexed.get(sequence_id)
        if row is None:
            errors.append(f"{sequence_id} 在 locations.csv 中没有位置时间记录")
            continue
        for column in (
            "capture_started_at_utc",
            "capture_started_at_local",
            "town",
            "map_name",
        ):
            if not (row.get(column) or "").strip():
                errors.append(f"{sequence_id} 的 locations.csv 缺少 {column}")
        if not (row.get("camera_poses") or "").strip():
            errors.append(f"{sequence_id} 的 locations.csv 缺少机位位置")
    for sequence_id in indexed:
        if sequence_id not in sequence_ids:
            errors.append(f"locations.csv 中的 {sequence_id} 未登记在 index_all.txt")
    return errors


def validate_dataset(root: Path) -> tuple[list[str], list[str]]:
    index = root / "index_all.txt"
    if not index.is_file():
        return [f"缺少全量索引：{index}"], []
    sequence_ids = [line.strip() for line in index.read_text(encoding="utf-8").splitlines()]
    errors: list[str] = []
    warnings: list[str] = []
    if len(sequence_ids) != len(set(sequence_ids)):
        errors.append(f"{index} 中存在重复序列 ID")
    # 索引可能指向已被删除的序列（实测索引 13 条、磁盘 8 条），那会让质检
    # 对不存在的目录刷一堆"缺少文件"。索引只用来做一致性核对，实际检查
    # 按磁盘上的序列目录走。
    on_disk = sorted(
        path.name
        for path in (root / "sequences").iterdir()
        if path.is_dir() and SEQUENCE_PATTERN.fullmatch(path.name)
    ) if (root / "sequences").is_dir() else []
    for missing in sorted(set(sequence_ids) - set(on_disk)):
        errors.append(
            f"index_all.txt 里的 {missing} 在 sequences/ 下不存在（索引与磁盘不一致）"
        )
    for sequence_id in on_disk:
        sequence_root = root / "sequences" / sequence_id
        cameras_root = sequence_root / "cameras"
        for calibration_file in (
            "extrinsics_world.yaml",
            "intrinsics_dvs.yaml",
            "intrinsics_rgb.yaml",
            "dvs_rgb_extrinsic.yaml",
            "time_offset.yaml",
        ):
            path = sequence_root / "calibration" / calibration_file
            if not path.is_file():
                errors.append(f"{sequence_id} 缺少标定文件：{path}")
        for metadata_file in (
            "episode_manifest.json",
            "dvs_params.json",
            "rig_geometry.json",
        ):
            path = sequence_root / "meta" / metadata_file
            if not path.is_file():
                errors.append(f"{sequence_id} 缺少元数据文件：{path}")
        manifest_path = sequence_root / "meta" / "episode_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"{sequence_id} 的 episode_manifest.json 解析失败：{exc}")
            else:
                if not manifest.get("location"):
                    errors.append(f"{sequence_id} 的清单缺少采集位置 location")
                if not manifest.get("capture_time"):
                    errors.append(f"{sequence_id} 的清单缺少采集时间 capture_time")
        if not cameras_root.is_dir():
            errors.append(f"{sequence_id} 缺少 cameras 目录")
            continue
        camera_dirs = sorted(path for path in cameras_root.iterdir() if path.is_dir())
        if not 1 <= len(camera_dirs) <= 8:
            errors.append(f"{sequence_id} 应包含 1–8 台相机，实际 {len(camera_dirs)} 台")
        reference_timestamps: list[int] | None = None
        for camera_dir in camera_dirs:
            timestamps, camera_errors, camera_warnings = _validate_camera(
                sequence_id, camera_dir
            )
            errors.extend(camera_errors)
            warnings.extend(camera_warnings)
            if reference_timestamps is None:
                reference_timestamps = timestamps
            elif timestamps != reference_timestamps:
                errors.append(f"{sequence_id} 多机位 RGB 时间戳不一致：{camera_dir.name}")
    errors.extend(_validate_location_index(root, sequence_ids))
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="校验多机位数据集结构和同步关系")
    parser.add_argument(
        "--dataset", type=Path, default=Path("dvs_roadside_multicam")
    )
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    errors, warnings = validate_dataset(root)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"数据集校验失败，共 {len(errors)} 项问题、{len(warnings)} 项报警。",
            file=sys.stderr,
        )
        return 1
    print(f"数据集校验通过：{root}（{len(warnings)} 项报警）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
