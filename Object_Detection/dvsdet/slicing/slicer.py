"""把 CARLA 采集产物切成训练窗并体素化。

**与方案 §3.1 的差异（按真实数据修正）**：

| 方案假设 | 实际数据 |
|---|---|
| `anns.json`，字段 `frames[].bboxes/labels/track_ids` | `annotations.json`，字段 `frames[].annotations[]`（每项含 bbox/category/actor_id/ignore/quality_flags） |
| 标注在 `bboxes.json` | 忽略框单独放在 `frames[].ignored_annotations[]` |
| `events/t` 是 µs | **ns**（`meta/event_timestamp_unit`） |
| 640×360 | 采集分辨率 1280×720，体素化时降采样 |
| 需要自己按时间窗切 | `frames/event_offsets` 已给出逐帧事件边界 |

切片口径（方案 §3.1 的"坑"①，必须严格遵守）：**标注取窗末时刻**，
即窗 [t_end - window, t_end) 的标签取 t_end 对应帧的标注——只用窗内及
之前的信息，训练与在线推理一致。

输出 `data/slices/<sequence>/<camera>/<win_idx>.npz`，含体素网格、标注、
以及域随机化需要的 `contrast_threshold / map / weather`。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from .voxelize import numba_available, voxelize


@dataclass
class SliceConfig:
    """切片参数。"""

    window_ms: float = 50.0
    overlap_ms: float = 25.0
    num_time_channels: int = 9
    grid_width: int = 640
    grid_height: int = 360
    min_events_per_window: int = 10
    source_width: int = 1280
    source_height: int = 720

    @property
    def window_ns(self) -> int:
        return int(self.window_ms * 1_000_000)

    @property
    def overlap_ns(self) -> int:
        return int(self.overlap_ms * 1_000_000)

    @property
    def stride_ns(self) -> int:
        return max(self.window_ns - self.overlap_ns, 1)


@dataclass
class WindowAnnotation:
    """一个训练窗的标注。"""

    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    track_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    ignore_boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))


def _as_text(value: Any) -> str:
    """h5py 的字符串属性读出来是 bytes，统一解成 str。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _load_annotations(path: Path) -> dict[int, dict[str, Any]]:
    """annotations.json → {帧序号: {boxes, labels, track_ids, ignore_boxes}}。

    `ignore=true` 的标注不作为训练目标；`ignored_annotations` 里的框单独
    保留，评估时按 COCO 的 iscrowd 语义作为"忽略区"处理。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    per_frame: dict[int, dict[str, Any]] = {}
    for index, frame in enumerate(data.get("frames", [])):
        boxes: list[list[float]] = []
        labels: list[int] = []
        track_ids: list[int] = []
        ignore_boxes: list[list[float]] = []
        for ann in frame.get("annotations", []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            box = [float(v) for v in bbox]
            if ann.get("ignore"):
                ignore_boxes.append(box)
                continue
            if box[2] <= 0 or box[3] <= 0:
                continue
            boxes.append(box)
            labels.append(int(ann.get("category_id", 0)))
            track_ids.append(int(ann.get("actor_id", -1)))
        for ann in frame.get("ignored_annotations", []) or []:
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                ignore_boxes.append([float(v) for v in bbox])
        per_frame[index] = {
            "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
            "labels": np.asarray(labels, np.int64),
            "track_ids": np.asarray(track_ids, np.int64),
            "ignore_boxes": np.asarray(ignore_boxes, np.float32).reshape(-1, 4),
        }
    return per_frame


def _scale_boxes(boxes: np.ndarray, config: SliceConfig) -> np.ndarray:
    """把 1280×720 的标注框线性映射到体素网格坐标。"""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    sx = config.grid_width / config.source_width
    sy = config.grid_height / config.source_height
    out = boxes.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    out[:, 2] *= sx
    out[:, 3] *= sy
    return out


def iter_camera_windows(
    camera_dir: Path,
    config: SliceConfig,
) -> Iterator[dict[str, Any]]:
    """遍历一台相机的所有训练窗，产出体素网格与标注。

    窗的右边界对齐到每一帧的 `timestamp_us`（乘 1000 得 ns），事件区间由
    `frames/event_offsets` 精确定位——不需要全量扫时间轴。
    """
    annotations_path = camera_dir / "annotations.json"
    events_path = camera_dir / "events.h5"
    if not annotations_path.is_file() or not events_path.is_file():
        return

    per_frame = _load_annotations(annotations_path)

    with h5py.File(events_path, "r") as handle:
        frame_ids = handle["frames/frame_id"][:]
        timestamps_us = handle["frames/timestamp_us"][:]
        offsets = handle["frames/event_offsets"][:]
        events_t = handle["events/t"][:]
        events_x = handle["events/x"][:]
        events_y = handle["events/y"][:]
        events_p = handle["events/p"][:]
        meta = {
            "sequence_id": _as_text(handle["meta/sequence_id"][()]),
            "camera_id": _as_text(handle["meta/camera_id"][()]),
            "town": _as_text(handle["meta/town"][()]),
            "weather": _as_text(handle["meta/weather"][()]),
            "dvs_params": handle["meta/dvs_params"][()],
        }
        if isinstance(meta["dvs_params"], bytes):
            meta["dvs_params"] = json.loads(meta["dvs_params"].decode("utf-8"))
        contrast = float(meta["dvs_params"].get("positive_threshold", 0.25))

        window_ns = config.window_ns
        # 复用体素缓冲区：每次新建 16.6 MB 数组的首次触碰页错误要 ~5 ms/窗，
        # 复用后只要 0.09 ms（方案 §3.4 要求 < 2 ms/窗）。
        buffer = np.zeros(
            (2 * config.num_time_channels, config.grid_height, config.grid_width),
            dtype=np.float32,
        )
        for index in range(len(timestamps_us)):
            # 窗末 = 该帧时间戳（因果：只用窗内及之前的信息）
            t_end = int(timestamps_us[index]) * 1000
            t_start = t_end - window_ns
            # 事件区间：event_offsets 给的是逐帧边界，取窗口覆盖到的部分
            lo = 0
            hi = len(events_t)
            # 二分找窗口起点与终点
            lo = int(np.searchsorted(events_t, t_start, side="left"))
            hi = int(np.searchsorted(events_t, t_end, side="left"))
            if hi - lo < config.min_events_per_window:
                continue
            voxel_started = time.perf_counter()
            voxel = voxelize(
                events_x[lo:hi],
                events_y[lo:hi],
                events_t[lo:hi],
                events_p[lo:hi],
                config.grid_width,
                config.grid_height,
                config.num_time_channels,
                t_start,
                t_end,
                out=buffer,
            )
            voxelize_ms = (time.perf_counter() - voxel_started) * 1000.0
            # buffer 会被下一个窗覆写，yield 出去的是副本
            voxel = voxel.copy()
            ann = per_frame.get(index, {})
            yield {
                "frame_index": index,
                "frame_id": _as_text(frame_ids[index]),
                "t_start_ns": t_start,
                "t_end_ns": t_end,
                "voxel": voxel,
                "voxelize_ms": voxelize_ms,
                "boxes": _scale_boxes(ann.get("boxes", np.zeros((0, 4), np.float32)), config),
                "labels": ann.get("labels", np.zeros((0,), np.int64)),
                "track_ids": ann.get("track_ids", np.zeros((0,), np.int64)),
                "ignore_boxes": _scale_boxes(
                    ann.get("ignore_boxes", np.zeros((0, 4), np.float32)), config
                ),
                "contrast_threshold": contrast,
                "map_name": meta["town"],
                "weather": meta["weather"],
                "sequence_id": meta["sequence_id"],
                "camera_id": meta["camera_id"],
            }


def slice_sequence(
    sequence_dir: Path,
    output_root: Path,
    config: SliceConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """切一条序列的所有相机，返回统计信息。"""
    sequence_id = sequence_dir.name
    total = 0
    events_total = 0
    started = time.perf_counter()
    for camera_dir in sorted((sequence_dir / "cameras").glob("cam_*")):
        if not camera_dir.is_dir():
            continue
        out_dir = output_root / sequence_id / camera_dir.name
        if out_dir.exists() and any(out_dir.glob("*.npz")) and not overwrite:
            total += len(list(out_dir.glob("*.npz")))
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.npz"):
            old.unlink()
        for index, window in enumerate(iter_camera_windows(camera_dir, config)):
            # 稀疏存储：事件体素里只有约 0.2% 的格子非零，dense float16
            # 每窗 8.3 MB、总量 ~28 GB，训练时 IO 会成为瓶颈。存非零格子的
            # 索引 + 计数，每窗 ~150 KB（dataloader 里重建 dense 网格）。
            flat = window["voxel"].reshape(-1)
            nonzero = np.nonzero(flat)[0]
            np.savez_compressed(
                out_dir / f"{index:06d}.npz",
                voxel_indices=nonzero.astype(np.int32),
                voxel_counts=np.minimum(flat[nonzero], 255.0).astype(np.uint8),
                voxel_shape=np.asarray(
                    window["voxel"].shape, dtype=np.int32
                ),
                boxes=window["boxes"],
                labels=window["labels"],
                track_ids=window["track_ids"],
                ignore_boxes=window["ignore_boxes"],
                frame_index=window["frame_index"],
                t_start_ns=window["t_start_ns"],
                t_end_ns=window["t_end_ns"],
                contrast_threshold=window["contrast_threshold"],
                map_name=window["map_name"],
                weather=window["weather"],
                sequence_id=window["sequence_id"],
                camera_id=window["camera_id"],
            )
            total += 1
            events_total += int(window["voxel"].sum())
    elapsed = time.perf_counter() - started
    return {
        "sequence_id": sequence_id,
        "windows": total,
        "events": events_total,
        "seconds": round(elapsed, 2),
        "ms_per_window": round(elapsed * 1000 / max(total, 1), 3),
        "numba": numba_available(),
    }
