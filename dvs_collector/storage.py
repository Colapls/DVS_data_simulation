from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from .config import DVS_EVENT_DTYPE


class CameraWriter:
    """Write one spec-compliant event sequence and its frame index to HDF5."""

    def __init__(
        self,
        path: Path,
        h5py: Any,
        np: Any,
        episode_id: str,
        sequence_id: str,
        camera_id: str,
        camera_mount: dict[str, Any],
        synced_rgb: str,
        width: int,
        height: int,
        time_origin_us: int,
        integration_time_us: int,
        flush_frames: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.np = np
        self.flush_frames = flush_frames
        self.time_origin_us = time_origin_us
        # 事件时间戳以纳秒保存：CARLA 的 DVS 原始时间戳就是纳秒，而 20 Hz 下
        # 每帧约 4 万个事件，微秒分辨率会出现大量同刻事件，无法保持严格递增。
        self.time_origin_ns = time_origin_us * 1000
        # 原始时间戳到纳秒的倍数：0.9.16 给纳秒（1），早期版本给微秒（1000）。
        self.source_scale: int | None = None
        self._string_dtype = h5py.string_dtype(encoding="utf-8")
        self.last_event_timestamp = -1
        self.timestamp_adjusted_event_count = 0
        # 没有任何事件的帧数：画面静止时 DVS 不发数据，这一帧就是空的。
        # 用来在清单里标出"传感器其实没在工作"。
        self.empty_frame_count = 0
        self.closed = False
        self.file = h5py.File(path, "w")

        events = self.file.create_group("events")
        meta = self.file.create_group("meta")
        frames = self.file.create_group("frames")
        self.events_t = events.create_dataset(
            "t", shape=(0,), maxshape=(None,), dtype="u8", chunks=(65536,), compression="lzf"
        )
        self.events_x = events.create_dataset(
            "x", shape=(0,), maxshape=(None,), dtype="u2", chunks=(65536,), compression="lzf"
        )
        self.events_y = events.create_dataset(
            "y", shape=(0,), maxshape=(None,), dtype="u2", chunks=(65536,), compression="lzf"
        )
        self.events_p = events.create_dataset(
            "p", shape=(0,), maxshape=(None,), dtype="i1", chunks=(65536,), compression="lzf"
        )
        meta.create_dataset("sensor_width", data=width, dtype="u2")
        meta.create_dataset("sensor_height", data=height, dtype="u2")
        meta.create_dataset(
            "episode_id", data=episode_id, dtype=h5py.string_dtype(encoding="utf-8")
        )
        meta.create_dataset(
            "sequence_id",
            data=sequence_id,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        meta.create_dataset(
            "synced_rgb",
            data=synced_rgb,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        meta.create_dataset("time_origin_us", data=time_origin_us, dtype="u8")
        meta.create_dataset("time_origin_ns", data=self.time_origin_ns, dtype="u8")
        meta.create_dataset("integration_time_us", data=integration_time_us, dtype="u8")
        meta.create_dataset(
            "event_timestamp_unit",
            data="ns",
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        meta.create_dataset(
            "polarity_encoding",
            data="1=ON,-1=OFF",
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        meta.create_dataset(
            "timestamp_tie_policy",
            data="timestamps are advanced by 1 ns as needed to ensure strict ordering",
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self.timestamp_adjusted_events = meta.create_dataset(
            "timestamp_adjusted_event_count", data=0, dtype="u8"
        )
        meta.create_dataset(
            "camera_id", data=camera_id, dtype=h5py.string_dtype(encoding="utf-8")
        )
        meta.create_dataset(
            "camera_mount",
            data=json.dumps(camera_mount, ensure_ascii=False),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        if extra_metadata:
            for name, value in extra_metadata.items():
                meta.create_dataset(
                    name,
                    data=json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
        self.frame_offsets = frames.create_dataset(
            "event_offsets", data=np.array([0], dtype=np.uint64), maxshape=(None,), chunks=True
        )
        self.frame_ids = frames.create_dataset(
            "frame_id",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
        )
        self.frame_timestamps = frames.create_dataset(
            "timestamp_us", shape=(0,), maxshape=(None,), dtype="u8", chunks=True
        )
        self.frame_event_counts = frames.create_dataset(
            "event_count", shape=(0,), maxshape=(None,), dtype="u8", chunks=True
        )
        self.pending_x: list[Any] = []
        self.pending_y: list[Any] = []
        self.pending_t: list[Any] = []
        self.pending_p: list[Any] = []
        self.pending_ids: list[str] = []
        self.pending_timestamps: list[int] = []
        self.pending_counts: list[int] = []

    def _resolve_source_scale(self, raw_timestamps: Any, timestamp_us: int) -> int:
        """判定 CARLA 原始事件时间戳的单位，返回换算到纳秒的倍数。

        0.9.16 的 DVS 传感器给的是纳秒，早期版本给的是微秒。这里把原始时间戳
        分别按两种假设和该帧的绝对时间比较，取更接近的一种。
        # ponytail: 世界刚起步（绝对时间为 0）时判定会偏向纳秒；实际采集时
        # 世界已运行数十秒，不影响。
        """
        if self.source_scale is None:
            frame_abs_us = max(self.time_origin_us + timestamp_us, 1)
            raw_max = int(raw_timestamps.max())
            as_nanoseconds = abs(raw_max // 1000 - frame_abs_us)
            as_microseconds = abs(raw_max - frame_abs_us)
            self.source_scale = 1 if as_nanoseconds < as_microseconds else 1000
        return self.source_scale

    def add_frame(
        self,
        frame_id: str,
        timestamp_us: int,
        raw_data: bytes,
    ) -> int:
        events = (
            self.np.frombuffer(raw_data, dtype=self.np.dtype(DVS_EVENT_DTYPE))
            if raw_data
            # 这一帧没有事件（DVS 在画面静止时根本不发数据）：空帧是合法数据。
            else self.np.empty(0, dtype=self.np.dtype(DVS_EVENT_DTYPE))
        )
        order = self.np.argsort(events["t"], kind="stable")
        ordered = events[order]
        raw_timestamps = ordered["t"].astype(self.np.int64, copy=True)
        if raw_timestamps.size:
            scale = self._resolve_source_scale(raw_timestamps, timestamp_us)
            original_timestamps = raw_timestamps * scale - self.time_origin_ns
        else:
            original_timestamps = raw_timestamps
        timestamps = original_timestamps.copy()

        if timestamps.size:
            timestamps = self.np.maximum(timestamps, 0)
            timestamps[0] = max(
                int(timestamps[0]), self.last_event_timestamp + 1
            )
            indices = self.np.arange(timestamps.size, dtype=self.np.int64)
            timestamps = self.np.maximum.accumulate(timestamps - indices) + indices
            self.last_event_timestamp = int(timestamps[-1])
            self.timestamp_adjusted_event_count += int(
                self.np.count_nonzero(timestamps != original_timestamps)
            )

        self.pending_t.append(timestamps.astype(self.np.uint64, copy=False))
        self.pending_x.append(ordered["x"].copy())
        self.pending_y.append(ordered["y"].copy())
        self.pending_p.append(
            self.np.where(ordered["pol"], 1, -1).astype(self.np.int8)
        )
        event_count = int(events.size)
        if event_count == 0:
            self.empty_frame_count += 1
        self.pending_ids.append(frame_id)
        self.pending_timestamps.append(timestamp_us)
        self.pending_counts.append(event_count)
        if len(self.pending_ids) >= self.flush_frames:
            self.flush()
        return event_count

    def flush(self) -> None:
        if not self.pending_ids:
            return
        t_values = self.np.concatenate(self.pending_t)
        x_values = self.np.concatenate(self.pending_x)
        y_values = self.np.concatenate(self.pending_y)
        p_values = self.np.concatenate(self.pending_p)

        event_start = self.events_t.shape[0]
        event_end = event_start + t_values.size
        for dataset, values in (
            (self.events_t, t_values),
            (self.events_x, x_values),
            (self.events_y, y_values),
            (self.events_p, p_values),
        ):
            dataset.resize((event_end,))
            dataset[event_start:event_end] = values

        offsets = event_start + self.np.cumsum(self.pending_counts, dtype=self.np.uint64)
        offset_start = self.frame_offsets.shape[0]
        self.frame_offsets.resize((offset_start + len(offsets),))
        self.frame_offsets[offset_start:] = offsets

        for dataset, values in (
            (self.frame_ids, self.pending_ids),
            (self.frame_timestamps, self.pending_timestamps),
            (self.frame_event_counts, self.pending_counts),
        ):
            start = dataset.shape[0]
            dataset.resize((start + len(values),))
            dataset[start:] = values

        self.pending_t.clear()
        self.pending_x.clear()
        self.pending_y.clear()
        self.pending_p.clear()
        self.pending_ids.clear()
        self.pending_timestamps.clear()
        self.pending_counts.clear()
        self.file.flush()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.flush()
        self.timestamp_adjusted_events[()] = self.timestamp_adjusted_event_count
        source_unit = {1: "ns", 1000: "us"}.get(self.source_scale, "unknown")
        self.file["meta"].create_dataset(
            "event_timestamp_source_unit",
            data=source_unit,
            dtype=self._string_dtype,
        )
        self.file.close()


_STOP = None  # 队列哨兵：通知落盘线程收尾


class CameraDiskWriter:
    """单相机后台落盘线程：HDF5 追加、PNG 编码、时间戳写入全部移出采集主循环。

    主循环每帧只把传感器数据入队，磁盘卡顿被队列吸收，不再阻塞传感器等待，
    多相机同时写盘也就不会引发回调超时。单相机单线程保证帧序。
    # ponytail: 队列上限 30 帧（约 130 MB/相机的原始数据）；磁盘长期跟不上时
    # put 会超时反压并报错，而不是无限吃内存。
    """

    QUEUE_SIZE_FRAMES = 30
    PUT_TIMEOUT_S = 60.0

    def __init__(
        self,
        writer: CameraWriter,
        rgb_dir: Path,
        timestamps_path: Path,
        label: str,
        rgb_format: str = "png",
        rgb_quality: int = 90,
    ) -> None:
        self.writer = writer
        self.rgb_dir = rgb_dir
        self.timestamps_path = timestamps_path
        self.label = label
        # PNG 无损但体积大；JPEG q=90 约为 PNG 的 1/10（方案 §8 存储瘦身）。
        self.rgb_suffix = ".jpg" if rgb_format == "jpeg" else ".png"
        self.rgb_quality = rgb_quality
        self.count = 0  # 已落盘的事件数；仅后台线程更新，主线程只读
        self.error: BaseException | None = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.QUEUE_SIZE_FRAMES)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"disk-writer-{label}"
        )
        self._timestamps_file = timestamps_path.open(
            "w", encoding="utf-8", newline="\n"
        )

    def start(self) -> None:
        self._thread.start()

    def submit(
        self,
        frame_index: int,
        frame_id: str,
        timestamp_us: int,
        dvs_raw: bytes,
        rgb_data: Any,
    ) -> None:
        """入队一帧；写入线程出错时立刻把真实原因抛回主循环。

        dvs_raw 是 DVS 事件的原始字节流，可能为空（这一帧没有事件）。
        """
        self.check()
        try:
            self._queue.put(
                (frame_index, frame_id, timestamp_us, dvs_raw, rgb_data),
                timeout=self.PUT_TIMEOUT_S,
            )
        except queue.Full:
            self.check()
            raise RuntimeError(
                f"相机 {self.label} 的落盘队列 {self.PUT_TIMEOUT_S:.0f} 秒未消化："
                "磁盘写入速度长期跟不上采集速度"
            ) from None

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(
                f"相机 {self.label} 的后台写入失败：{self.error}"
            ) from self.error

    def finish(self) -> None:
        """排空队列并等待写入线程结束；磁盘错误在此抛出。"""
        self._queue.put(_STOP)
        self._thread.join()
        self._timestamps_file.close()
        self.check()

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                frame_index, frame_id, timestamp_us, dvs_raw, rgb_data = item
                self.count += self.writer.add_frame(frame_id, timestamp_us, dvs_raw)
                self._save_rgb(rgb_data, frame_index)
                self._timestamps_file.write(f"{timestamp_us}\n")
        except BaseException as exc:  # 磁盘满等：停下，原因留给主循环抛出
            self.error = exc

    def _save_rgb(self, rgb_data: Any, frame_index: int) -> None:
        """写一帧 RGB 图像；JPEG 用 PIL 转 BGRA→RGB 后按指定质量存。"""
        path = self.rgb_dir / f"{frame_index:06d}{self.rgb_suffix}"
        if self.rgb_suffix != ".jpg":
            rgb_data.save_to_disk(str(path))
            return
        from PIL import Image

        image = Image.frombytes(
            "RGB",
            (rgb_data.width, rgb_data.height),
            rgb_data.raw_data,
            "raw",
            "BGRX",
        )
        image.save(path, quality=self.rgb_quality)
