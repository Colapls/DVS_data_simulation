#!/usr/bin/env python3
"""事件时间戳单位（纳秒/微秒）的自检。

覆盖三件事：
1. 写入器能把 CARLA 的纳秒原始时间戳正确换算成相对纳秒时间轴；
2. 历史数据集里的错位时间轴能被识别，并被常量平移修正；
3. 质检工具能发现单位错误，修好后不再报错。

    python tools/test_event_timestamps.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from dvs_collector.config import DVS_EVENT_DTYPE  # noqa: E402
from dvs_collector.storage import CameraWriter  # noqa: E402
from tools.check_dataset import _validate_event_time_unit  # noqa: E402
from tools.fix_event_timestamps import _diagnose, fix_file  # noqa: E402

import h5py  # noqa: E402


def _raw_events(timestamps_ns: list[int]) -> bytes:
    """构造 CARLA DVS 原始事件字节（x/y/pol 无关紧要，时间戳为准）。"""
    dtype = np.dtype(DVS_EVENT_DTYPE)
    data = np.zeros(len(timestamps_ns), dtype=dtype)
    data["t"] = np.asarray(timestamps_ns, dtype="i8")
    data["x"] = np.arange(len(timestamps_ns), dtype="u2")
    data["y"] = np.arange(len(timestamps_ns), dtype="u2")
    data["pol"] = True
    return data.tobytes()


def _write_file(path: Path, origin_us: int, frame_ts_us: list[int], raw_ns: list[int]) -> None:
    """按新写入器的方式落盘一个 events.h5。"""
    writer = CameraWriter(
        path,
        h5py,
        np,
        "sequence_000001",
        "sequence_000001",
        "cam_01",
        {},
        "rgb/",
        1280,
        720,
        origin_us,
        50_000,
        1,
    )
    try:
        for index, timestamp_us in enumerate(frame_ts_us):
            writer.add_frame(
                f"f{index}", timestamp_us, _raw_events(raw_ns[index])
            )
    finally:
        writer.close()


def main() -> int:
    origin_us = 294_348_561
    origin_ns = origin_us * 1000
    # 仿真时间 294.348561 s 起，第 0 帧的曝光窗口是 [50 ms, 100 ms]。
    frame_ts_us = [100_000, 150_000]
    frame_raw_ns = [
        origin_ns + 50_000_000 + np.arange(1000, dtype="i8") * 100,
        origin_ns + 100_000_000 + np.arange(1000, dtype="i8") * 100,
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = root / "good.h5"
        _write_file(good, origin_us, frame_ts_us, frame_raw_ns)

        with h5py.File(good, "r") as handle:
            unit = handle["meta/event_timestamp_unit"][()]
            assert unit == b"ns", unit
            assert int(handle["meta/time_origin_ns"][()]) == origin_ns
            assert handle["meta/event_timestamp_source_unit"][()] == b"ns"
            stored = handle["events/t"][:]
            # 相对纳秒：第一个事件应落在第 0 帧曝光窗口内（50–100 ms）。
            assert 50_000_000 <= int(stored[0]) <= 100_000_000, int(stored[0])
            assert np.all(stored[1:] > stored[:-1]), "事件时间戳必须严格递增"
            assert int(stored[0]) == 50_000_000, int(stored[0])

        # 质检：单位正确时不应报错。
        with h5py.File(good, "r") as handle:
            errors = _validate_event_time_unit(handle, good, frame_ts_us)
        assert not errors, errors

        # 模拟历史数据：写入器错位（相对纳秒 + 999 * 起点微秒）。
        broken = root / "broken.h5"
        _write_file(broken, origin_us, frame_ts_us, frame_raw_ns)
        with h5py.File(broken, "r+") as handle:
            dataset = handle["events/t"]
            dataset[:] = (dataset[:].astype("int64") + 999 * origin_us).astype(
                "uint64"
            )
            del handle["meta/event_timestamp_unit"]
            handle["meta"].create_dataset(
                "event_timestamp_unit",
                data="us",
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        with h5py.File(broken, "r") as handle:
            errors = _validate_event_time_unit(handle, broken, frame_ts_us)
        assert errors, "单位错误必须被质检发现"

        with h5py.File(broken, "r") as handle:
            info = _diagnose(handle)
        assert info["state"] == "needs_fix", info
        assert int(info["shift"]) == -(999 * origin_us), info

        # 走真实修复路径（副本上平移后原子替换）。
        message = fix_file(broken, dry_run=False)
        assert message.startswith("已修复"), message
        with h5py.File(broken, "r") as handle:
            errors = _validate_event_time_unit(handle, broken, frame_ts_us)
        assert not errors, errors
        with h5py.File(broken, "r") as handle:
            assert int(handle["events/t"][0]) == 50_000_000
            assert handle["meta/event_timestamp_unit"][()] == b"ns"

        # 微秒版本的 CARLA 也能识别：原始时间戳给的是微秒。
        micro = root / "micro.h5"
        writer = CameraWriter(
            micro,
            h5py,
            np,
            "sequence_000001",
            "sequence_000001",
            "cam_01",
            {},
            "rgb/",
            1280,
            720,
            origin_us,
            50_000,
            1,
        )
        writer.add_frame(
            "f0",
            100_000,
            _raw_events(
                [
                    origin_us + 50_000 + index
                    for index in range(1000)
                ]
            ),
        )
        writer.close()
        with h5py.File(micro, "r") as handle:
            assert handle["meta/event_timestamp_source_unit"][()] == b"us"
            stored = handle["events/t"][:]
            assert 50_000_000 <= int(stored[0]) <= 100_000_000, int(stored[0])

    print("事件时间戳单位自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
