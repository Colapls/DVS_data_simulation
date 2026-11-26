#!/usr/bin/env python3
"""DVS 事件读取的自检（不依赖 CARLA）。

回归背景：CARLA 的 DVS 传感器不是每帧都发数据——画面完全静止、这一帧
没有任何事件时它就不发。旧代码像 RGB 那样死等某个 frame 号，画面一安静
下来就永远等不到，整条序列以"等待 cam_XX_dvs 的 CARLA frame=N 超时"退出
（实测 cam_02 停在 frame=1934）。

    python tools/test_sensors.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvs_collector.sensors import SensorFrameReader  # noqa: E402


class _FakeSensorData:
    """最小化的 carla.Image 替身（DVS 帧）。"""

    def __init__(self, frame: int, raw: bytes) -> None:
        self.frame = frame
        self.raw_data = raw


class _FakeRig:
    def __init__(self) -> None:
        self.queues = {"cam_01_dvs": queue.Queue()}
        self.future_data = {"cam_01_dvs": {}}


def test_get_dvs_merges_queued_frames() -> None:
    rig = _FakeRig()
    rig.queues["cam_01_dvs"].put(_FakeSensorData(10, b"aaa"))
    rig.queues["cam_01_dvs"].put(_FakeSensorData(11, b"bbb"))
    reader = SensorFrameReader(rig, timeout=1.0, dvs_grace_s=0.05)
    assert reader.get_dvs("cam_01_dvs", 11) == b"aaabbb"
    assert reader.last_received_frame["cam_01_dvs"] == 11


def test_get_dvs_returns_empty_when_sensor_is_silent() -> None:
    """传感器不发数据时必须立刻返回空帧，不能把采集卡死。"""
    rig = _FakeRig()
    reader = SensorFrameReader(rig, timeout=1.0, dvs_grace_s=0.05)
    began = time.monotonic()
    assert reader.get_dvs("cam_01_dvs", 12) == b""
    # 只等宽限期，不能按 sensor_timeout 死等。
    assert time.monotonic() - began < 0.5


def test_get_dvs_waits_for_late_frame() -> None:
    """数据慢一拍到达时，宽限期内要等得到它。"""
    rig = _FakeRig()
    reader = SensorFrameReader(rig, timeout=1.0, dvs_grace_s=1.0)

    def deliver() -> None:
        time.sleep(0.05)
        rig.queues["cam_01_dvs"].put(_FakeSensorData(20, b"late"))

    thread = threading.Thread(target=deliver)
    thread.start()
    assert reader.get_dvs("cam_01_dvs", 20) == b"late"
    thread.join()


def test_get_dvs_does_not_merge_future_frame() -> None:
    """DVS 跑到目标帧前面时，要留给下一轮，不能并进当前帧。"""
    rig = _FakeRig()
    rig.queues["cam_01_dvs"].put(_FakeSensorData(12, b"next"))
    reader = SensorFrameReader(rig, timeout=1.0, dvs_grace_s=0.05)
    assert reader.get_dvs("cam_01_dvs", 11) == b""
    assert 12 in rig.future_data["cam_01_dvs"]
    # 下一轮正好对上帧号，直接拿到那一帧的数据。
    assert reader.get_dvs("cam_01_dvs", 12) == b"next"


def main() -> int:
    tests = [
        test_get_dvs_merges_queued_frames,
        test_get_dvs_returns_empty_when_sensor_is_silent,
        test_get_dvs_waits_for_late_frame,
        test_get_dvs_does_not_merge_future_frame,
    ]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print("DVS 事件读取自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
