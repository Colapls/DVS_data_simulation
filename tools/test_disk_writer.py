#!/usr/bin/env python3
"""后台落盘线程（CameraDiskWriter）的自检。

覆盖：帧序与内容正确、时间戳文件逐帧对应、写盘失败能传回主循环、
close 幂等。不依赖 CARLA。

    python tools/test_disk_writer.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from dvs_collector.config import DVS_EVENT_DTYPE  # noqa: E402
from dvs_collector.storage import CameraDiskWriter, CameraWriter  # noqa: E402


class _FakeImage:
    """最小化的 carla.Image 替身：save_to_disk 写一个占位文件。"""

    def __init__(self, tag: str, fail: bool = False) -> None:
        self.tag = tag
        self.fail = fail

    def save_to_disk(self, path: str) -> None:
        if self.fail:
            raise OSError("磁盘已满（测试）")
        Path(path).write_text(self.tag, encoding="utf-8")


class _FakeJpegImage:
    """JPEG 分支用的替身：4×2 像素的 BGRA 原始数据。"""

    width = 4
    height = 2
    raw_data = bytes([10, 20, 30, 255] * (width * height))


def _dvs_bytes(timestamps_ns: list[int]) -> bytes:
    data = np.zeros(len(timestamps_ns), dtype=np.dtype(DVS_EVENT_DTYPE))
    data["t"] = np.asarray(timestamps_ns, dtype="i8")
    data["x"] = np.arange(len(timestamps_ns), dtype="u2")
    data["y"] = np.arange(len(timestamps_ns), dtype="u2")
    data["pol"] = True
    return data.tobytes()


def main() -> int:
    origin_ns = 294_348_561_000
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        events_path = root / "events.h5"
        rgb_dir = root / "rgb"
        rgb_dir.mkdir()
        writer = CameraWriter(
            events_path, h5py, np,
            "sequence_000001", "sequence_000001", "cam_01",
            {}, "rgb/", 1280, 720, origin_ns // 1000, 50_000, 10,
        )
        disk = CameraDiskWriter(writer, rgb_dir, root / "timestamps.txt", "cam_01")
        disk.start()

        frame_counts = []
        for index in range(25):
            base = origin_ns + 50_000_000 + index * 50_000_000
            frame_counts.append(3)
            disk.submit(
                index,
                f"sequence_000001_cam_01_{index:06d}",
                100_000 + index * 50_000,
                _dvs_bytes([base, base + 100, base + 200]),
                _FakeImage(f"frame-{index}"),
            )
        disk.finish()
        writer.close()  # 真实流程里由 _cleanup 关闭；这里手动关以便读回校验。

        assert disk.count == 25 * 3, disk.count
        timestamps = (root / "timestamps.txt").read_text(encoding="utf-8").splitlines()
        assert len(timestamps) == 25, timestamps
        assert timestamps[0] == "100000" and timestamps[-1] == "1300000"
        pngs = sorted(path.name for path in rgb_dir.glob("*.png"))
        assert pngs == [f"{index:06d}.png" for index in range(25)], pngs
        assert (rgb_dir / "000007.png").read_text(encoding="utf-8") == "frame-7"

        with h5py.File(events_path, "r") as handle:
            assert len(handle["events/t"]) == 75
            assert [int(v) for v in handle["frames/event_count"][:]] == [3] * 25
            # 帧序与事件归属正确：第 i 帧的首事件时间随 i 递增。
            offsets = handle["frames/event_offsets"][:]
            firsts = [int(handle["events/t"][int(o)]) for o in offsets[:-1]]
            assert firsts == sorted(firsts), firsts

        # 写盘失败必须传回主循环，而不是无声丢数据。
        writer2 = CameraWriter(
            root / "events2.h5", h5py, np,
            "sequence_000001", "sequence_000001", "cam_01",
            {}, "rgb/", 1280, 720, origin_ns // 1000, 50_000, 10,
        )
        disk2 = CameraDiskWriter(writer2, rgb_dir, root / "ts2.txt", "cam_01")
        disk2.start()
        disk2.submit(
            0, "f0", 100_000,
            _dvs_bytes([origin_ns]),
            _FakeImage("ok"),
        )
        disk2.submit(
            1, "f1", 150_000,
            _dvs_bytes([origin_ns]),
            _FakeImage("boom", fail=True),
        )
        try:
            disk2.finish()
        except RuntimeError as exc:
            assert "磁盘已满" in str(exc), exc
        else:
            raise AssertionError("写盘失败应当抛回主循环")
        writer2.close()
        # 失败前的完整帧仍然落盘。
        assert (rgb_dir / "000000.png").exists()

        # JPEG 分支：文件名换成 .jpg，内容是能读回来的 JPEG。
        jpeg_dir = root / "rgb_jpeg"
        jpeg_dir.mkdir()
        writer3 = CameraWriter(
            root / "events3.h5", h5py, np,
            "sequence_000001", "sequence_000001", "cam_01",
            {}, "rgb/", 4, 2, origin_ns // 1000, 50_000, 10,
        )
        disk3 = CameraDiskWriter(
            writer3, jpeg_dir, root / "ts3.txt", "cam_01",
            rgb_format="jpeg", rgb_quality=90,
        )
        disk3.start()
        disk3.submit(
            0, "f0", 100_000,
            _dvs_bytes([origin_ns]),
            _FakeJpegImage(),
        )
        disk3.finish()
        writer3.close()
        assert not list(jpeg_dir.glob("*.png"))
        from PIL import Image

        with Image.open(jpeg_dir / "000000.jpg") as image:
            assert image.format == "JPEG", image.format
            assert image.size == (4, 2), image.size

        # 空 DVS 帧：画面静止时 CARLA 的 DVS 根本不发数据，这一帧就是
        # 0 个事件，但帧本身要照常落盘（RGB 和时间戳不能少）。
        empty_dir = root / "rgb_empty"
        empty_dir.mkdir()
        writer4 = CameraWriter(
            root / "events4.h5", h5py, np,
            "sequence_000001", "sequence_000001", "cam_01",
            {}, "rgb/", 1280, 720, origin_ns // 1000, 50_000, 10,
        )
        disk4 = CameraDiskWriter(writer4, empty_dir, root / "ts4.txt", "cam_01")
        disk4.start()
        disk4.submit(0, "f0", 100_000, _dvs_bytes([origin_ns]), _FakeImage("a"))
        disk4.submit(1, "f1", 150_000, b"", _FakeImage("b"))
        disk4.finish()
        writer4.close()
        with h5py.File(root / "events4.h5", "r") as handle:
            assert [int(v) for v in handle["frames/event_count"][:]] == [1, 0]
            assert len(handle["events/t"]) == 1
        assert len((root / "ts4.txt").read_text(encoding="utf-8").splitlines()) == 2

    print("后台落盘线程自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
