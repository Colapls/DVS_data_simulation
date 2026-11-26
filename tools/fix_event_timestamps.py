#!/usr/bin/env python3
"""修复历史数据集中事件时间戳的单位错误。

背景：CARLA 0.9.16 的 DVS 传感器给出的原始事件时间戳是纳秒，而旧版写入器
按微秒减去序列起点，导致 events/t 比真实时间轴大约 1000 倍（事件时间与
RGB 帧时间对不上）。本工具把受影响文件就地平移到正确的纳秒时间轴：

    旧值 = 原始纳秒 - 起点微秒        新值 = 原始纳秒 - 起点纳秒
    新值 = 旧值 - 999 * 起点微秒

平移是常量偏移，不改变事件顺序，也不改变任何其它数据。

    python tools/fix_event_timestamps.py --dataset dvs_roadside_multicam
    python tools/fix_event_timestamps.py --dataset dvs_roadside_multicam --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

CHUNK = 4_000_000


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _diagnose(handle: Any) -> dict[str, Any]:
    """判断文件里的事件时间轴是否需要平移。

    两种假设：
      A 已经是正确的相对纳秒时间轴（新的写入器）；
      B 旧写入器的错位时间轴（相对纳秒再加 999 * 起点微秒）。
    对若干采样帧分别计算残差，全部匹配才算成立，否则文件本身已经损坏。
    """
    origin_us = int(handle["meta/time_origin_us"][()])
    origin_ns = origin_us * 1000
    unit = "ns"
    if "meta/event_timestamp_unit" in handle:
        unit = _text(handle["meta/event_timestamp_unit"][()])
    integration_us = 0
    if "meta/integration_time_us" in handle:
        integration_us = int(handle["meta/integration_time_us"][()])
    count = len(handle["events/t"])
    if count == 0 or "frames/timestamp_us" not in handle:
        return {"state": "empty", "origin_us": origin_us, "unit": unit}
    offsets = handle["frames/event_offsets"]
    timestamps = handle["frames/timestamp_us"]
    frame_count = len(timestamps)
    tolerance_ns = integration_us * 1000 + 5_000_000
    # 采样若干帧：事件应落在该帧曝光窗口内（最多早一个积分时间）。
    samples = sorted({0, frame_count // 4, frame_count // 2, frame_count - 1})
    residual_fixed = 0
    residual_buggy = 0
    for index in samples:
        if index + 1 >= len(offsets):
            continue
        start, end = int(offsets[index]), int(offsets[index + 1])
        if end <= start:
            continue
        event_ns = int(handle["events/t"][end - 1])
        frame_ns = int(timestamps[index]) * 1000
        residual_fixed = max(residual_fixed, abs(event_ns - frame_ns))
        residual_buggy = max(
            residual_buggy, abs(event_ns - (frame_ns + 999 * origin_us))
        )
    if residual_fixed <= tolerance_ns:
        state, shift = "ok", 0
    elif residual_buggy <= tolerance_ns:
        state, shift = "needs_fix", -(origin_ns - origin_us)
    else:
        state, shift = "unknown", 0
    return {
        "state": state,
        "shift": shift,
        "origin_us": origin_us,
        "origin_ns": origin_ns,
        "unit": unit,
        "events": count,
        "residual_fixed": residual_fixed,
        "residual_buggy": residual_buggy,
    }


def _apply_shift(handle: Any, shift: int) -> None:
    """把 events/t 就地平移常量偏移，保持严格递增。"""
    dataset = handle["events/t"]
    count = len(dataset)
    for start in range(0, count, CHUNK):
        end = min(start + CHUNK, count)
        values = dataset[start:end].astype("int64") + shift
        dataset[start:end] = values.astype("uint64")


def fix_file(path: Path, dry_run: bool) -> str:
    """检查并按需修复一个 events.h5，返回状态描述。

    修复在副本上进行，最后再原子替换原文件：中途失败不会留下半修的文件。
    """
    import h5py
    import os
    import shutil

    with h5py.File(path, "r") as handle:
        info = _diagnose(handle)
    if info["state"] != "needs_fix":
        return f"{info['state']}（{path}）"
    if dry_run:
        return (
            f"需要修复：{path}（平移 {int(info['shift'])} ns，"
            f"{info['events']} 个事件，起点 {info['origin_us']} us，"
            f"残差 正确 {info['residual_fixed']} / 旧错位 {info['residual_buggy']} ns）"
        )
    temporary = path.with_name(path.name + ".fixing")
    shutil.copy2(path, temporary)
    try:
        with h5py.File(temporary, "r+") as handle:
            _apply_shift(handle, int(info["shift"]))
            meta = handle["meta"]
            if "event_timestamp_unit" in meta:
                del meta["event_timestamp_unit"]
            meta.create_dataset(
                "event_timestamp_unit",
                data="ns",
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            if "time_origin_ns" not in meta:
                meta.create_dataset(
                    "time_origin_ns", data=int(info["origin_ns"]), dtype="u8"
                )
            if "event_timestamp_source_unit" not in meta:
                meta.create_dataset(
                    "event_timestamp_source_unit",
                    data="ns",
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return (
        f"已修复：{path}（平移 {int(info['shift'])} ns，"
        f"{info['events']} 个事件，起点 {info['origin_us']} us）"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="修复历史数据集中事件时间戳的纳秒/微秒单位错误。"
    )
    parser.add_argument("--dataset", type=Path, default=Path("dvs_roadside_multicam"))
    parser.add_argument("--dry-run", action="store_true", help="只报告，不改文件")
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    files = sorted(root.glob("sequences/sequence_*/cameras/*/events.h5"))
    if not files:
        print(f"{root} 中没有找到 events.h5")
        return 1
    changed = 0
    for path in files:
        message = fix_file(path, args.dry_run)
        print(message)
        if message.startswith("已修复"):
            changed += 1
    print(f"共 {len(files)} 个文件，{changed} 个已修复。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
