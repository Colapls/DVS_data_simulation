#!/usr/bin/env python3
"""一条命令验证网页端的「采集 + 自动标注」链路。

网页控制台的四个按钮各自对应一条命令，这里把它们按网页的调用方式串起来：

    采集   POST /api/collect → collect_dvs.py
    重投影 POST /api/project → tools/annotate_offline.py project
    统计   POST /api/stats   → tools/annotate_offline.py stats
    质检   POST /api/check   → tools/check_dataset.py

最后用网页自己的读图函数（list_sequences / list_cameras / _load_annotations /
render_event_frame）把新序列过一遍，确认页面能出图、出框、出事件图。

    python tools/verify_pipeline.py                    # 采一次再验
    python tools/verify_pipeline.py --skip-collect     # 只验已有数据
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run(command: list[str], title: str) -> int:
    print(f"\n===== {title} =====", flush=True)
    print("$ " + " ".join(command), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return subprocess.run(command, cwd=str(PROJECT_ROOT), env=env).returncode


def _newest_sequence(dataset: Path) -> Path:
    sequences = sorted(
        (path for path in (dataset / "sequences").iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not sequences:
        raise SystemExit("数据集里一条序列都没有")
    return sequences[-1]


def _prune_index(dataset: Path) -> None:
    """把索引和位置表对齐到磁盘上真实存在的序列。

    手动删掉序列目录后，index_all.txt / locations.csv 里会留下悬空条目，
    质检会报"缺少标定文件"。这里按磁盘现状重建这两个索引：
    只登记状态为 completed 的序列。
    """
    import csv

    completed: list[str] = []
    for path in sorted((dataset / "sequences").iterdir()):
        if not path.is_dir():
            continue
        manifest_path = path / "meta" / "episode_manifest.json"
        if not manifest_path.is_file():
            continue
        status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        if status == "completed":
            completed.append(path.name)

    index_path = dataset / "index_all.txt"
    old = index_path.read_text(encoding="utf-8").splitlines() if index_path.is_file() else []
    if old != completed:
        index_path.write_text("\n".join(completed) + "\n", encoding="utf-8")
        print(f"  index_all.txt 已对齐：{old} → {completed}")

    locations_path = dataset / "locations.csv"
    if not locations_path.is_file():
        return
    with locations_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    kept = [row for row in rows if row.get("sequence_id") in completed]
    if len(kept) != len(rows):
        with locations_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        print(f"  locations.csv 已对齐：{len(rows)} 行 → {len(kept)} 行")


def _summarise(dataset: Path, sequence: Path) -> None:
    """把新序列的关键数字打出来：事件量、触发像素比例、标注数量。"""
    import h5py
    import numpy as np

    manifest = json.loads(
        (sequence / "meta" / "episode_manifest.json").read_text(encoding="utf-8")
    )
    print(f"\n===== 序列 {sequence.name} 概况 =====")
    print(
        f"  状态={manifest.get('status')} 天气={manifest.get('weather')} "
        f"帧数={manifest.get('frame_count')} 跳过={manifest.get('skipped_frames')}"
    )
    print(f"  曝光模式={manifest.get('dvs_exposure_mode_applied')}")
    print(f"  空帧统计={manifest.get('dvs_empty_frames')}")

    cameras_dir = sequence / "cameras"
    for camera_dir in sorted(cameras_dir.iterdir()):
        if not camera_dir.is_dir():
            continue
        events_path = camera_dir / "events.h5"
        if not events_path.is_file():
            continue
        with h5py.File(events_path, "r") as handle:
            width = int(handle["meta/sensor_width"][()])
            height = int(handle["meta/sensor_height"][()])
            offsets = handle["frames/event_offsets"]
            counts = handle["frames/event_count"]
            frames = len(counts)
            total = int(offsets[-1])
            print(f"\n  [{camera_dir.name}] 事件 {total:,}，{frames} 帧")
            print(
                f"    每帧平均 {total / max(frames, 1):,.0f} 个"
                f"（占全画面 {total / max(frames, 1) / (width * height):.1%}）"
            )
            # 抽几帧看"有多少像素被触发"：>50% 说明是曝光抖动而不是事件流。
            for index in (frames // 4, frames // 2, (3 * frames) // 4):
                if not 0 <= index < frames:
                    continue
                start, end = int(offsets[index]), int(offsets[index + 1])
                xs = handle["events/x"][start:end]
                ys = handle["events/y"][start:end]
                if end == start:
                    print(f"    帧 {index}: 空帧")
                    continue
                keys = ys.astype(np.int64) * width + xs
                ratio = len(np.unique(keys)) / (width * height)
                print(
                    f"    帧 {index}: {end - start:,} 个事件，"
                    f"触发像素 {ratio:.1%}"
                )

        annotations_path = camera_dir / "annotations.json"
        if annotations_path.is_file():
            data = json.loads(annotations_path.read_text(encoding="utf-8"))
            frames_data = data.get("frames", [])
            labels = sum(len(f.get("annotations", [])) for f in frames_data)
            ignored = sum(len(f.get("ignored_annotations", [])) for f in frames_data)
            print(
                f"    标注：{len(frames_data)} 帧，目标 {labels:,} 个，"
                f"忽略 {ignored:,} 个"
            )


def _check_web_readers(dataset: Path, sequence: Path) -> bool:
    """用网页自己的读图函数过一遍新序列：页面要能出图、出框、出事件图。"""
    import webui

    print("\n===== 网页读图接口自检 =====")
    ok = True
    sequence_id = sequence.name
    rows = webui.list_sequences(dataset)
    listed = [row["id"] for row in rows]
    print(f"  /api/sequences 列到 {len(listed)} 条序列，新序列在列表里："
          f"{sequence_id in listed}")
    ok = ok and sequence_id in listed

    cameras = webui.list_cameras(dataset, sequence_id)
    print(f"  /api/seq/{sequence_id}/cameras → "
          f"{[c['camera_id'] for c in cameras['cameras']]}，"
          f"snapshot={'有' if cameras['has_snapshot'] else '无'}")
    ok = ok and bool(cameras["cameras"]) and cameras["has_snapshot"]

    for camera in cameras["cameras"]:
        camera_id = camera["camera_id"]
        rgb_dir = sequence / "cameras" / camera_id / "rgb"
        frames = sorted(rgb_dir.iterdir())
        annotations_path = sequence / "cameras" / camera_id / "annotations.json"
        data = webui._load_annotations(annotations_path)
        image = webui.render_event_frame(
            sequence / "cameras" / camera_id / "events.h5", 10
        )
        print(
            f"  [{camera_id}] RGB {len(frames)} 张，标注 {len(data.get('frames', []))} 帧，"
            f"事件图 {len(image):,} 字节"
        )
        ok = ok and bool(frames) and bool(data.get("frames")) and len(image) > 0
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="网页端采集+自动标注链路验证")
    parser.add_argument("--skip-collect", action="store_true", help="不重新采集")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--rig", default="", help="机位组（默认用网页的默认值）")
    parser.add_argument("--dataset", default="dvs_roadside_multicam")
    args = parser.parse_args()

    import webui

    dataset = (PROJECT_ROOT / args.dataset).resolve()
    rig = args.rig or webui.list_rigs()[0]

    if not args.skip_collect:
        code = _run(
            [
                sys.executable,
                "collect_dvs.py",
                "--rig", rig,
                "--output", str(dataset),
                "--frames", str(args.frames),
                "--port", str(args.port),
                "--load-town",
            ],
            "采集（网页 /api/collect）",
        )
        if code != 0:
            print(f"\n采集失败（exit {code}），后面几步没跑。")
            return code

    sequence = _newest_sequence(dataset)

    print("\n===== 索引对齐 =====", flush=True)
    _prune_index(dataset)

    code_project = _run(
        [
            sys.executable,
            "tools/annotate_offline.py",
            "project",
            "--sequence", str(sequence),
        ],
        "离线重投影（网页 /api/project）",
    )
    code_stats = _run(
        [
            sys.executable,
            "tools/annotate_offline.py",
            "stats",
            "--dataset", str(dataset),
        ],
        "事件统计（网页 /api/stats）",
    )
    code_check = _run(
        [sys.executable, "tools/check_dataset.py", "--dataset", str(dataset)],
        "质检（网页 /api/check）",
    )

    _summarise(dataset, sequence)
    ok = _check_web_readers(dataset, sequence)

    print("\n===== 结论 =====")
    print(f"  重投影 exit={code_project}  统计 exit={code_stats}  "
          f"质检 exit={code_check}")
    print(f"  网页读图接口{'正常' if ok else '有问题'}；数据在 {sequence}")
    return 0 if ok and not code_check else 1


if __name__ == "__main__":
    raise SystemExit(main())
