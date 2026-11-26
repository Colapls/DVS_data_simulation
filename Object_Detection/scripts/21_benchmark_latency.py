#!/usr/bin/env python3
"""端到端延迟基准：体素化 + 模型前向（方案 §4.6 DoD：batch=1 < 15 ms）。

    python scripts/21_benchmark_latency.py --ckpt runs/light_detector_v1/best.pth \
        --config configs/train/light_detector.yaml

分三段计时：
    体素化     事件流 → 18 通道体素网格（不含 HDF5 读盘）
    前向       体素 → 检测框（解码 + NMS 都在里面）
    端到端     上面两段之和
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dvsdet.models.detector import build_model  # noqa: E402
from dvsdet.slicing.slicer import SliceConfig, iter_camera_windows  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="延迟基准")
    parser.add_argument("--ckpt", type=Path, help="权重（不给则只测体素化）")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"E:/Projects/DVS_data/dvs_roadside_multicam"),
    )
    parser.add_argument("--sequence", default="sequence_000011")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    grid_width = int(data_cfg.get("grid_width", 640))
    grid_height = int(data_cfg.get("grid_height", 360))
    in_channels = 2 * int(data_cfg.get("num_time_channels", 9))
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    slice_config = SliceConfig(
        window_ms=50.0,
        overlap_ms=25.0,
        num_time_channels=int(data_cfg.get("num_time_channels", 9)),
        grid_width=grid_width,
        grid_height=grid_height,
    )
    camera_dir = (
        args.dataset / "sequences" / args.sequence / "cameras" / "cam_01"
    )
    if not camera_dir.is_dir():
        print(f"找不到相机目录：{camera_dir}")
        return 1

    # 先读一批窗出来（读盘不计时），只测计算
    windows = []
    voxel_times = []
    for window in iter_camera_windows(camera_dir, slice_config):
        windows.append(np.asarray(window["voxel"], dtype=np.float32))
        voxel_times.append(float(window["voxelize_ms"]))
        if len(windows) >= args.repeat:
            break
    if not windows:
        print("没有可用窗口")
        return 1
    print(f"窗口数 {len(windows)}，体素形状 {windows[0].shape}，设备 {device}")

    # ---- 体素化耗时（voxelize 本身，读盘与拷贝不计时）----
    voxel_ms = float(np.median(voxel_times))
    print(f"体素化（不含读盘）：中位数 {voxel_ms:.2f} ms")

    if args.ckpt is None:
        print("\n（没给权重，只测体素化）")
        return 0

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model = build_model(
        {
            "in_channels": in_channels,
            "num_classes": 3,
            "width": int(config.get("model", {}).get("width", 64)),
        }
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    tensors = [
        torch.from_numpy(window).unsqueeze(0).to(device) for window in windows
    ]

    def run(index: int) -> None:
        with torch.no_grad():
            model.predict(
                tensors[index % len(tensors)],
                score_threshold=0.3,
                image_width=grid_width,
                image_height=grid_height,
            )

    for index in range(args.warmup):
        run(index)
    _sync(device)

    forward_times = []
    for index in range(len(tensors)):
        start = time.perf_counter()
        run(index)
        _sync(device)
        forward_times.append((time.perf_counter() - start) * 1000)
    forward_ms = float(np.median(forward_times))

    total_ms = voxel_ms + forward_ms
    print(f"前向（含解码 + NMS）：中位数 {forward_ms:.2f} ms")
    print(f"端到端（体素化 + 前向）：中位数 {total_ms:.2f} ms")
    print(f"DoD（batch=1 < 15 ms）：{'✓ 通过' if total_ms < 15 else '✗ 未达标'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
