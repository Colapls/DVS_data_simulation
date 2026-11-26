#!/usr/bin/env python3
"""序列级推理：体素网格 → 检测框 + 三分类，落盘可视化与 JSON 结果。

    python scripts/20_infer_sequence.py --sequence sequence_000011 \
        --ckpt runs/light_detector_v1/best.pth \
        --config configs/train/light_detector.yaml

对每条序列的每个相机流逐窗推理，输出：
    runs/infer/<sequence>/<camera>/detections.json  检测结果
    runs/infer/<sequence>/<camera>/overlay_000000.png 事件帧 + 框（叠加可视化）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dvsdet.datasets.label_map import CLASS_NAMES_CN, class_name_cn  # noqa: E402
from dvsdet.models.detector import build_model  # noqa: E402
from dvsdet.slicing.slicer import SliceConfig, iter_camera_windows  # noqa: E402


def render_overlay(
    voxel: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    grid_width: int,
    grid_height: int,
) -> np.ndarray:
    """把体素画成灰度图（ON 红 / OFF 蓝）并叠加检测框，返回 RGB uint8。"""
    channels = voxel.shape[0] // 2
    on = voxel[:channels].sum(axis=0)
    off = voxel[channels:].sum(axis=0)
    peak = max(float(on.max()), float(off.max()), 1.0)
    image = np.zeros((grid_height, grid_width, 3), dtype=np.float32)
    image[..., 0] = on / peak
    image[..., 2] = off / peak
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = (int(v) for v in box)
        x1 = max(0, min(x1, grid_width - 1))
        x2 = max(0, min(x2, grid_width - 1))
        y1 = max(0, min(y1, grid_height - 1))
        y2 = max(0, min(y2, grid_height - 1))
        color = (0.0, 1.0, 0.0) if int(label) != 2 else (1.0, 0.85, 0.0)
        image[y1 : y2 + 1, x1 : x2 + 1] = color
        image[y1 : y2 + 1, x1 : x1 + 2] = color
        image[y2 - 1 : y2 + 1, x1 : x2 + 1] = color
        image[y1 : y2 + 1, x2 - 1 : x2 + 1] = color
    return (image * 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="序列级检测推理")
    parser.add_argument("--sequence", required=True, help="序列 ID 或路径")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"E:/Projects/DVS_data/dvs_roadside_multicam"),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "infer")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--save-overlay", action="store_true", default=True)
    parser.add_argument("--no-overlay", dest="save_overlay", action="store_false")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    grid_width = int(data_cfg.get("grid_width", 640))
    grid_height = int(data_cfg.get("grid_height", 360))
    in_channels = 2 * int(data_cfg.get("num_time_channels", 9))

    sequence_dir = (
        Path(args.sequence)
        if Path(args.sequence).is_absolute()
        else args.dataset / "sequences" / args.sequence
    )
    if not sequence_dir.is_dir():
        print(f"找不到序列目录：{sequence_dir}")
        return 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model = build_model(
        {
            "in_channels": in_channels,
            "num_classes": len(CLASS_NAMES_CN),
            "width": int(config.get("model", {}).get("width", 64)),
        }
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"设备：{device}；权重：{args.ckpt}（epoch {checkpoint.get('epoch')}）")

    slice_config = SliceConfig(
        window_ms=50.0,
        overlap_ms=25.0,
        num_time_channels=int(data_cfg.get("num_time_channels", 9)),
        grid_width=grid_width,
        grid_height=grid_height,
    )

    out_root = args.out / sequence_dir.name
    total_windows = 0
    total_dets = 0
    for camera_dir in sorted((sequence_dir / "cameras").glob("cam_*")):
        if not camera_dir.is_dir():
            continue
        cam_out = out_root / camera_dir.name
        cam_out.mkdir(parents=True, exist_ok=True)
        records = []
        for index, window in enumerate(
            iter_camera_windows(camera_dir, slice_config)
        ):
            voxel = torch.from_numpy(window["voxel"].astype(np.float32))
            with torch.no_grad():
                results = model.predict(
                    voxel.unsqueeze(0).to(device),
                    score_threshold=args.score_threshold,
                    image_width=grid_width,
                    image_height=grid_height,
                )
            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            records.append(
                {
                    "frame_index": int(window["frame_index"]),
                    "t_end_ns": int(window["t_end_ns"]),
                    "boxes": boxes.tolist(),
                    "scores": scores.tolist(),
                    "labels": labels.tolist(),
                    "label_names": [class_name_cn(int(v)) for v in labels],
                }
            )
            total_windows += 1
            total_dets += len(scores)
            if args.save_overlay and index < 20:
                image = render_overlay(
                    window["voxel"], boxes, scores, labels, grid_width, grid_height
                )
                try:
                    from PIL import Image

                    Image.fromarray(image).save(cam_out / f"overlay_{index:06d}.png")
                except ImportError:
                    pass
        (cam_out / "detections.json").write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"  {camera_dir.name}: {len(records)} 窗，"
            f"共 {sum(len(r['scores']) for r in records)} 个检测 → {cam_out}"
        )
    print(f"\n完成：{total_windows} 窗，{total_dets} 个检测 → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
