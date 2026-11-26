#!/usr/bin/env python3
"""检测+识别评估：COCO mAP + 每类召回率（方案 §4.6 的 DoD 口径）。

    python scripts/30_eval_detection.py --ckpt runs/light_detector_v1/best.pth \
        --config configs/train/light_detector.yaml --split test

评估纪律：test 集只在实验冻结后跑一次，结果写入 reports/ 留档。
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

from dvsdet.datasets.carla_dvs import build_dataloader  # noqa: E402
from dvsdet.datasets.label_map import CLASS_NAMES, CLASS_NAMES_CN  # noqa: E402
from dvsdet.eval.detection_eval import (  # noqa: E402
    build_coco_ground_truth,
    detections_to_coco,
    evaluate_coco,
    format_report,
)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="检测评估")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
        help="COCO 评估惯例用极低阈值，靠排序算 PR；阈值高会把"
        "尚未训好的低分检测直接丢掉，mAP 虚低",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    grid_width = int(data_cfg.get("grid_width", 640))
    grid_height = int(data_cfg.get("grid_height", 360))
    in_channels = 2 * int(data_cfg.get("num_time_channels", 9))

    split_file = ROOT / "data" / "splits" / f"{args.split}.txt"
    if not split_file.is_file():
        print(f"划分文件不存在：{split_file}")
        return 1
    samples = [
        Path(line.strip())
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"{args.split}: {len(samples)} 窗")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    from dvsdet.models.detector import build_model

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model = build_model(
        {
            "in_channels": in_channels,
            "num_classes": len(CLASS_NAMES),
            "width": int(config.get("model", {}).get("width", 64)),
        }
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    loader = build_dataloader(
        samples,
        batch_size=int(config.get("train", {}).get("batch_size", 16)),
        grid_size=(grid_height, grid_width),
        training=False,
        workers=int(config.get("train", {}).get("workers", 4)),
    )
    ground_truth = build_coco_ground_truth(samples, grid_width, grid_height)
    image_ids = [image["id"] for image in ground_truth["images"]]
    detections = []
    with torch.no_grad():
        for batch in loader:
            voxel = batch["voxel"].to(device, non_blocking=True)
            for result in model.predict(
                voxel,
                score_threshold=args.score_threshold,
                image_width=grid_width,
                image_height=grid_height,
            ):
                detections.append(
                    {
                        # predict 已返回图像像素坐标（xyxy），不再缩放
                        "boxes": result["boxes"].cpu().numpy(),
                        "scores": result["scores"].cpu().numpy(),
                        "labels": result["labels"].cpu().numpy(),
                    }
                )
    metrics = evaluate_coco(
        ground_truth,
        detections_to_coco(detections, image_ids),
        iou_thresholds=tuple(config.get("eval", {}).get("iou_thresholds", (0.5, 0.75))),
    )
    print()
    print(format_report(metrics))
    print("\n每类中文对照：")
    for index, name in enumerate(CLASS_NAMES):
        ap = metrics["per_class_ap"].get(name, 0.0)
        recall = metrics["per_class_recall"].get(name, 0.0)
        print(f"  {CLASS_NAMES_CN[index]:6s} AP={ap:.4f} 召回={recall:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"eval_{args.split}.json"
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
