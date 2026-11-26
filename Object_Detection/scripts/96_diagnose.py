#!/usr/bin/env python3
"""诊断脚本：看模型到底预测了什么（真值框 vs 预测框、分数分布、IoU）。

    python scripts/96_diagnose.py --ckpt runs/light_detector_v1/best.pth \
        --config configs/train/light_detector.yaml --split val --samples 8

排查"训练 loss 在降但 mAP≈0"时先跑这个：它直接打印每个样本的真值框、
预测框（分数/类别）、最佳 IoU，以及预测框的尺寸分布。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dvsdet.datasets.carla_dvs import CarlaDVSDataset, _sparse_to_dense  # noqa: E402
from dvsdet.datasets.label_map import CLASS_NAMES  # noqa: E402
from dvsdet.models.detector import build_model  # noqa: E402


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="预测结果诊断")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.05)
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

    split_file = ROOT / "data" / "splits" / f"{args.split}.txt"
    samples = [
        Path(line.strip())
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.samples]

    dataset = CarlaDVSDataset(samples, grid_size=(grid_height, grid_width))
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
    print(f"权重 epoch {checkpoint.get('epoch')}，设备 {device}")

    for index in range(len(samples)):
        sample = dataset[index]
        dense = _sparse_to_dense(sample["sparse"])
        voxel = torch.from_numpy(dense)
        with torch.no_grad():
            result = model.predict(
                voxel.unsqueeze(0).to(device),
                score_threshold=args.score_threshold,
                image_width=grid_width,
                image_height=grid_height,
            )[0]
        # 真值：cxcywh 归一化 → xyxy 像素
        gt = sample["boxes"].numpy()
        gt_xyxy = np.stack(
            [
                (gt[:, 0] - gt[:, 2] / 2) * grid_width,
                (gt[:, 1] - gt[:, 3] / 2) * grid_height,
                (gt[:, 0] + gt[:, 2] / 2) * grid_width,
                (gt[:, 1] + gt[:, 3] / 2) * grid_height,
            ],
            axis=1,
        ) if len(gt) else np.zeros((0, 4), np.float32)
        labels = sample["labels"].numpy()
        pred = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        pred_labels = result["labels"].cpu().numpy()

        print(f"\n--- 样本 {index} {samples[index].name} ---")
        print(f"真值 {len(gt_xyxy)} 个：")
        for box, label in zip(gt_xyxy, labels):
            print(
                f"   {CLASS_NAMES[int(label)]:11s} "
                f"({box[0]:6.1f},{box[1]:6.1f})-({box[2]:6.1f},{box[3]:6.1f}) "
                f"尺寸 {box[2]-box[0]:5.1f}×{box[3]-box[1]:5.1f}"
            )
        print(f"预测 {len(pred)} 个（阈值 {args.score_threshold}）：")
        order = np.argsort(-scores)[:6]
        for i in order:
            box = pred[i]
            print(
                f"   {CLASS_NAMES[int(pred_labels[i])]:11s} score={scores[i]:.3f} "
                f"({box[0]:6.1f},{box[1]:6.1f})-({box[2]:6.1f},{box[3]:6.1f}) "
                f"尺寸 {box[2]-box[0]:5.1f}×{box[3]-box[1]:5.1f}"
            )
        if len(gt_xyxy) and len(pred):
            matrix = iou_matrix(gt_xyxy, pred)
            best = matrix.max(axis=1)
            print(f"每个真值的最佳 IoU：{np.round(best, 3).tolist()}")
            print(f"最高分预测：{scores.max():.3f}")
    return 0


def _dense(sparse):
    return _sparse_to_dense(sparse)


if __name__ == "__main__":
    raise SystemExit(main())
