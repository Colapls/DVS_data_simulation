#!/usr/bin/env python3
"""链路自检：模型前向 → 损失 → 推理 → 评估 → 增强对齐。

不依赖数据集，用合成数据跑通全链路并验证关键不变量：

1. 前向输出的形状与尺度（三个尺度的特征图、分类/回归/centerness）；
2. 损失有限且可反传；
3. predict 的框格式（xyxy、像素范围）与 NMS 生效；
4. **增强后标注框与体素仍然对齐**（hflip/translate/scale 是最容易出错的地方：
   框平移了体素没平移，训练就会学到错位的标签）；
5. COCO 评估能跑通并给出 mAP 与每类召回。

    python scripts/99_selftest.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dvsdet.augment.event_augment import EventAugment  # noqa: E402
from dvsdet.datasets.carla_dvs import collate  # noqa: E402
from dvsdet.eval.detection_eval import (  # noqa: E402
    build_coco_ground_truth,
    detections_to_coco,
    evaluate_coco,
)
from dvsdet.models.criterion import DetectionCriterion  # noqa: E402
from dvsdet.models.detector import build_model  # noqa: E402

GRID = (360, 640)


def _fake_batch(batch_size: int = 4, seed: int = 0):
    """合成一批带"目标"的体素样本：框位置堆高事件计数。"""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(batch_size):
        voxel = rng.random((18, *GRID)).astype(np.float32) * 3
        image_boxes = []
        image_labels = []
        for _ in range(5):
            cx = rng.uniform(0.2, 0.8)
            cy = rng.uniform(0.2, 0.8)
            w = rng.uniform(0.05, 0.2)
            h = rng.uniform(0.05, 0.2)
            x1 = int((cx - w / 2) * GRID[1])
            y1 = int((cy - h / 2) * GRID[0])
            x2 = int((cx + w / 2) * GRID[1])
            y2 = int((cy + h / 2) * GRID[0])
            voxel[:, y1:y2, x1:x2] += 20.0
            image_boxes.append([cx, cy, w, h])
            image_labels.append(int(rng.integers(0, 3)))
        samples.append(
            {
                "voxel": torch.from_numpy(voxel),
                "boxes": torch.from_numpy(
                    np.asarray(image_boxes, np.float32)
                ),
                "labels": torch.from_numpy(np.asarray(image_labels, np.int64)),
                "track_ids": torch.arange(len(image_labels)),
            }
        )
    return collate(samples)


def test_forward_and_loss() -> None:
    model = build_model({"in_channels": 18, "num_classes": 3, "width": 64})
    batch = _fake_batch()
    outputs = model(batch["voxel"])
    assert len(outputs["cls_logits"]) == 3, "应有 3 个尺度"
    for level, stride in enumerate((8, 16, 32)):
        logits = outputs["cls_logits"][level]
        # stride-2 卷积用 ceil 模式：360/16=22.5 → 23
        expected = (
            math.ceil(GRID[0] / stride),
            math.ceil(GRID[1] / stride),
        )
        assert logits.shape[1] == 3, f"分类通道应为 3：{logits.shape}"
        assert logits.shape[-2:] == expected, (
            f"尺度 {level}（stride {stride}）尺寸应为 {expected}，实际 {logits.shape[-2:]}"
        )
        assert outputs["reg_preds"][level].shape[1] == 4
        assert outputs["ctr_logits"][level].shape[1] == 1
    criterion = DetectionCriterion(num_classes=3)
    losses = criterion(outputs, batch["boxes"], batch["labels"])
    total = losses["loss"]
    assert torch.isfinite(total), f"损失不是有限值：{total}"
    total.backward()
    grads = [
        p.grad for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert grads, "没有参数拿到梯度"
    print(
        f"OK  前向+损失：loss={float(total):.4f} "
        f"(cls={float(losses['loss_cls']):.3f} reg={float(losses['loss_reg']):.3f} "
        f"ctr={float(losses['loss_ctr']):.3f})，{len(grads)} 个张量拿到梯度"
    )


def test_predict_format() -> None:
    model = build_model({"in_channels": 18, "num_classes": 3, "width": 64})
    batch = _fake_batch(batch_size=2)
    results = model.predict(batch["voxel"], score_threshold=0.0, topk=50)
    assert len(results) == 2
    for result in results:
        boxes = result["boxes"]
        assert boxes.shape[1] == 4
        assert (boxes[:, 2] >= boxes[:, 0]).all(), "框的 x2 < x1"
        assert (boxes[:, 3] >= boxes[:, 1]).all(), "框的 y2 < y1"
        assert (boxes[:, :2] >= 0).all() and (boxes[:, 2:] <= 640.01).all(), "框超出图像"
        assert result["scores"].shape[0] == boxes.shape[0]
        if result["labels"].numel():
            assert int(result["labels"].max()) < 3
    print(f"OK  predict：每图 {results[0]['boxes'].shape[0]} 个框，格式与范围正确")


def test_augment_keeps_alignment() -> None:
    """增强后框必须跟着体素一起变换，否则学到错位标签。"""
    rng = np.random.default_rng(3)
    voxel = np.zeros((18, *GRID), np.float32)
    boxes = np.asarray([[0.2, 0.3, 0.1, 0.1]], np.float32)
    # 在框对应的像素位置放高计数
    x1 = int((0.2 - 0.05) * GRID[1])
    x2 = int((0.2 + 0.05) * GRID[1])
    y1 = int((0.3 - 0.05) * GRID[0])
    y2 = int((0.3 + 0.05) * GRID[0])
    voxel[:, y1:y2, x1:x2] = 10.0
    sample = {
        "voxel": torch.from_numpy(voxel),
        "boxes": torch.from_numpy(boxes),
        "labels": torch.tensor([0]),
        "track_ids": torch.tensor([1]),
        "contrast_threshold": 0.25,
    }
    augment = EventAugment(
        temporal_jitter_ratio=0.0,
        event_drop_ratio=(0.0, 0.0),
        noise_scale=0.0,
        geometry=("hflip", "translate", "scale"),
        prob=1.0,
        seed=11,
    )
    out = augment(dict(sample))
    new_voxel = out["voxel"].numpy()
    new_boxes = out["boxes"].numpy()
    assert new_voxel.shape[0] == 18
    # 事件峰值位置应落在新框内
    peak = np.unravel_index(np.argmax(new_voxel.sum(axis=0)), new_voxel.shape[-2:])
    bx1 = (new_boxes[0, 0] - new_boxes[0, 2] / 2) * new_voxel.shape[-1]
    bx2 = (new_boxes[0, 0] + new_boxes[0, 2] / 2) * new_voxel.shape[-1]
    by1 = (new_boxes[0, 1] - new_boxes[0, 3] / 2) * new_voxel.shape[-2]
    by2 = (new_boxes[0, 1] + new_boxes[0, 3] / 2) * new_voxel.shape[-2]
    inside = bx1 - 8 <= peak[1] <= bx2 + 8 and by1 - 8 <= peak[0] <= by2 + 8
    assert inside, (
        f"增强后事件峰值 {peak} 落在新框 "
        f"({bx1:.0f},{by1:.0f})-({bx2:.0f},{by2:.0f}) 之外——框与体素错位"
    )
    print("OK  增强对齐：几何变换后事件峰值仍在标注框内")


def test_coco_eval() -> None:
    """用合成样本跑通 COCO 评估。"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(5)
        paths = []
        for index in range(4):
            path = Path(tmp) / f"{index:06d}.npz"
            # 与切片一致：**xywh** 像素坐标
            boxes = np.asarray(
                [[100.0, 80.0, 60.0, 80.0], [380.0, 150.0, 120.0, 50.0]],
                np.float32,
            )
            np.savez(
                path,
                boxes=boxes,
                labels=np.asarray([0, 2], np.int64),
                ignore_boxes=np.zeros((0, 4), np.float32),
                sequence_id="seq_test",
                camera_id="cam_01",
                frame_index=index,
                map_name="Town03",
                weather="ClearNoon",
            )
            paths.append(path)
        ground_truth = build_coco_ground_truth(paths, GRID[1], GRID[0])
        assert len(ground_truth["images"]) == 4
        assert len(ground_truth["annotations"]) == 8
        # 构造"完美"检测：真值 xywh → 模型输出的 xyxy
        detections = []
        for image in ground_truth["images"]:
            with np.load(paths[image["id"] - 1], allow_pickle=True) as sample:
                boxes = sample["boxes"]
                labels = sample["labels"]
            xyxy = np.stack(
                [
                    boxes[:, 0],
                    boxes[:, 1],
                    boxes[:, 0] + boxes[:, 2],
                    boxes[:, 1] + boxes[:, 3],
                ],
                axis=1,
            )
            detections.append(
                {
                    "boxes": xyxy,
                    "scores": np.ones(len(boxes), np.float32),
                    "labels": labels,
                }
            )
        image_ids = [image["id"] for image in ground_truth["images"]]
        metrics = evaluate_coco(
            ground_truth, detections_to_coco(detections, image_ids)
        )
        assert metrics["mAP"]["@0.5"] > 0.99, f"完美检测的 mAP 应接近 1：{metrics}"
        assert set(metrics["per_class_ap"]) == {"car", "motorcycle", "pedestrian"}
    print(f"OK  COCO 评估：完美检测 mAP@0.5={metrics['mAP']['@0.5']:.3f}")


def test_box_format_contract() -> None:
    """切片 npz 的框是 **xywh**：dataloader 与 COCO 评估必须同一口径。

    这一条曾经把 mAP 拉到 0：评估侧按 xyxy 拆框，负宽高让所有框都
    匹配不上。改任一侧的框格式都会在这里炸。
    """
    import tempfile

    from dvsdet.datasets.carla_dvs import CarlaDVSDataset

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "000000.npz"
        xywh = np.asarray([[100.0, 80.0, 60.0, 80.0]], np.float32)
        np.savez(
            path,
            boxes=xywh,
            labels=np.asarray([0], np.int64),
            track_ids=np.asarray([1], np.int64),
            ignore_boxes=np.zeros((0, 4), np.float32),
            sequence_id="seq_test",
            camera_id="cam_01",
            frame_index=0,
            map_name="Town03",
            weather="ClearNoon",
        )
        sample = CarlaDVSDataset([path], grid_size=GRID)[0]
        cx, cy, w, h = sample["boxes"].numpy()[0]
        assert abs(cx - 130.0 / GRID[1]) < 1e-5, f"cx 不对：{cx}"
        assert abs(cy - 120.0 / GRID[0]) < 1e-5, f"cy 不对：{cy}"
        assert abs(w - 60.0 / GRID[1]) < 1e-5, f"w 不对：{w}"
        assert abs(h - 80.0 / GRID[0]) < 1e-5, f"h 不对：{h}"
    print("OK  框格式契约：切片 npz 是 xywh，dataloader 与评估口径一致")


def main() -> int:
    torch.manual_seed(0)
    test_forward_and_loss()
    test_predict_format()
    test_augment_keeps_alignment()
    test_coco_eval()
    test_box_format_contract()
    print("\n链路自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
