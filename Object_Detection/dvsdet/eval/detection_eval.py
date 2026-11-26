"""检测评估：COCO mAP（pycocotools）+ 每类召回率。

对应方案 §4.6 的验收口径：

- val mAP@0.5 达到留档线（首训合理预期 ≥ 0.55）；
- 输出每类召回率表，pedestrian 与 car 差 > 15 点须分析并记录；
- motorcycle AP 非零且 ≥ car AP 的 40%。

COCO 评估里，被标为 ignore 的框按 COCO 的 iscrowd 语义处理（既不参与
匹配，也不算漏检）——采集器的 `ignored_annotations`（超距/太小/遮挡到
不可判）就属于这一类。
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..datasets.carla_dvs import _load_voxel
from ..datasets.label_map import CLASS_NAMES, to_coco_id
from ..datasets.visibility import (
    DEFAULT_MIN_EVENT_RATIO,
    box_event_counts,
    visible_mask,
)


def build_coco_ground_truth(
    samples: Iterable[Path],
    grid_width: int,
    grid_height: int,
    image_prefix: str = "",
    min_event_ratio: float | None = DEFAULT_MIN_EVENT_RATIO,
) -> dict[str, Any]:
    """把切片样本的标注汇总成 COCO json（方案 §3.3）。

    每条序列的每个相机流算一个"图片"，这样同一帧的多相机不会互相遮挡。

    **切片 npz 里的框是 xywh 像素**（采集器 annotations.json 的 bbox
    格式），COCO 的 bbox 也是 [x, y, w, h]，直接对上。

    ``min_event_ratio`` 不为 None 时，把"DVS 看不见"的框（框内几乎没有
    事件）按 COCO 的 iscrowd 语义处理——与训练侧剔除空框的口径一致
    （见 dvsdet.datasets.visibility）。
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for sample_path in samples:
        # 用 with 及时关闭文件句柄：Windows 上不关的话临时目录删不掉
        with np.load(sample_path, allow_pickle=True) as data:
            sequence_id = str(data["sequence_id"])
            camera_id = str(data["camera_id"])
            frame_index = int(data["frame_index"])
            image_id = len(images) + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{sequence_id}/{camera_id}/{frame_index:06d}",
                    "width": grid_width,
                    "height": grid_height,
                }
            )
            boxes = data["boxes"].astype(np.float32)
            labels = data["labels"].astype(np.int64)
            ignore_boxes = data["ignore_boxes"].astype(np.float32)
            visible = np.ones((len(boxes),), dtype=bool)
            # 没有体素字段的样本（合成的测试数据）不做可见性判定，
            # 全部视为可见。
            has_voxel = "voxel_indices" in data or "voxel" in data
            if min_event_ratio is not None and has_voxel and len(boxes):
                voxel = _load_voxel(data)
                counts = box_event_counts(voxel, boxes)
                visible = visible_mask(counts, boxes, min_event_ratio)
        for box, label, is_visible in zip(boxes, labels, visible):
            # 切片里存的是 **xywh** 像素（采集器 annotations.json 的 bbox
            # 就是 [x, y, w, h]），COCO 正好也要 [x, y, w, h]。
            # 此前这里按 xyxy 拆（w = x2 - x1），负宽高会让所有框都匹配不上，
            # mAP 恒为 0。
            x, y, w, h = (float(v) for v in box)
            w, h = max(w, 0.0), max(h, 0.0)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": to_coco_id(int(label)),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    # DVS 看不见的框：按 COCO 的 iscrowd 语义（不匹配、不算漏检）
                    "iscrowd": 0 if is_visible else 1,
                }
            )
            ann_id += 1
        # ignore 框（采集器标注的超距/不可判目标）：同样按 iscrowd 处理
        for box in ignore_boxes:
            x, y, w, h = (float(v) for v in box)
            w, h = max(w, 0.0), max(h, 0.0)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": to_coco_id(0),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 1,
                }
            )
            ann_id += 1
    return {
        "info": {"dataset": "carla_dvs_3cls", "version": "v1.1"},
        "categories": [
            {"id": to_coco_id(index), "name": name}
            for index, name in enumerate(CLASS_NAMES)
        ],
        "images": images,
        "annotations": annotations,
    }


def detections_to_coco(
    detections: list[dict[str, Any]],
    image_ids: list[int],
) -> list[dict[str, Any]]:
    """把模型输出转成 COCO detection 格式。

    detections 的每项：{"image_id": int, "boxes": (N,4) xyxy 像素,
    "scores": (N,), "labels": (N,)}
    """
    results: list[dict[str, Any]] = []
    for det, image_id in zip(detections, image_ids):
        boxes = np.asarray(det["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(det["scores"], dtype=np.float32).reshape(-1)
        labels = np.asarray(det["labels"], dtype=np.int64).reshape(-1)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (float(v) for v in box)
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": to_coco_id(int(label)),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return results


def evaluate_coco(
    ground_truth: dict[str, Any],
    detections: list[dict[str, Any]],
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
) -> dict[str, Any]:
    """跑 COCO 评估，返回 mAP 与每类 AP/召回率。"""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "评估需要 pycocotools：pip install pycocotools"
        ) from exc

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        if not detections:
            # 一个检测都没有时也要给出完整的三类结构（否则报告里每类
            # 召回是空字典，看不出"0 召回"这个结论）
            return {
                "mAP": {f"@{t}": 0.0 for t in iou_thresholds},
                "per_class_ap": {name: 0.0 for name in CLASS_NAMES},
                "per_class_recall": {name: 0.0 for name in CLASS_NAMES},
            }
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.iouThrs = np.asarray(list(iou_thresholds))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    # stats 布局随 iouThrs 长度变化，这里按实际阈值重新解析
    precision = evaluator.eval["precision"]  # (T, R, K, A, M)
    recall = evaluator.eval["recall"]  # (T, K, A, M)

    def _ap_at(iou_index: int) -> float:
        """指定 IoU 阈值下、全部类别的平均 AP。"""
        slice_ = precision[iou_index, :, :, 0, 2]  # (R, K)
        valid = slice_ > -1
        return float(np.mean(slice_[valid])) if valid.any() else 0.0

    per_class_ap: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for k, name in enumerate(CLASS_NAMES):
        # 每类 AP 在 [0.5, 0.75] 两个阈值上平均（COCO 口径的子集）
        mask = precision[:, :, k, 0, 2] > -1
        per_class_ap[name] = (
            float(np.mean(precision[:, :, k, 0, 2][mask])) if mask.any() else 0.0
        )
        rmask = recall[:, k, 0, 2] > -1
        per_class_recall[name] = (
            float(np.mean(recall[:, k, 0, 2][rmask])) if rmask.any() else 0.0
        )
    return {
        "mAP": {
            f"@{threshold}": _ap_at(index)
            for index, threshold in enumerate(iou_thresholds)
        },
        "per_class_ap": per_class_ap,
        "per_class_recall": per_class_recall,
        "stats": [float(value) for value in stats],
    }


def format_report(metrics: dict[str, Any]) -> str:
    """把评估结果排成可读表格（写进实验记录表与报告）。"""
    lines = ["mAP:"]
    for key, value in metrics.get("mAP", {}).items():
        lines.append(f"  {key}: {value:.4f}")
    lines.append("每类 AP:")
    for name, value in metrics.get("per_class_ap", {}).items():
        lines.append(f"  {name:12s}: {value:.4f}")
    lines.append("每类召回率:")
    for name, value in metrics.get("per_class_recall", {}).items():
        lines.append(f"  {name:12s}: {value:.4f}")
    gap = None
    ap = metrics.get("per_class_ap", {})
    recall = metrics.get("per_class_recall", {})
    if "car" in recall and "pedestrian" in recall:
        gap = recall["car"] - recall["pedestrian"]
        lines.append(f"car − pedestrian 召回差: {gap:.4f}（> 0.15 需分析并记录）")
    if "car" in ap and "motorcycle" in ap and ap["car"] > 0:
        ratio = ap["motorcycle"] / ap["car"]
        lines.append(f"motorcycle / car AP: {ratio:.2f}（< 0.40 触发专项补采）")
    return "\n".join(lines)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
