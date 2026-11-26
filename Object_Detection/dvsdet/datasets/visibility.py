"""标注框的 DVS 可见性判定。

DVS 相机只在**亮度变化**时出事件：静止的目标、远到只剩几个像素的目标，
在事件流里几乎没有痕迹。实测本数据集：

- 标注框内非零体素占比中位数 0.20（正常目标）
- 但 **15.7% 的标注框里几乎���有事件**（占比 < 0.01）
- **26.5% 的框短边 < 16 px**，低于 DVS 的可见性下限

这些框作为训练标签会教模型"这里该有目标、但事件流里什么也没有"——
正是技术方案 §9 风险表里的「静止目标假阴性」。方案 §3.1 的口径是：
静止样本单独导出，**不进训练主集**。

本模块提供统一判据，训练与评估都用同一套：

- **训练**：丢弃不可见的框（模型只学看得见的目标）；
- **评估**：把不可见的框按 COCO 的 iscrowd 语义处理（既不参与匹配，
  也不算漏检）——否则指标会被"本来就不可能检测到"的目标拉低。
"""

from __future__ import annotations

import numpy as np

#: 框内非零体素占比低于此值即视为"DVS 看不见"。
DEFAULT_MIN_EVENT_RATIO = 0.005


def box_event_counts(voxel: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """每个标注框内的非零体素格数。

    voxel: (C, H, W)；boxes: (N, 4) **xywh** 像素坐标（切片 npz 的格式）。返回 (N,)。
    """
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    active = voxel.sum(axis=0) > 0
    height, width = active.shape
    counts = np.zeros((len(boxes),), dtype=np.int64)
    for index, box in enumerate(boxes):
        x1 = max(int(box[0]), 0)
        y1 = max(int(box[1]), 0)
        x2 = min(int(box[0] + box[2]), width)
        y2 = min(int(box[1] + box[3]), height)
        if x2 <= x1 or y2 <= y1:
            continue
        counts[index] = int(active[y1:y2, x1:x2].sum())
    return counts


def visible_mask(
    counts: np.ndarray,
    boxes: np.ndarray,
    min_event_ratio: float = DEFAULT_MIN_EVENT_RATIO,
) -> np.ndarray:
    """哪些框在 DVS 上看得见（True=可见）。

    判据：框内非零体素格数 ≥ max(1, 框面积 × min_event_ratio)。
    """
    if len(boxes) == 0:
        return np.zeros((0,), dtype=bool)
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1.0)
    threshold = np.maximum(1.0, area * min_event_ratio)
    return counts >= threshold
