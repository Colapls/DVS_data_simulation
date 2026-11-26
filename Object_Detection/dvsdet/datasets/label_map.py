"""类别映射：采集器的 category_id ↔ 训练用的连续类别索引。

采集器 `annotations.json` 里的 `category_id`（方案 §4.2 的 "class_map_version"
记录了该表）：

    car=0, motorcycle=1, pedestrian=2

训练侧统一用 0 起连续索引；COCO 评估用 1 起 id。GEN1 风格的公开数据
（car / pedestrian 两类）映射到 [0, 2]，保住日后混训接口（方案 §4.2）。
"""

from __future__ import annotations

CLASS_NAMES = ("car", "motorcycle", "pedestrian")

#: 采集器 category_id → 训练索引
CATEGORY_ID_TO_INDEX = {0: 0, 1: 1, 2: 2}

#: 训练索引 → COCO category_id（1 起）
INDEX_TO_COCO_ID = {0: 1, 1: 2, 2: 3}

#: GEN1 风格公开数据的名称 → 训练索引
GEN1_NAME_TO_INDEX = {"car": 0, "pedestrian": 2}

#: 训练索引 → 中文名（界面与报告用）
CLASS_NAMES_CN = ("汽车", "摩托车", "行人")


def to_index(category_id: int) -> int:
    """采集器 category_id → 训练索引；未知的按 car 处理。"""
    return CATEGORY_ID_TO_INDEX.get(int(category_id), 0)


def to_coco_id(index: int) -> int:
    """训练索引 → COCO category_id。"""
    return INDEX_TO_COCO_ID.get(int(index), 1)


def class_name(index: int) -> str:
    """训练索引 → 英文类名。"""
    return CLASS_NAMES[int(index)] if 0 <= int(index) < len(CLASS_NAMES) else "unknown"


def class_name_cn(index: int) -> str:
    """训练索引 → 中文类名。"""
    return (
        CLASS_NAMES_CN[int(index)]
        if 0 <= int(index) < len(CLASS_NAMES_CN)
        else "未知"
    )
