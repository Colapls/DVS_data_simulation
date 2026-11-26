"""类别映射与数据集。

``carla_dvs`` 需要 torch，只在真正构建 dataloader 时才导入——数据层自检
与切片脚本不碰 torch，不该被拖住。
"""

from .label_map import (
    CLASS_NAMES,
    CLASS_NAMES_CN,
    class_name,
    class_name_cn,
    to_coco_id,
    to_index,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_NAMES_CN",
    "class_name",
    "class_name_cn",
    "to_coco_id",
    "to_index",
]
