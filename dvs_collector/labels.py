"""目标标注：3D 框投影、三类别映射、可见度与质量标记。

采集时（在线）用 CARLA 实例分割图算可见度；事后重投影时（离线）用快照里
保存的可见像素统计。两条路径共用下面的投影与判定代码，结果一致。

方案依据：``docs/DVS自动标注方案_精简可执行版.md``（缺陷 D1–D5、判定表 §5.3）。
"""

from __future__ import annotations

import math
from typing import Any

from .classes import CLASSES, FILL_RATIO, classify_blueprint


#: 近裁剪面：角点落到相机后方时投影无意义，直接丢弃而不是当成截断（缺陷 D5）。
NEAR_PLANE_M = 0.1

#: 判定表（方案 §5.3）用到的阈值，先命中先定级。
VISIBILITY_IGNORE = 0.05
VISIBILITY_OCCLUDED = 0.20
TINY_AREA_PX = 400

#: 可探测的最小投影短边（像素）。
#:
#: 依据 DVSense DVSync 的公开参数反推：DVS 面板 1 MP、像元 1.45 μm、
#: C-MOUNT 镜头 FOV 70°~130°，1280 px 横向铺开时角分辨率约 0.055°~0.10°/px。
#: 人眼在同样角分辨率下也看不清目标轮廓，所以"肉眼看不到"可以换算成
#: "投影短边不足 N 像素"。
#:
#: 阈值由本项目实测标定（events.h5 框内事件数 vs 投影短边，运动目标）：
#: 短边 <10 px 时仅 10%~39% 的帧有事件；10~18 px 为 19%~40%；
#: 18~24 px 升到 67%~74%；≥24 px 稳定在 75%~100%。
#: 取 16 px 作为"能看到轮廓"的下限：低于它，标注框多半对应一个
#: 相机根本没采到事件的目标，训练时是纯噪声标签。
MIN_DETECT_SHORT_SIDE_PX = 16

#: 超过这个距离（米）的目标直接不标注，与像素判据互补。
#:
#: 对应 DVSync 在其标称 FOV 下对车辆的有效观测尺度：150 m 外的车只有
#: 几个像素高，DVS 不会产生足以定位的事件簇。取 120 m 作为硬上限，
#: 避免极端远的目标进入标注。
MAX_DETECT_DISTANCE_M = 120.0

#: 3D 框的 8 个角点符号组合。
_CORNER_SIGNS = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
)


def actor_records(world: Any, min_speed: float = 0.1) -> list[dict[str, Any]]:
    """遍历一次世界里的 actor，产出可用于投影的快照记录。

    记录里保存了蓝图 ID、类别、3D 框几何、位姿矩阵和速度：这些是事后无法
    重算的信息（缺陷 D3），落进 snapshot.json 后投影可以完全离线重跑。
    """
    records: list[dict[str, Any]] = []
    for actor in world.get_actors():
        blueprint_id = actor.type_id
        info = classify_blueprint(blueprint_id)
        category = info["category"]
        if category == "unknown":
            # 信号灯、传感器等非目标命名空间：直接跳过，不进忽略名单。
            continue
        velocity = actor.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        transform = actor.get_transform()
        matrix = transform.get_matrix()
        box = actor.bounding_box
        records.append(
            {
                "actor_id": int(actor.id),
                "blueprint_id": blueprint_id,
                "category": category,
                "category_id": info["category_id"],
                "subtype": info["subtype"],
                "ignore_reason": info["ignore_reason"],
                # 4×4 世界矩阵：离线重投影直接用它，不依赖 CARLA 的旋转约定。
                "matrix": [
                    round(float(value), 5) for row in matrix for value in row
                ],
                "bbox_location": [
                    round(float(box.location.x), 5),
                    round(float(box.location.y), 5),
                    round(float(box.location.z), 5),
                ],
                "extent": [
                    round(float(box.extent.x), 5),
                    round(float(box.extent.y), 5),
                    round(float(box.extent.z), 5),
                ],
                "velocity": [
                    round(float(velocity.x), 4),
                    round(float(velocity.y), 4),
                    round(float(velocity.z), 4),
                ],
                "speed_mps": round(speed, 4),
                "stationary": bool(speed < min_speed),
            }
        )
    return records


def box_corners(record: dict[str, Any], np: Any) -> Any:
    """3D 框的 8 个世界坐标角点（与 CARLA ``get_world_vertices`` 等价）。"""
    matrix = np.asarray(record["matrix"], dtype=np.float64).reshape(4, 4)
    base = np.asarray(record["bbox_location"], dtype=np.float64)
    extent = np.asarray(record["extent"], dtype=np.float64)
    corners = base + np.asarray(_CORNER_SIGNS, dtype=np.float64) * extent
    homogeneous = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def project_corners(
    corners: Any,
    inverse_matrix: Any,
    focal: float,
    width: int,
    height: int,
    np: Any,
) -> Any:
    """把世界角点投到像素平面；任一角点越过近裁剪面就返回 None。"""
    homogeneous = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    camera_points = homogeneous @ np.asarray(inverse_matrix, dtype=np.float64).T
    depths = camera_points[:, 0]
    if bool(np.any(depths <= NEAR_PLANE_M)):
        # 跨越相机平面：2D 框会异常放大，直接丢弃（缺陷 D5）。
        return None
    xs = focal * camera_points[:, 1] / depths + width / 2.0
    ys = height / 2.0 - focal * camera_points[:, 2] / depths
    return xs, ys


def decode_instance_ids(image: Any, np: Any) -> Any:
    """解码 CARLA 0.9.16 实例分割图。

    通道顺序是 BGRA：G = actor id 低字节、B = 高字节、R = 语义标签。
    （与 CARLA 官方示例 ``PythonAPI/examples/bounding_boxes.py`` 一致。）
    """
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = raw.reshape(image.height, image.width, 4)
    return array[..., 1].astype(np.int32) + (array[..., 0].astype(np.int32) << 8)


def visible_stats(
    ids: Any,
    actor_id: int,
    bbox: list[int],
    scale_x: float,
    scale_y: float,
    np: Any,
) -> tuple[int, list[int] | None, int]:
    """在投影框内统计目标自身的可见像素、可见紧框和区域面积。

    ``ids`` 是实例分割图的 actor id 图，``scale_x/scale_y`` 是分割图相对
    标注分辨率的缩放（分割图可以低分辨率采集）。
    """
    height, width = ids.shape
    x1 = max(0, int(math.floor(bbox[0] * scale_x)))
    y1 = max(0, int(math.floor(bbox[1] * scale_y)))
    x2 = min(width, int(math.ceil((bbox[0] + bbox[2]) * scale_x)))
    y2 = min(height, int(math.ceil((bbox[1] + bbox[3]) * scale_y)))
    if x2 <= x1 or y2 <= y1:
        return 0, None, 0
    crop = ids[y1:y2, x1:x2]
    mask = crop == actor_id
    count = int(mask.sum())
    area = (x2 - x1) * (y2 - y1)
    if count == 0:
        return 0, None, area
    rows, columns = np.nonzero(mask)
    tight = [
        int(x1 + columns.min()),
        int(y1 + rows.min()),
        int(x1 + columns.max() + 1),
        int(y1 + rows.max() + 1),
    ]
    return count, tight, area


def detectability(
    bbox: list[int],
    focal_px: float,
    height_m: float,
    min_short_side_px: int = MIN_DETECT_SHORT_SIDE_PX,
    max_distance_m: float = MAX_DETECT_DISTANCE_M,
) -> tuple[bool, float]:
    """判断目标是否落在相机的有效探测范围内。

    返回 ``(是否可探测, 推算距离_米)``。判据有两条，任意一条不满足即不可探测：

    1. **投影短边像素数**：目标在画面上必须足够大，相机才能分辨它的轮廓。
       这是主判据——同样的距离，正面朝相机的车短边大、侧向的车短边小，
       只按距离过滤会误伤近距离的大目标。
    2. **推算距离**：超过 ``max_distance_m`` 的目标直接不标注，
       避免极端远的目标进入标注集。

    距离由投影框高度反推：``d ≈ focal_px × 目标实际高度 / 框高像素``。
    ``height_m`` 取目标 3D 框的竖直尺寸（汽车约 1.5 m、行人约 1.8 m），
    与目标朝向无关，所以正面和侧向的目标推算距离一致。
    """
    width_px, height_px = bbox[2], bbox[3]
    short_side = min(width_px, height_px)
    # 显式转成 Python 的 float/bool：调用方可能传入 numpy 标量，
    # numpy.float64 不能直接写进 JSON。
    distance = float(
        focal_px * height_m / height_px if height_px > 0 else float("inf")
    )
    detectable = bool(
        short_side >= min_short_side_px and distance <= max_distance_m
    )
    return detectable, distance


def project_records(
    records: list[dict[str, Any]],
    inverse_matrix: Any,
    width: int,
    height: int,
    fov: float,
    np: Any,
    ids: Any = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    fill_ratio: dict[str, float] | None = None,
    visible_by_actor: dict[int, list[int]] | None = None,
    min_short_side_px: int = MIN_DETECT_SHORT_SIDE_PX,
    max_distance_m: float = MAX_DETECT_DISTANCE_M,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[int]]]:
    """把一份 actor 记录投影成某台相机的标注。

    返回 ``(labels, ignored, visibility)``：

    - ``labels``：三类别目标的标注列表
    - ``ignored``：自行车等忽略类目标的标注列表（带 reason）
    - ``visibility``：{actor_id: [visible_px, x1, y1, x2, y2, area]}，快照要存

    落在相机有效探测范围之外的目标（太远、在画面上只有几个像素）标为
    ``ignore=true`` 并带 ``ignore_reason="beyond_detection_range"``，
    保留在文件里由加载器过滤——采集时直接删掉会丢掉真实背景分布。

    ``ids`` 是在线采集时的实例分割图；离线重投影时改传 ``visible_by_actor``
    （快照里存下来的可见像素统计），两条路径算出的可见度完全一致。
    """
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    ratios = fill_ratio or FILL_RATIO
    labels: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    visibility: dict[int, list[int]] = {}

    for record in records:
        corners = box_corners(record, np)
        projected = project_corners(
            corners, inverse_matrix, focal, width, height, np
        )
        if projected is None:
            continue
        xs, ys = projected
        raw_x1, raw_y1 = float(xs.min()), float(ys.min())
        raw_x2, raw_y2 = float(xs.max()), float(ys.max())
        # 截断只表示"框被图像边界切掉"，与角点越过近裁剪面无关（缺陷 D5）。
        truncated = (
            raw_x1 < 0 or raw_y1 < 0 or raw_x2 > width or raw_y2 > height
        )
        x1 = max(0.0, min(float(width), raw_x1))
        y1 = max(0.0, min(float(height), raw_y1))
        x2 = max(0.0, min(float(width), raw_x2))
        y2 = max(0.0, min(float(height), raw_y2))
        if x2 <= x1 or y2 <= y1:
            continue
        left = int(math.floor(x1))
        top = int(math.floor(y1))
        bbox = [
            left,
            top,
            int(math.ceil(x2)) - left,
            int(math.ceil(y2)) - top,
        ]

        # 超出有效探测范围的目标：相机采不到足够的事件，留着只是噪声标签。
        # 目标高度取 3D 框的竖直尺寸（extent 是半尺寸），与朝向无关。
        detectable, distance_m = detectability(
            bbox,
            focal,
            2.0 * abs(float(record["extent"][2])),
            min_short_side_px,
            max_distance_m,
        )

        visible_px = 0
        visible_bbox: list[int] | None = None
        area = 0
        occupancy = 0.0
        visibility_ratio: float | None = None
        if ids is not None:
            visible_px, tight, area = visible_stats(
                ids, record["actor_id"], bbox, scale_x, scale_y, np
            )
            if tight is not None:
                # 可见紧框换算回标注分辨率。
                visible_bbox = [
                    int(round(tight[0] / scale_x)),
                    int(round(tight[1] / scale_y)),
                    int(round(tight[2] / scale_x)),
                    int(round(tight[3] / scale_y)),
                ]
        elif visible_by_actor is not None:
            stored = visible_by_actor.get(record["actor_id"])
            if stored:
                visible_px = int(stored[0])
                area = int(stored[5])
                if stored[1:5] != [0, 0, 0, 0]:
                    visible_bbox = [int(value) for value in stored[1:5]]
        if ids is not None or visible_by_actor is not None:
            occupancy = visible_px / area if area else 0.0
            visibility[record["actor_id"]] = [
                visible_px,
                *(visible_bbox if visible_bbox else [0, 0, 0, 0]),
                area,
            ]
            reference = ratios.get(record["category"], 0.5)
            visibility_ratio = min(1.0, occupancy / reference) if reference else 0.0

        flags: list[str] = []
        occluded = False
        difficult = False
        ignore = False
        ignore_reason = None
        if not detectable:
            # 超出探测范围：标 ignore 而不是删掉，保留真实背景分布。
            ignore = True
            ignore_reason = "beyond_detection_range"
            flags.append("beyond_detection_range")
        if visibility_ratio is not None:
            if visibility_ratio < VISIBILITY_IGNORE:
                ignore = True
                ignore_reason = ignore_reason or "low_visibility"
                flags.append("ignored_low_visibility")
            elif visibility_ratio < VISIBILITY_OCCLUDED:
                occluded = True
                difficult = True
                flags.append("severe_occlusion")
        area_px = bbox[2] * bbox[3]
        if area_px < TINY_AREA_PX:
            difficult = True
            flags.append("tiny")

        label = {
            "id": len(labels) + len(ignored) + 1,
            "category": record["category"],
            "category_id": record["category_id"],
            "subtype": record["subtype"],
            "blueprint_id": record["blueprint_id"],
            "bbox": bbox,
            "truncated": bool(truncated),
            "occluded": bool(occluded),
            "difficult": bool(difficult),
            "ignore": bool(ignore),
            "ignore_reason": ignore_reason,
            "visibility": (
                None if visibility_ratio is None else round(visibility_ratio, 4)
            ),
            "visible_px": visible_px,
            "visible_bbox": visible_bbox,
            "occupancy": round(occupancy, 4),
            "actor_id": record["actor_id"],
            "speed_mps": record["speed_mps"],
            "stationary": record["stationary"],
            "distance_m": round(distance_m, 2),
            "quality_flags": flags,
        }
        if record["category"] == "ignore":
            label["ignore_reason"] = record["ignore_reason"]
            ignored.append(label)
        else:
            labels.append(label)
    return labels, ignored, visibility


def track_statistics(
    frames: list[dict[str, Any]],
    integration_time_us: int = 50_000,
) -> dict[str, dict[str, float]]:
    """按类别统计轨迹数量、真实运动轨迹数和平均位移。

    用于验收项"行人轨迹有真实位移"（方案 §7.1、§10）：静止堆叠的行人
    会被算成位移接近 0 的轨迹，从而在采集结束时就暴露出来。
    """
    step_s = integration_time_us / 1_000_000.0
    tracks: dict[tuple[str, int], dict[str, float]] = {}
    for frame in frames:
        for annotation in frame.get("annotations", []):
            key = (annotation.get("category", "unknown"), annotation["actor_id"])
            entry = tracks.setdefault(
                key,
                {"frames": 0, "displacement": 0.0, "max_speed": 0.0},
            )
            entry["frames"] += 1
            entry["displacement"] += annotation.get("speed_mps", 0.0) * step_s
            entry["max_speed"] = max(entry["max_speed"], annotation.get("speed_mps", 0.0))

    summary: dict[str, dict[str, float]] = {}
    for (category, _), entry in tracks.items():
        bucket = summary.setdefault(
            category,
            {"tracks": 0.0, "moving_tracks": 0.0, "static_tracks": 0.0, "displacement_sum": 0.0},
        )
        bucket["tracks"] += 1
        if entry["max_speed"] > 0.5:
            bucket["moving_tracks"] += 1
        else:
            bucket["static_tracks"] += 1
        bucket["displacement_sum"] += entry["displacement"]
    for category, bucket in summary.items():
        total = bucket["tracks"] or 1.0
        bucket["mean_displacement_m"] = round(bucket["displacement_sum"] / total, 3)
        bucket.pop("displacement_sum")
    return summary


__all__ = [
    "CLASSES",
    "MAX_DETECT_DISTANCE_M",
    "MIN_DETECT_SHORT_SIDE_PX",
    "NEAR_PLANE_M",
    "TINY_AREA_PX",
    "VISIBILITY_IGNORE",
    "VISIBILITY_OCCLUDED",
    "actor_records",
    "box_corners",
    "decode_instance_ids",
    "detectability",
    "project_corners",
    "project_records",
    "track_statistics",
    "visible_stats",
]
