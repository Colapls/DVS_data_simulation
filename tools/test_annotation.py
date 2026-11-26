#!/usr/bin/env python3
"""标注逻辑自检（不需要 CARLA 服务端）。

覆盖三类别映射、3D 框投影、截断/近裁剪面判定、可见度与质量标记、
离线重投影与在线结果的一致性，以及事件统计。

运行：``python tools/test_annotation.py``
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dvs_collector.classes import (  # noqa: E402
    CLASS_MAP_VERSION,
    CLASSES,
    classify_blueprint,
)
from dvs_collector.labels import (  # noqa: E402
    TINY_AREA_PX,
    detectability,
    project_records,
    track_statistics,
)
from tools.annotate_offline import annotate_camera  # noqa: E402


WIDTH, HEIGHT, FOV = 640, 360, 90.0
FOCAL = WIDTH / (2.0 * np.tan(np.radians(FOV) / 2.0))


def _actor(
    actor_id: int = 7,
    blueprint_id: str = "vehicle.audi.a2",
    location: tuple[float, float, float] = (10.0, 0.0, 0.0),
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """构造一条快照记录：相机在原点朝 +X 看。"""
    matrix = [
        1.0, 0.0, 0.0, location[0],
        0.0, 1.0, 0.0, location[1],
        0.0, 0.0, 1.0, location[2],
        0.0, 0.0, 0.0, 1.0,
    ]
    return {
        "actor_id": actor_id,
        "blueprint_id": blueprint_id,
        "category": classify_blueprint(blueprint_id)["category"],
        "category_id": classify_blueprint(blueprint_id)["category_id"],
        "subtype": classify_blueprint(blueprint_id)["subtype"],
        "ignore_reason": None,
        "matrix": matrix,
        "bbox_location": [0.0, 0.0, 0.0],
        "extent": list(extent),
        "velocity": [0.0, 0.0, 0.0],
        "speed_mps": 0.0,
        "stationary": True,
    }


def test_class_map() -> None:
    assert classify_blueprint("vehicle.kawasaki.ninja")["category"] == "motorcycle"
    assert classify_blueprint("vehicle.vespa.zx125")["category"] == "motorcycle"
    assert classify_blueprint("walker.pedestrian.0001")["category"] == "pedestrian"
    assert classify_blueprint("vehicle.bh.crossbike")["category"] == "ignore"
    assert classify_blueprint("vehicle.tesla.model3")["category"] == "car"
    assert CLASSES == {"car": 0, "motorcycle": 1, "pedestrian": 2}
    # 兜底规则必须被标出来，不允许静默归类。
    assert classify_blueprint("vehicle.tesla.model3")["by_fallback"] is True
    assert classify_blueprint("vehicle.kawasaki.ninja")["by_fallback"] is False


def test_projection() -> None:
    identity = np.eye(4)
    labels, ignored, _ = project_records(
        [_actor()], identity, WIDTH, HEIGHT, FOV, np
    )
    assert not ignored and len(labels) == 1
    label = labels[0]
    x, y, width, height = label["bbox"]
    assert width > 0 and height > 0
    # 目标中心 (10,0,0) 投影到画面正中。
    assert abs((x + width / 2) - WIDTH / 2) <= 1.5
    assert abs((y + height / 2) - HEIGHT / 2) <= 1.5
    assert label["truncated"] is False
    assert label["category"] == "car"


def test_near_plane_is_dropped() -> None:
    identity = np.eye(4)
    # 目标在相机背后：不能当成截断，必须整条丢弃。
    labels, _, _ = project_records(
        [_actor(location=(-5.0, 0.0, 0.0))], identity, WIDTH, HEIGHT, FOV, np
    )
    assert labels == []


def test_truncated_only_when_clipped() -> None:
    identity = np.eye(4)
    labels, _, _ = project_records(
        [_actor(location=(1.0, 0.0, 0.0), extent=(0.4, 2.0, 2.0))],
        identity,
        WIDTH,
        HEIGHT,
        FOV,
        np,
    )
    assert len(labels) == 1
    label = labels[0]
    assert label["truncated"] is True
    x, y, width, height = label["bbox"]
    assert x >= 0 and y >= 0 and x + width <= WIDTH and y + height <= HEIGHT


def test_visibility_rules() -> None:
    identity = np.eye(4)
    record = _actor(location=(10.0, 0.0, 0.0), extent=(0.6, 0.6, 0.6))
    labels, _, visibility = project_records(
        [record], identity, WIDTH, HEIGHT, FOV, np, ids=None
    )
    assert labels[0]["visibility"] is None

    # 造一张实例分割图：目标只露出画面中央的一小块。
    ids = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    ids[160:200, 300:340] = record["actor_id"]
    online_labels, _, visibility = project_records(
        [record], identity, WIDTH, HEIGHT, FOV, np, ids=ids
    )
    label = online_labels[0]
    assert label["visible_px"] == 40 * 40
    assert label["visible_bbox"] == [300, 160, 340, 200]
    # 轮廓占满 3D 框：可见度按填充率归一化后为 1。
    assert label["visibility"] == 1.0
    assert label["occluded"] is False

    # 只露出很小一块：可见度低于 0.20，判为严重遮挡。
    partial = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    partial[170:180, 310:320] = record["actor_id"]
    labels, _, _ = project_records(
        [record], identity, WIDTH, HEIGHT, FOV, np, ids=partial
    )
    label = labels[0]
    assert label["visible_px"] == 100
    assert label["visible_bbox"] == [310, 170, 320, 180]
    assert 0.0 < label["visibility"] < 0.2
    assert label["occluded"] is True and label["difficult"] is True

    # 目标完全被挡住：可见度为 0，进入 ignore。
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    labels, _, _ = project_records(
        [record], identity, WIDTH, HEIGHT, FOV, np, ids=empty
    )
    assert labels[0]["visible_px"] == 0
    assert labels[0]["visibility"] == 0.0
    assert labels[0]["ignore"] is True
    assert "ignored_low_visibility" in labels[0]["quality_flags"]

    # 在线结果和离线重投影（用快照里的可见像素统计）必须一致。
    offline_labels, _, _ = project_records(
        [record],
        identity,
        WIDTH,
        HEIGHT,
        FOV,
        np,
        visible_by_actor=visibility,
    )
    assert offline_labels == online_labels


def test_tiny_flag() -> None:
    identity = np.eye(4)
    labels, _, _ = project_records(
        [_actor(location=(60.0, 0.0, 0.0), extent=(0.2, 0.2, 0.2))],
        identity,
        WIDTH,
        HEIGHT,
        FOV,
        np,
    )
    label = labels[0]
    assert label["bbox"][2] * label["bbox"][3] < TINY_AREA_PX
    assert label["difficult"] is True
    assert "tiny" in label["quality_flags"]


def test_track_statistics() -> None:
    frames = [
        {
            "annotations": [
                {
                    "actor_id": 1,
                    "category": "pedestrian",
                    "speed_mps": 1.2,
                },
                {"actor_id": 2, "category": "pedestrian", "speed_mps": 0.0},
            ]
        }
        for _ in range(10)
    ]
    stats = track_statistics(frames, 50_000)
    assert stats["pedestrian"]["tracks"] == 2
    assert stats["pedestrian"]["moving_tracks"] == 1
    assert stats["pedestrian"]["static_tracks"] == 1
    assert stats["pedestrian"]["mean_displacement_m"] > 0


def test_event_statistics() -> None:
    import h5py

    with tempfile.TemporaryDirectory() as temporary:
        events_path = Path(temporary) / "events.h5"
        with h5py.File(events_path, "w") as handle:
            handle.create_dataset("events/t", data=np.arange(4, dtype="u8"))
            handle.create_dataset(
                "events/x", data=np.array([1, 2, 3, 4], dtype="u2")
            )
            handle.create_dataset(
                "events/y", data=np.array([1, 2, 3, 4], dtype="u2")
            )
            handle.create_dataset(
                "events/p", data=np.array([1, -1, 1, -1], dtype="i1")
            )
            handle.create_dataset("frames/event_offsets", data=np.array([0, 4]))
            handle.create_dataset("meta/sensor_width", data=8, dtype="u2")
            handle.create_dataset("meta/sensor_height", data=8, dtype="u2")
        annotations = {
            "frames": [
                {
                    "annotations": [
                        {
                            "bbox": [0, 0, 4, 4],
                            "stationary": True,
                            "quality_flags": [],
                        },
                        {
                            "bbox": [6, 6, 2, 2],
                            "stationary": False,
                            "quality_flags": [],
                        },
                    ]
                }
            ]
        }
        counts = annotate_camera(annotations, events_path, np)
        first, second = annotations["frames"][0]["annotations"]
        # 事件落在 (1,1) (2,2) (3,3) (4,4)；框 [0,0,4,4] 只覆盖前三个。
        assert first["event_count"] == 3
        assert first["active_pixel_ratio"] == round(3 / 16, 4)
        assert "low_event" not in first["quality_flags"]
        assert "static" in first["quality_flags"]
        assert second["event_count"] == 0
        assert "low_event" in second["quality_flags"]
        assert counts["instances"] == 2


def test_snapshot_roundtrip() -> None:
    """采集时写的快照必须能被离线重投影成同一份标注。"""
    from dvs_collector.dataset import SnapshotWriter

    from tools.annotate_offline import project_sequence

    identity = np.eye(4)
    record = _actor()
    ids = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    ids[160:200, 300:340] = record["actor_id"]
    online_labels, _, visibility = project_records(
        [record], identity, WIDTH, HEIGHT, FOV, np, ids=ids
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "cameras" / "cam_01").mkdir(parents=True)
        writer = SnapshotWriter(
            root / "snapshot.json",
            {
                "sequence_id": "sequence_000001",
                "class_map_version": CLASS_MAP_VERSION,
                "classes": dict(CLASSES),
                "cameras": {
                    "cam_01": {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "fov": FOV,
                        "world_matrix": identity.tolist(),
                    }
                },
            },
        )
        writer.add_frame(
            {
                "frame_index": 0,
                "carla_frame": 1,
                "timestamp_us": 100_000,
                "actors": [record],
                "visibility": {
                    "cam_01": {
                        str(record["actor_id"]): visibility[record["actor_id"]]
                    }
                },
            }
        )
        writer.close()

        project_sequence(root, np)
        payload = json.loads(
            (root / "cameras" / "cam_01" / "annotations.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["class_map_version"] == CLASS_MAP_VERSION
        assert payload["frames"][0]["annotations"] == online_labels


def test_annotation_validation() -> None:
    """质检工具的标注检查：越界、重复 actor、无原因的 ignore 都要报错。"""
    from tools.check_dataset import validate_annotations

    def _frame(annotations: list[dict]) -> dict:
        return {"frames": [{"annotations": annotations}]}

    good = _frame(
        [
            {
                "actor_id": 1,
                "bbox": [10, 20, 30, 40],
                "ignore": False,
                "quality_flags": [],
            }
        ]
    )
    errors, warnings = validate_annotations(good, 640, 360, "t")
    assert not errors and not warnings, (errors, warnings)

    out_of_bounds = _frame([{"actor_id": 1, "bbox": [600, 20, 100, 40]}])
    errors, _ = validate_annotations(out_of_bounds, 640, 360, "t")
    assert any("越界" in error for error in errors), errors

    duplicated = _frame(
        [
            {"actor_id": 1, "bbox": [10, 20, 30, 40]},
            {"actor_id": 1, "bbox": [12, 22, 30, 40]},
        ]
    )
    errors, _ = validate_annotations(duplicated, 640, 360, "t")
    assert any("重复" in error for error in errors), errors

    unjustified = _frame([{"actor_id": 1, "bbox": [10, 20, 30, 40], "ignore": True}])
    errors, _ = validate_annotations(unjustified, 640, 360, "t")
    assert any("ignore" in error for error in errors), errors

    # 相邻帧框尺寸突变只报警，不算错误。
    jumped = {
        "frames": [
            {"annotations": [{"actor_id": 1, "bbox": [10, 20, 30, 40]}]},
            {"annotations": [{"actor_id": 1, "bbox": [10, 20, 120, 160]}]},
        ]
    }
    errors, warnings = validate_annotations(jumped, 640, 360, "t")
    assert not errors, errors
    assert any("突变" in warning for warning in warnings), warnings


def test_detection_range() -> None:
    """超出相机有效探测范围的目标标 ignore，不进训练标签。"""
    identity = np.eye(4)

    # 近处大目标：可探测，不标 ignore。
    labels, _, _ = project_records(
        [_actor(location=(12.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))],
        identity,
        WIDTH,
        HEIGHT,
        FOV,
        np,
    )
    assert labels[0]["ignore"] is False
    assert labels[0]["ignore_reason"] is None
    assert "beyond_detection_range" not in labels[0]["quality_flags"]
    assert labels[0]["distance_m"] < 20.0

    # 远处小目标：投影短边不足 16 px，标 ignore 并给出原因。
    labels, _, _ = project_records(
        [_actor(location=(400.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))],
        identity,
        WIDTH,
        HEIGHT,
        FOV,
        np,
    )
    label = labels[0]
    assert label["ignore"] is True
    assert label["ignore_reason"] == "beyond_detection_range"
    assert "beyond_detection_range" in label["quality_flags"]
    # 目标仍保留在文件里：删掉会丢掉真实背景分布。
    assert label["bbox"][2] > 0 and label["bbox"][3] > 0
    assert label["distance_m"] > 100.0

    # 阈值可调：放宽到 1 px 后，同样的远目标不再被忽略。
    labels, _, _ = project_records(
        [_actor(location=(400.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))],
        identity,
        WIDTH,
        HEIGHT,
        FOV,
        np,
        min_short_side_px=1,
        max_distance_m=10_000.0,
    )
    assert labels[0]["ignore"] is False


def test_detectability_helper() -> None:
    """判据函数：短边与距离两条都要满足。"""
    focal = WIDTH / (2.0 * np.tan(np.radians(FOV) / 2.0))
    # 短边够大、距离够近 → 可探测。
    detectable, distance = detectability([0, 0, 40, 40], focal, 1.5)
    assert detectable is True and distance > 0
    # 短边太小 → 不可探测。
    detectable, _ = detectability([0, 0, 5, 5], focal, 1.5)
    assert detectable is False
    # 距离超限 → 不可探测（短边够大也不行）。
    detectable, distance = detectability(
        [0, 0, 40, 1], focal, 1.5, max_distance_m=1.0
    )
    assert detectable is False


def main() -> int:
    tests = [
        test_class_map,
        test_projection,
        test_near_plane_is_dropped,
        test_truncated_only_when_clipped,
        test_visibility_rules,
        test_tiny_flag,
        test_detection_range,
        test_detectability_helper,
        test_track_statistics,
        test_event_statistics,
        test_snapshot_roundtrip,
        test_annotation_validation,
    ]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"全部 {len(tests)} 项自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
