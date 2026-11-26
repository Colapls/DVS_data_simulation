#!/usr/bin/env python3
"""位置时间标注的自检（不连接 CARLA）。

    python tools/test_location_metadata.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dvs_collector.collector import _camera_poses, _geo_of  # noqa: E402
from dvs_collector.config import load_rig  # noqa: E402
from dvs_collector.dataset import LOCATION_COLUMNS, DatasetLayout  # noqa: E402
from tools.check_dataset import _validate_location_index  # noqa: E402


RIG_YAML = """
town: Town10HD_Opt
sequence_duration_s: 90.0
fps: 20.0
seed: null
location:
  map_name: Town10HD_Opt
  junction_id: 23
  junction_center_m: [-64.264, 7.063, 0.0]
  geo: {latitude: 41.8712, longitude: -87.6298, altitude: 181.0}
  aim: junction_center
cameras:
  cam_01:
    location: [-64.264, 7.063, 4.054]
    pitch: -10.88
    yaw: 36.402
    fov: 60.0
    preset: road_day
traffic: {vehicles: 80, walkers: 40}
weather_cycle: [ClearNoon]
"""


class _StubMap:
    """只实现坐标换算的地图替身。"""

    def transform_to_geolocation(self, location):
        return SimpleNamespace(
            latitude=location.x / 1000.0,
            longitude=location.y / 1000.0,
            altitude=location.z,
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. 机位组 YAML 里的位置信息要被采集器读到。
        rig_path = root / "rig.yaml"
        rig_path.write_text(RIG_YAML, encoding="utf-8")
        rig = load_rig(rig_path)
        assert rig.location["junction_id"] == 23, rig.location
        assert rig.location["geo"]["latitude"] == 41.8712, rig.location

        # 2. 地理坐标换算与机位摘要。
        geo = _geo_of(_StubMap(), SimpleNamespace(Location=lambda **kw: SimpleNamespace(**kw)), 1.0, 2.0, 3.0)
        assert geo == {"latitude": 0.001, "longitude": 0.002, "altitude": 3.0}, geo
        poses = _camera_poses(
            {
                "cameras": {
                    "cam_01": {
                        "location_m": [-64.264, 7.063, 4.054],
                        "rotation_degrees": {"pitch": -10.88, "yaw": 36.402, "roll": 0.0},
                        "geo": {"latitude": 41.8712, "longitude": -87.6298},
                    }
                }
            }
        )
        assert "cam_01:x=-64.26" in poses and "geo=41.8712" in poses, poses

        # 3. 位置时间索引：写一行、按序列号覆盖、字段齐全。
        layout = DatasetLayout(root / "dataset")
        layout.initialize(rig)
        record = {column: "" for column in LOCATION_COLUMNS}
        record.update(
            sequence_id="sequence_000001",
            capture_started_at_utc="2026-10-26T06:45:12+00:00",
            capture_started_at_local="2026-10-26T14:45:12+08:00",
            timezone="中国标准时间",
            town="Town10HD_Opt",
            map_name="Town10HD_Opt",
            junction_id=23,
            camera_count=1,
            camera_poses=poses,
        )
        layout.write_location_record(record)
        layout.write_location_record({**record, "weather": "ClearNoon"})
        with (root / "dataset" / "locations.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1, rows
        assert rows[0]["weather"] == "ClearNoon", rows
        assert rows[0]["junction_id"] == "23", rows

        # 4. 索引缺行时质检必须报错。
        (root / "dataset" / "index_all.txt").write_text(
            "sequence_000001\nsequence_000002\n", encoding="utf-8"
        )
        errors = _validate_location_index(root / "dataset", ["sequence_000001", "sequence_000002"])
        assert any("sequence_000002" in error for error in errors), errors

    print("位置时间标注自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
