from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classes import CLASSES, CLASS_MAP_VERSION

from .config import CameraConfig, RigConfig


SEQUENCE_PATTERN = re.compile(r"^sequence_(\d{6})$")

# 位置时间索引的列顺序；每个序列一行，重复采集同一路口时按行追加。
LOCATION_COLUMNS = (
    "sequence_id",
    "capture_started_at_utc",
    "capture_started_at_local",
    "timezone",
    "duration_s",
    "town",
    "map_name",
    "junction_id",
    "junction_center_x_m",
    "junction_center_y_m",
    "geo_latitude",
    "geo_longitude",
    "geo_altitude_m",
    "camera_count",
    "camera_poses",
    "frame_count",
    "fps",
    "weather",
    "seed",
    "total_events",
    "vehicles",
    "walkers",
)


DATASET_README = """# CARLA 路侧多机位 DVS+RGB 事件流数据集

所有序列平铺在 `sequences/` 里，**不做 train/val/test 划分**；`index_all.txt`
是全量索引（每行一个成功序列），`locations.csv` 记录每条序列的采集位置
（地图、路口、地理坐标）和采集时间。需要划分时按 `locations.csv` 自行过滤。

完整说明见项目文档 `docs/数据集说明.md`（逐个文件、逐个字段的含义）。

```text
sequences/sequence_XXXXXX/
├── cameras/cam_XX/
│   ├── events.h5          事件流：/events/{t,x,y,p}、/frames/*、/meta/*
│   ├── rgb/000000.png     与事件同 tick 的 RGB 帧
│   ├── timestamps.txt     每帧时间戳（微秒，相对序列起点）
│   └── annotations.json   逐帧检测标注（三类别 + 可见度 + 质量标记）
├── calibration/*.yaml     内外参、DVS↔RGB 外参、时间偏移
├── event_representations/ 事件帧可视化输出目录（当前为空，预留）
├── snapshot.json          采集时的 actor 快照（位姿/蓝图/可见像素），投影可离线重算
└── meta/                  episode_manifest.json / dvs_params.json / rig_geometry.json
```

要点：

- `events/t` 的单位是**纳秒**（`meta/event_timestamp_unit`），`frames/timestamp_us`
  是**微秒**；`events/t / 1000` 即与帧时间同一时间轴。
- 标注类别是 `car` / `motorcycle` / `pedestrian`（`class_map_version` 记录映射表版本）。
- `snapshot.json` 保存了不可重算的信息（蓝图 ID、位姿、可见像素），
  用 `tools/annotate_offline.py project` 可以不连 CARLA 重新投影。
- 本数据为仿真数据，DVS 与 RGB 同位同 tick，标定是理论值而非硬件实测标定。
- 质检：`python tools/check_dataset.py --dataset <本目录>`。
"""


@dataclass(frozen=True)
class CameraPaths:
    """单个视角在当前序列目录中的文件路径。"""

    camera_id: str
    root: Path
    events_file: Path
    rgb_dir: Path
    timestamps_file: Path
    annotations_file: Path
    synced_rgb_relative: str


@dataclass(frozen=True)
class SequencePaths:
    """一次多机位采集共享的序列目录和各相机子目录。"""

    sequence_id: str
    root: Path
    meta_dir: Path
    calibration_dir: Path
    snapshot_file: Path
    cameras: dict[str, CameraPaths]

    @property
    def manifest_file(self) -> Path:
        return self.meta_dir / "episode_manifest.json"


class SnapshotWriter:
    """流式写 snapshot.json：采集时只存快照，投影放到离线做。

    快照保存的是事后**算不出来**的信息：蓝图 ID、actor 位姿、3D 框几何、
    速度，以及每个机位上每个 actor 的可见像素统计。有了它，投影代码出 bug
    时重跑一个脚本即可，不必重跑仿真。
    """

    def __init__(self, path: Path, header: dict[str, Any]) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8", newline="\n")
        self._file.write("{\n")
        for key, value in header.items():
            self._file.write(
                f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},\n"
            )
        self._file.write('"frames": [\n')
        self._first = True
        self._closed = False

    def add_frame(self, frame: dict[str, Any]) -> None:
        """追加一帧的 actor 快照。"""
        if self._closed:
            raise RuntimeError("快照已经写完，不能再追加")
        if not self._first:
            self._file.write(",\n")
        self._first = False
        self._file.write(
            json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
        )

    def close(self) -> None:
        """收尾并关闭文件。"""
        if self._closed:
            return
        self._closed = True
        self._file.write("\n]\n}\n")
        self._file.close()

    def __enter__(self) -> "SnapshotWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class DatasetLayout:

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def initialize(self, rig: RigConfig) -> None:
        """创建数据集公共目录，并保存本次使用的 rig 与质检工具副本。"""
        for relative in (
            "sequences",
            "calibration/rigs",
            "tools",
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        rig_copy = self.root / "calibration" / "rigs" / rig.source.name
        if rig.source != rig_copy.resolve():
            shutil.copy2(rig.source, rig_copy)
        # 杆位明细（与机位组同目录的 poles.json）一并留档，便于追溯杆位来源。
        poles_copy = self.root / "calibration" / "rigs" / "poles.json"
        poles_source = rig.source.parent / "poles.json"
        if poles_source.is_file() and poles_source.resolve() != poles_copy.resolve():
            shutil.copy2(poles_source, poles_copy)
        checker_source = (
            Path(__file__).resolve().parent.parent / "tools" / "check_dataset.py"
        )
        checker_target = self.root / "tools" / checker_source.name
        if checker_source.resolve() != checker_target.resolve():
            shutil.copy2(checker_source, checker_target)
        fixer_source = (
            Path(__file__).resolve().parent.parent / "tools" / "fix_event_timestamps.py"
        )
        fixer_target = self.root / "tools" / fixer_source.name
        if fixer_source.resolve() != fixer_target.resolve():
            shutil.copy2(fixer_source, fixer_target)
        # README 里让用户用 tools/annotate_offline.py 重投影，得真的带上它。
        annotator_source = (
            Path(__file__).resolve().parent.parent / "tools" / "annotate_offline.py"
        )
        if annotator_source.is_file():
            annotator_target = self.root / "tools" / annotator_source.name
            if annotator_source.resolve() != annotator_target.resolve():
                shutil.copy2(annotator_source, annotator_target)
        index = self.root / "index_all.txt"
        index.touch(exist_ok=True)
        self._locations_file().touch(exist_ok=True)
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(DATASET_README, encoding="utf-8")
        changelog = self.root / "CHANGELOG.md"
        if not changelog.exists():
            changelog.write_text(
                "# Changelog\n\n- Initial roadside multi-camera dataset.\n",
                encoding="utf-8",
            )
        license_file = self.root / "LICENSE.txt"
        if not license_file.exists():
            license_file.write_text(
                "No redistribution license has been assigned. "
                "Do not redistribute until the dataset owner specifies a license.\n",
                encoding="utf-8",
            )

    def allocate_sequence(self, camera_ids: list[str]) -> SequencePaths:
        """使用全局递增序列号创建所有相机目录；失败目录也保留编号。"""
        used = self._existing_sequence_numbers()
        number = max(used, default=0) + 1
        while number <= 999999:
            sequence_id = f"sequence_{number:06d}"
            sequence_root = self.root / "sequences" / sequence_id
            try:
                sequence_root.mkdir()
                break
            except FileExistsError:
                number += 1
        else:
            raise ValueError("sequence_ 六位序列编号已用尽")

        cameras: dict[str, CameraPaths] = {}
        for camera_id in camera_ids:
            camera_root = sequence_root / "cameras" / camera_id
            rgb_dir = camera_root / "rgb"
            rgb_dir.mkdir(parents=True)
            relative_rgb = (
                Path("sequences")
                / sequence_id
                / "cameras"
                / camera_id
                / "rgb"
            )
            cameras[camera_id] = CameraPaths(
                camera_id=camera_id,
                root=camera_root,
                events_file=camera_root / "events.h5",
                rgb_dir=rgb_dir,
                timestamps_file=camera_root / "timestamps.txt",
                annotations_file=camera_root / "annotations.json",
                synced_rgb_relative=relative_rgb.as_posix() + "/",
            )
        meta_dir = sequence_root / "meta"
        meta_dir.mkdir()
        calibration_dir = sequence_root / "calibration"
        calibration_dir.mkdir()
        (sequence_root / "event_representations").mkdir()
        return SequencePaths(
            sequence_id=sequence_id,
            root=sequence_root,
            meta_dir=meta_dir,
            calibration_dir=calibration_dir,
            snapshot_file=sequence_root / "snapshot.json",
            cameras=cameras,
        )

    def write_annotations(
        self,
        sequence: SequencePaths,
        camera: CameraConfig,
        frames: list[dict[str, Any]],
    ) -> None:
        """写入单视角 RGB 帧锚定的序列级检测标注。"""
        payload = {
            "sequence_id": sequence.sequence_id,
            "camera_id": camera.camera_id,
            "sensor_width": camera.width,
            "sensor_height": camera.height,
            "class_map_version": CLASS_MAP_VERSION,
            "classes": dict(CLASSES),
            "frames": frames,
        }
        path = sequence.cameras[camera.camera_id].annotations_file
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_calibration(
        self,
        sequence: SequencePaths,
        cameras: tuple[CameraConfig, ...],
        runtime_calibration: dict[str, dict[str, Any]],
        np: Any,
    ) -> None:
        """保存逐相机内外参、DVS/RGB 同位关系和相机间相对位姿。"""
        intrinsics: dict[str, Any] = {}
        extrinsics: dict[str, Any] = {}
        dvs_rgb: dict[str, Any] = {}
        time_offsets: dict[str, Any] = {}
        rig_geometry: dict[str, Any] = {}
        # 以配置排序后的第一台相机作为 rig 参考坐标系。
        reference_matrix = None

        for camera in cameras:
            values = runtime_calibration[camera.camera_id]
            matrix = np.asarray(values["world_matrix"], dtype=np.float64)
            if reference_matrix is None:
                reference_matrix = matrix
            focal_length = camera.width / (
                2.0 * math.tan(math.radians(camera.fov) / 2.0)
            )
            intrinsics[camera.camera_id] = {
                "camera_model": "pinhole",
                "width": camera.width,
                "height": camera.height,
                "fx": focal_length,
                "fy": focal_length,
                "cx": camera.width / 2.0,
                "cy": camera.height / 2.0,
                "distortion_model": "none",
                "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
            extrinsics[camera.camera_id] = {
                "camera_to_world_matrix": values["world_matrix"],
                "location_m": list(camera.location),
                "rotation_degrees": {
                    "pitch": camera.pitch,
                    "yaw": camera.yaw,
                    "roll": camera.roll,
                },
            }
            dvs_rgb[camera.camera_id] = {
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_m": [0.0, 0.0, 0.0],
                "calibration_status": "theoretical_same_transform_not_hardware_measured",
            }
            time_offsets[camera.camera_id] = {
                "events_to_rgb_offset_us": 0,
                "alignment_method": "same CARLA simulation tick",
                "hardware_measurement": False,
            }
            # 将相机到世界位姿转换为相机到参考机位的相对变换。
            relative = np.linalg.inv(reference_matrix) @ matrix
            rig_geometry[camera.camera_id] = {
                "camera_to_world_matrix": values["world_matrix"],
                "camera_to_reference_matrix": relative.tolist(),
            }

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("写入标定 YAML 需要 PyYAML") from exc
        files = {
            "extrinsics_world.yaml": extrinsics,
            "intrinsics_dvs.yaml": intrinsics,
            "intrinsics_rgb.yaml": intrinsics,
            "dvs_rgb_extrinsic.yaml": dvs_rgb,
            "time_offset.yaml": time_offsets,
        }
        for filename, content in files.items():
            (sequence.calibration_dir / filename).write_text(
                yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        (sequence.meta_dir / "rig_geometry.json").write_text(
            json.dumps(rig_geometry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_manifest(self, sequence: SequencePaths, manifest: dict[str, Any]) -> None:
        """持续更新序列清单，使失败和成功状态均可追溯。"""
        sequence.manifest_file.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_dvs_params(
        self,
        sequence: SequencePaths,
        cameras: tuple[CameraConfig, ...],
        applied_exposure_modes: dict[str, str | None],
    ) -> None:
        """记录每台相机实际配置和 CARLA 是否接受了曝光模式参数。"""
        params = {
            camera.camera_id: {
                **camera.dvs_parameters(),
                "applied_exposure_mode": applied_exposure_modes[camera.camera_id],
            }
            for camera in cameras
        }
        (sequence.meta_dir / "dvs_params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def register_sequence(self, sequence_id: str) -> None:
        """登记新序列并把索引重建为"磁盘上状态为 completed 的序列"。

        只追加不清理的旧逻辑会让索引指向已被删除的序列（实测：索引 11 条、
        磁盘 6 条，其中还包括唯一一条夜间序列）。每次登记时按磁盘重建，
        删掉的序列自动出索引，新序列去重后仍写入。
        """
        index = self.root / "index_all.txt"
        existing = set(index.read_text(encoding="utf-8").splitlines())
        if sequence_id in existing:
            return
        completed = self._completed_sequence_ids()
        if sequence_id not in completed:
            completed.append(sequence_id)
        index.write_text(
            "\n".join(sorted(completed)) + "\n", encoding="utf-8", newline="\n"
        )

    def _completed_sequence_ids(self) -> list[str]:
        """磁盘上存在且 manifest 标记 completed 的序列（按编号排序）。"""
        sequences = self.root / "sequences"
        done: list[str] = []
        if sequences.is_dir():
            for path in sequences.iterdir():
                if not SEQUENCE_PATTERN.fullmatch(path.name):
                    continue
                manifest = path / "meta" / "episode_manifest.json"
                try:
                    status = json.loads(
                        manifest.read_text(encoding="utf-8")
                    ).get("status")
                except (OSError, ValueError):
                    continue
                if status == "completed":
                    done.append(path.name)
        return sorted(done)

    def write_location_record(self, record: dict[str, Any]) -> Path:
        """把一条序列的采集位置和时间写入 locations.csv（同一序列覆盖旧行）。"""
        path = self._locations_file()
        rows: list[dict[str, str]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as source:
                rows = [
                    row
                    for row in csv.DictReader(source)
                    if row.get("sequence_id") != record.get("sequence_id")
                ]
        rows.append(
            {
                column: (
                    ""
                    if record.get(column) is None
                    else str(record.get(column, ""))
                )
                for column in LOCATION_COLUMNS
            }
        )
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=LOCATION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _locations_file(self) -> Path:
        """位置时间索引文件路径。"""
        return self.root / "locations.csv"

    def _existing_sequence_numbers(self) -> set[int]:
        """从序列目录和索引中恢复已占用编号，避免覆盖旧数据。"""
        sequences = self.root / "sequences"
        used: set[int] = set()
        if sequences.exists():
            for path in sequences.iterdir():
                match = SEQUENCE_PATTERN.fullmatch(path.name)
                if match:
                    used.add(int(match.group(1)))
        index = self.root / "index_all.txt"
        if index.exists():
            for value in index.read_text(encoding="utf-8").splitlines():
                match = SEQUENCE_PATTERN.fullmatch(value.strip())
                if match:
                    used.add(int(match.group(1)))
        return used
