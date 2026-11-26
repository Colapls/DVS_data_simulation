"""采集主流程：初始化路侧场景、同步采集、生成标注并恢复 CARLA 状态。

采集时只做两件事：把事件流和 RGB 落盘、把不可重算的信息写进 snapshot.json；
2D 投影和可见度判定也在这里完成一遍（在线），但事后可以完全离线重算
（``tools/annotate_offline.py project``）。
"""

from __future__ import annotations

import contextlib
import csv
import math
import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classes import CLASSES, CLASS_MAP_VERSION, classify_blueprint
from .config import load_dependencies
from .labels import (
    MAX_DETECT_DISTANCE_M,
    MIN_DETECT_SHORT_SIDE_PX,
    actor_records,
    decode_instance_ids,
    project_records,
    track_statistics,
)
from .storage import CameraDiskWriter

from .config import (
    CameraConfig,
    RigConfig,
    load_rig,
    parse_args,
    select_cameras,
    validate_args,
)
from .dataset import DatasetLayout, SequencePaths, SnapshotWriter
from .sensors import (
    CameraRig,
    SensorFrameReader,
    capture_calibration,
    spawn_static_camera_rig,
)
from .traffic import spawn_road_traffic, update_walkers

#: 连续缺帧达到这个数就判定传感器挂了（等一次超时要 --sensor-timeout 秒，
#: 5 次就是 75 s，不能再等了）。
MAX_CONSECUTIVE_SKIPPED_FRAMES = 5
#: 整条序列允许跳过的帧数上限，超了就停：跳太多说明数据不完整。
MAX_SKIPPED_FRAMES = 20


def pick_balanced_weather(
    cycle: tuple[str, ...],
    used_counts: dict[str, int],
    rng: random.Random,
) -> str:
    """选数据集里用得最少的天气；并列时由 rng 决定（种子固定则可复现）。

    之前的纯随机 choice 会让天气分布漂移（实测 6 条序列里 WetSunset、
    CloudyNoon、SoftRainSunset 各重复 2 次，一次夜间都没有）。按"缺口
    优先"补齐后，前 N 次采集必然覆盖全部 N 种天气。
    """
    counts = {weather: int(used_counts.get(weather, 0)) for weather in cycle}
    fewest = min(counts.values())
    return rng.choice(
        [weather for weather, count in counts.items() if count == fewest]
    )


def _weather_usage(locations_csv: Path) -> dict[str, int]:
    """统计 locations.csv 里已采集序列的天气分布。"""
    usage: dict[str, int] = {}
    if not locations_csv.is_file():
        return usage
    with locations_csv.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            weather = (row.get("weather") or "").strip()
            if weather:
                usage[weather] = usage.get(weather, 0) + 1
    return usage


def _weather_for_sequence(
    carla: Any,
    args: Any,
    rig: RigConfig,
    seed: int,
    dataset_root: Path | None = None,
) -> str:
    """命令行天气优先；否则按数据集天气缺口均衡选择（种子固定可复现）。"""
    if args.weather:
        weather_name = args.weather
    else:
        used = (
            _weather_usage(dataset_root / "locations.csv")
            if dataset_root is not None
            else {}
        )
        weather_name = pick_balanced_weather(
            rig.weather_cycle, used, random.Random(seed)
        )
    if not hasattr(carla.WeatherParameters, weather_name):
        raise ValueError(f"当前 CARLA 版本不支持天气预设：{weather_name}")
    return weather_name


def _collect_frames(
    world: Any,
    rig: CameraRig,
    sequence: SequencePaths,
    cameras: tuple[CameraConfig, ...],
    frames: int,
    time_origin_us: int,
    integration_time_us: int,
    sensor_timeout: float,
    min_speed: float,
    np_module: Any,
    carla: Any,
    walker_plans: list[dict[str, Any]] | None = None,
    instance_seg: bool = True,
    rgb_format: str = "png",
    rgb_quality: int = 90,
    min_detect_px: int = MIN_DETECT_SHORT_SIDE_PX,
    max_detect_distance: float = MAX_DETECT_DISTANCE_M,
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]], int, int]:
    """按 CARLA frame 配对传感器数据；落盘交给每相机一个后台线程。

    RGB 和实例分割按 frame 号严格配对（多机位之间必须帧号对齐）；DVS 按
    事件流取，画面静止时这一帧没有事件，写空帧而不是把采集卡死。

    返回 (各机位事件数, 各机位标注, 实际落盘帧数, 因缺数据被跳过的帧数)。
    """
    frame_reader = SensorFrameReader(rig, sensor_timeout)
    annotations = {camera.camera_id: [] for camera in cameras}
    # 每个机位连续多少帧没有 DVS 事件：用来发现"传感器彻底停了"的情况。
    empty_dvs_streak = {camera.camera_id: 0 for camera in cameras}
    # 每相机一个后台写入线程：PNG 编码和 HDF5 追加不再占用采集主循环，
    # 多相机同时写盘的卡顿被队列吸收，不会传导成传感器回调超时。
    disk_writers = {
        camera.camera_id: CameraDiskWriter(
            rig.writers[camera.camera_id],
            sequence.cameras[camera.camera_id].rgb_dir,
            sequence.cameras[camera.camera_id].timestamps_file,
            label=camera.camera_id,
            rgb_format=rgb_format,
            rgb_quality=rgb_quality,
        )
        for camera in cameras
    }
    for disk_writer in disk_writers.values():
        disk_writer.start()

    # 机位是固定的：相机位姿不会变，只取一次逆矩阵，省掉每帧每相机的 RPC。
    inverse_matrices = {
        camera.camera_id: np_module.asarray(
            rig.dvs_sensors[camera.camera_id]
            .get_transform()
            .get_inverse_matrix(),
            dtype=np_module.float64,
        )
        for camera in cameras
    }
    snapshot = SnapshotWriter(
        sequence.snapshot_file,
        {
            "sequence_id": sequence.sequence_id,
            "class_map_version": CLASS_MAP_VERSION,
            "classes": dict(CLASSES),
            "cameras": {
                camera.camera_id: {
                    "width": camera.width,
                    "height": camera.height,
                    "fov": camera.fov,
                    # 存投影实际使用的世界→相机矩阵（CARLA 解析式求逆）。
                    # 离线重投影直接用这个矩阵，不再做数值反演，
                    # 否则与在线结果会有浮点差异（像素边界处差 1 px）。
                    "inverse_matrix": [
                        [float(value) for value in row]
                        for row in np_module.asarray(
                            rig.dvs_sensors[camera.camera_id]
                            .get_transform()
                            .get_inverse_matrix(),
                            dtype=np_module.float64,
                        )
                    ],
                    "world_matrix": rig.calibration[camera.camera_id][
                        "world_matrix"
                    ],
                }
                for camera in cameras
            },
        },
    )

    # 标签投影放入单独 worker；等待传感器期间可并行计算本帧的 actor 快照。
    started = time.perf_counter()
    written = 0
    skipped = 0
    consecutive_skips = 0
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="bbox-projection"
    ) as label_worker:
        try:
            while written < frames:
                # 行人没有 AI 控制器（导航网格在部分地图上不可用），
                # 每个 tick 前手动推一步。
                if walker_plans:
                    update_walkers(carla, walker_plans)
                # 同步模式下每次 tick 推进固定仿真步长，各传感器回调带同一 frame ID。
                carla_frame = world.tick()
                # RGB 时间戳和 DVS 事件使用同一个序列起点，单位均为微秒。
                timestamp_us = round(
                    world.get_snapshot().timestamp.elapsed_seconds * 1_000_000
                ) - time_origin_us
                records_future = label_worker.submit(
                    actor_records, world, min_speed
                )
                # 先把这一帧所有传感器的数据取齐，再决定写不写：
                # 任何一个机位缺这一帧，整帧就作废。多机位之间必须帧号
                # 对齐（标注和事件要能按帧配对），不能让 cam_01 有、
                # cam_02 没有。连续缺太多帧说明传感器真的挂了，直接中止。
                # （DVS 除外：它没有事件时就不发数据，见下面的 get_dvs。）
                try:
                    # 先把所有机位的 RGB/分割取齐，再取 DVS：
                    # 服务端按 cam_XX_dvs → rgb → seg 的顺序发同一帧的数据，
                    # 等到最后一个机位的 RGB 就说明这一帧的 DVS 都已经发出来了，
                    # DVS 不用再等；而且某个机位缺帧时不会白吃掉别机位的 DVS。
                    rgb_seg: list[tuple[Any, Any, Any]] = []
                    for camera in cameras:
                        camera_id = camera.camera_id
                        rgb_data = frame_reader.get(f"{camera_id}_rgb", carla_frame)
                        seg_data = (
                            frame_reader.get(f"{camera_id}_seg", carla_frame)
                            if instance_seg
                            else None
                        )
                        rgb_seg.append((camera, rgb_data, seg_data))
                    pending: list[tuple[Any, Any, Any, Any]] = []
                    for camera, rgb_data, seg_data in rgb_seg:
                        camera_id = camera.camera_id
                        # DVS 是流式的：这一帧没有事件时传感器根本不发数据，
                        # 死等 frame 号会把整条采集拖死（见 SensorFrameReader.get_dvs）。
                        dvs_raw = frame_reader.get_dvs(
                            f"{camera_id}_dvs", carla_frame
                        )
                        if dvs_raw:
                            empty_dvs_streak[camera_id] = 0
                        else:
                            empty_dvs_streak[camera_id] += 1
                            if empty_dvs_streak[camera_id] == 60:
                                print(
                                    f"提示：{camera_id} 的 DVS 已连续 60 帧没有事件，"
                                    "画面可能完全静止，也可能是传感器停止输出",
                                    flush=True,
                                )
                        pending.append(
                            (camera, dvs_raw, rgb_data, seg_data)
                        )
                except TimeoutError as exc:
                    records_future.cancel()
                    skipped += 1
                    consecutive_skips += 1
                    print(f"跳过 CARLA frame={carla_frame}：{exc}", flush=True)
                    if consecutive_skips >= MAX_CONSECUTIVE_SKIPPED_FRAMES:
                        raise
                    if skipped >= MAX_SKIPPED_FRAMES:
                        raise RuntimeError(
                            f"已有 {skipped} 帧因传感器缺数据被跳过，"
                            "采集质量不可用，停止采集"
                        ) from exc
                    continue
                consecutive_skips = 0
                frame_records: dict[str, dict[str, Any]] = {}
                segmentation_data: dict[str, Any] = {}

                for camera, dvs_raw, rgb_data, seg_data in pending:
                    camera_id = camera.camera_id
                    frame_id = (
                        f"{sequence.sequence_id}_{camera_id}_{written:06d}"
                    )
                    disk_writers[camera_id].submit(
                        written,
                        frame_id,
                        timestamp_us,
                        dvs_raw,
                        rgb_data,
                    )
                    frame_records[camera_id] = {
                        "frame_id": frame_id,
                        "timestamp_us": timestamp_us,
                        "integration_time_us": integration_time_us,
                    }
                    segmentation_data[camera_id] = seg_data
                frame_index = written
                written += 1
                # 后台写盘线程一旦失败（磁盘满等），立即中止而不是卡到超时。
                for disk_writer in disk_writers.values():
                    disk_writer.check()

                records = records_future.result()
                visibility_by_camera: dict[str, dict[str, list[int]]] = {}
                for camera in cameras:
                    camera_id = camera.camera_id
                    ids = (
                        decode_instance_ids(
                            segmentation_data[camera_id], np_module
                        )
                        if instance_seg
                        else None
                    )
                    scale_x, scale_y = rig.segmentation_scales.get(
                        camera_id, (1.0, 1.0)
                    )
                    labels, ignored, visibility = project_records(
                        records,
                        inverse_matrices[camera_id],
                        camera.width,
                        camera.height,
                        camera.fov,
                        np_module,
                        ids,
                        scale_x,
                        scale_y,
                        min_short_side_px=min_detect_px,
                        max_distance_m=max_detect_distance,
                    )
                    frame_records[camera_id]["annotations"] = labels
                    frame_records[camera_id]["ignored_annotations"] = ignored
                    annotations[camera_id].append(frame_records[camera_id])
                    visibility_by_camera[camera_id] = {
                        str(actor_id): values
                        for actor_id, values in visibility.items()
                    }

                snapshot.add_frame(
                    {
                        "frame_index": frame_index,
                        "carla_frame": carla_frame,
                        "timestamp_us": timestamp_us,
                        "actors": records,
                        "visibility": visibility_by_camera,
                    }
                )

                if (
                    frame_index == 0
                    or (frame_index + 1) % 10 == 0
                    or written >= frames
                ):
                    elapsed = time.perf_counter() - started
                    live_counts = {
                        camera_id: disk_writer.count
                        for camera_id, disk_writer in disk_writers.items()
                    }
                    print(
                        f"采集进度 {written}/{frames} 帧"
                        f"（已用 {elapsed:.0f} s，{elapsed / max(written, 1):.2f} s/帧，"
                        f"跳过 {skipped} 帧），"
                        f"各机位累计事件数 {live_counts}",
                        flush=True,
                    )
        except BaseException:
            # 中止时也把已入队的完整帧写完再退出，保留现场。
            for disk_writer in disk_writers.values():
                with contextlib.suppress(Exception):
                    disk_writer.finish()
            snapshot.close()
            raise
        event_counts = {}
        for camera_id, disk_writer in disk_writers.items():
            disk_writer.finish()
            event_counts[camera_id] = disk_writer.count
    snapshot.close()
    return event_counts, annotations, written, skipped


def _cleanup(
    client: Any,
    carla: Any,
    world: Any,
    traffic_manager: Any,
    previous_settings: Any,
    previous_weather: Any,
    previous_tm_sync: bool,
    rig: CameraRig,
    spawned_actors: list[Any],
) -> None:
    """关闭写入器、销毁本次 actor，并恢复采集前的 CARLA 全局设置。"""
    errors: list[str] = []
    for sensor in rig.sensors:
        try:
            sensor.stop()
        except RuntimeError:
            pass
    for writer in rig.writers.values():
        try:
            writer.close()
        except (OSError, RuntimeError) as exc:
            errors.append(f"关闭 HDF5 写入器失败：{exc}")

    for actor in reversed(spawned_actors):
        if actor.type_id.startswith("controller.ai.walker"):
            try:
                actor.stop()
            except RuntimeError:
                pass
    actor_ids = [actor.id for actor in reversed(rig.sensors)]
    actor_ids.extend(actor.id for actor in reversed(spawned_actors))
    if actor_ids:
        commands = [
            carla.command.DestroyActor(actor_id) for actor_id in actor_ids
        ]
        try:
            # 0.9.16 起 apply_batch 不再返回逐条结果，只有 apply_batch_sync 有。
            responses = client.apply_batch_sync(commands, do_tick=True)
        except RuntimeError as exc:
            errors.append(f"提交 CARLA actor 清理命令失败：{exc}")
            responses = []
        errors.extend(
            f"清理 CARLA actor 失败：{response.error}"
            for response in responses
            if response.error and "not found" not in response.error.lower()
        )
        for actor_id, response in zip(actor_ids, responses):
            if response.error and "not found" in response.error.lower():
                actor_type = next(
                    (
                        f"{actor.type_id}(id={actor.id})"
                        for actor in (*rig.sensors, *spawned_actors)
                        if actor.id == actor_id
                    ),
                    f"id={actor_id}",
                )
                print(
                    f"清理提示：CARLA actor 已不存在，跳过重复销毁："
                    f"{actor_type}（{response.error}）",
                    file=sys.stderr,
                )
    try:
        traffic_manager.set_synchronous_mode(previous_tm_sync)
    except RuntimeError as exc:
        errors.append(f"恢复 Traffic Manager 同步模式失败：{exc}")
    try:
        world.set_weather(previous_weather)
    except RuntimeError as exc:
        errors.append(f"恢复 CARLA 天气设置失败：{exc}")
    try:
        world.apply_settings(previous_settings)
    except RuntimeError as exc:
        errors.append(f"恢复 CARLA world 设置失败：{exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _town_name(map_name: str) -> str:
    """去掉 CARLA 地图路径，仅保留地图名。"""
    return map_name.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]


def _geo_of(game_map: Any, carla: Any, x: float, y: float, z: float) -> dict[str, float] | None:
    """把 CARLA 世界坐标换算成地理坐标（地图自带地理参考）。"""
    try:
        geolocation = game_map.transform_to_geolocation(
            carla.Location(x=x, y=y, z=z)
        )
    except (AttributeError, RuntimeError):
        return None
    return {
        "latitude": round(float(geolocation.latitude), 7),
        "longitude": round(float(geolocation.longitude), 7),
        "altitude": round(float(geolocation.altitude), 3),
    }


def _camera_poses(location: dict[str, Any]) -> str:
    """把各机位的位置和朝向压成一行，便于在位置索引表中查看。"""
    parts = []
    for camera_id, values in location["cameras"].items():
        x, y, z = values["location_m"]
        rotation = values["rotation_degrees"]
        geo = values.get("geo")
        geo_text = (
            f",geo={geo['latitude']},{geo['longitude']}" if geo else ""
        )
        parts.append(
            f"{camera_id}:x={x:.2f},y={y:.2f},z={z:.2f},"
            f"yaw={rotation['yaw']:.1f},pitch={rotation['pitch']:.1f}{geo_text}"
        )
    return " | ".join(parts)


def _canonical_town(name: str) -> str:
    """合并 Town10/Town10HD 及优化地图后缀的别名，便于比较地图。"""
    canonical = _town_name(name)
    if canonical.lower().endswith("_opt"):
        canonical = canonical[:-4]
    if canonical.lower() == "town10":
        return "Town10HD"
    return canonical


def _sweep_leftover_actors(client: Any, carla: Any, world: Any) -> int:
    """销毁世界里残留的本采集器 actor（车辆/行人/控制器/相机传感器）。

    正常结束的采集在 finally 里自清；原生崩溃（栈溢出、被杀）没有这个
    机会，残留 actor 会污染后续序列。地图刚 load_world 过时世界是空的，
    这里自然扫不到东西。红绿灯等地图基础设施不在过滤清单里，不受影响。
    """
    actors = world.get_actors()
    doomed: list[Any] = []
    for pattern in (
        "vehicle.*",
        "walker.pedestrian.*",
        "controller.ai.*",
        "sensor.camera.*",
    ):
        doomed.extend(actors.filter(pattern))
    if not doomed:
        return 0
    client.apply_batch_sync(
        [carla.command.DestroyActor(actor.id) for actor in doomed], do_tick=True
    )
    return len(doomed)


def _record_unknown_blueprints(
    root: Path, blueprint_ids: set[str]
) -> Path:
    """把本序列出现过的、映射表没覆盖的蓝图类型记进 unknown_blueprints.csv。"""
    path = root / "unknown_blueprints.csv"
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        if not exists:
            writer.writerow(("blueprint_id", "reason"))
        for blueprint_id in sorted(blueprint_ids):
            writer.writerow((blueprint_id, "not_in_class_map"))
    return path


def _seen_blueprints(
    annotation_frames: dict[str, list[dict[str, Any]]]
) -> list[str]:
    """本序列实际出现过的目标蓝图 ID（用于核对类别映射表）。"""
    return sorted(
        {
            annotation["blueprint_id"]
            for frames in annotation_frames.values()
            for frame in frames
            for annotation in frame.get("annotations", [])
        }
    )


def collect(args: Any) -> Path:
    """运行一个多机位序列；只有采集、标注和标定成功后才登记索引。"""
    validate_args(args)
    rig_config = load_rig(args.rig)
    if args.cameras is not None:
        # 只启用前 N 台相机；其余机位不创建传感器，也不写标注。
        rig_config = select_cameras(rig_config, args.cameras)
    carla, h5py, np_module = load_dependencies()

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    # 真实墙钟时间：标注每条数据的实际采集时刻。
    started_at = datetime.now(timezone.utc)
    # 已在目标地图上就不再 load_world：CARLA 对"同图重载"并不稳定，
    # 实测会把客户端直接崩掉（exit 0xC00000FD 栈溢出）甚至带崩服务端。
    current_town = _town_name(client.get_world().get_map().name)
    needs_load = args.load_town and (
        _canonical_town(current_town).lower()
        != _canonical_town(rig_config.town).lower()
    )
    if needs_load:
        # 上一次采集可能原生崩溃在同步模式里，服务端的同步开关没人还原。
        # 同步模式下重载地图，新世界的传感器会不出帧（表现为采集首帧就
        # 等超时），先切回异步再重载。
        try:
            settings = client.get_world().get_settings()
            if settings.synchronous_mode:
                settings.synchronous_mode = False
                client.get_world().apply_settings(settings)
                print("提示：已把服务端切回异步模式后再重载地图", flush=True)
        except RuntimeError as exc:
            print(f"提示：切换异步模式失败（{exc}），继续", flush=True)
        world = client.load_world(rig_config.town)
    else:
        world = client.get_world()
    game_map = world.get_map()
    town = _town_name(game_map.name)
    if _canonical_town(town).lower() != _canonical_town(rig_config.town).lower():
        raise ValueError(
            f"机位组适用于 {rig_config.town}，当前 CARLA 地图是 {town}；"
            "请加载匹配地图或选择对应 rig YAML。"
        )
    # 跳过重载后，世界里可能残留崩溃采集泄漏的车辆/行人/传感器
    # （原生崩溃不走 finally 清理）。这里清一次场，保证干净开局。
    leftover = _sweep_leftover_actors(client, carla, world)
    if leftover:
        print(f"提示：清理了上个崩溃任务残留的 {leftover} 个 actor", flush=True)

    layout = DatasetLayout(args.output)
    layout.initialize(rig_config)
    print(
        f"地图已就绪：{town}（{len(rig_config.cameras)} 台相机）",
        flush=True,
    )
    sequence = layout.allocate_sequence(
        [camera.camera_id for camera in rig_config.cameras]
    )
    # 默认按全局序列编号派生种子，失败后重试也会因新编号形成独立场景。
    sequence_number = int(sequence.sequence_id.rsplit("_", 1)[1])
    seed = (
        args.seed
        if args.seed is not None
        else rig_config.seed
        if rig_config.seed is not None
        else sequence_number
    )
    weather = _weather_for_sequence(carla, args, rig_config, seed, layout.root)
    fps = args.fps or rig_config.fps
    fixed_delta = 1.0 / fps
    if args.frames is not None:
        frame_count = args.frames
        duration = frame_count / fps
    else:
        duration = args.duration or rig_config.sequence_duration_s
        frame_count = math.ceil(duration * fps)
    integration_time_us = round(fixed_delta * 1_000_000)
    vehicle_target = (
        args.vehicles if args.vehicles is not None else rig_config.traffic["vehicles"]
    )
    walker_target = (
        args.walkers if args.walkers is not None else rig_config.traffic["walkers"]
    )
    traffic_manager = client.get_trafficmanager(args.tm_port)
    previous_settings = world.get_settings()
    previous_weather = world.get_weather()
    previous_tm_sync = (
        traffic_manager.get_synchronous_mode()
        if hasattr(traffic_manager, "get_synchronous_mode")
        else previous_settings.synchronous_mode
    )
    camera_rig = CameraRig(camera_names=[])
    spawned_actors: list[Any] = []
    # 采集位置：路口信息来自机位组 YAML，每台相机的地理坐标按实际位姿运行时换算。
    rig_location = dict(rig_config.location)
    location = {
        "town": town,
        "map_name": game_map.name,
        "junction_id": rig_location.get("junction_id"),
        "junction_center_m": rig_location.get("junction_center_m"),
        "geo": rig_location.get("geo"),
        "aim": rig_location.get("aim"),
        "cameras": {
            camera.camera_id: {
                "location_m": list(camera.location),
                "rotation_degrees": {
                    "pitch": camera.pitch,
                    "yaw": camera.yaw,
                    "roll": camera.roll,
                },
                "geo": _geo_of(game_map, carla, *camera.location),
            }
            for camera in rig_config.cameras
        },
    }
    local_start = started_at.astimezone()
    capture_time: dict[str, Any] = {
        "started_at_utc": started_at.isoformat(),
        "started_at_local": local_start.isoformat(),
        "timezone": local_start.tzname(),
        "frame_count": frame_count,
        "fps": fps,
    }
    # 先写入 in_progress 清单，崩溃时保留现场信息但不会进入全量索引。
    manifest: dict[str, Any] = {
        "schema_version": "roadside_multicam_v3",
        "sequence_id": sequence.sequence_id,
        "status": "in_progress",
        "town": town,
        "location": location,
        "capture_time": capture_time,
        "weather": weather,
        "rig_preset": rig_config.source.name,
        "rig_config": str(rig_config.source),
        "seed": seed,
        "fps": fps,
        "fixed_delta_seconds": fixed_delta,
        "sequence_duration_s": duration,
        "frame_count": frame_count,
        "requested_traffic": {
            "vehicles": vehicle_target,
            "walkers": walker_target,
        },
        "class_map_version": CLASS_MAP_VERSION,
        "classes": dict(CLASSES),
        "camera_ids": [camera.camera_id for camera in rig_config.cameras],
        "cameras": [
            {
                **camera.mount(),
                "camera_id": camera.camera_id,
                "width": camera.width,
                "height": camera.height,
                "fov": camera.fov,
                "dvs_params": camera.dvs_parameters(),
            }
            for camera in rig_config.cameras
        ],
        "weather_cycle": list(rig_config.weather_cycle),
    }
    layout.write_manifest(sequence, manifest)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if hasattr(traffic_manager, "set_random_device_seed"):
            traffic_manager.set_random_device_seed(seed)
        random.seed(seed)
        world.set_weather(getattr(carla.WeatherParameters, weather))

        traffic_started = time.perf_counter()
        traffic_counts, walker_plans = spawn_road_traffic(
            world,
            carla,
            traffic_manager,
            vehicle_target,
            walker_target,
            seed,
            rig_config.traffic,
            spawned_actors,
            junction_center=rig_location.get("junction_center_m"),
            # 出生点按相机视野排序：全图随机会让某些机位拍不到车。
            cameras=[
                {
                    "camera_id": camera.camera_id,
                    "location": list(camera.location),
                    "yaw": camera.yaw,
                }
                for camera in rig_config.cameras
            ],
        )
        # 交通流阶段的耗时单独打出来：行人寻路/车辆调度一旦变慢，
        # 这里会先于"传感器等不到帧"暴露出来。
        print(
            f"交通流阶段耗时 {time.perf_counter() - traffic_started:.1f} s",
            flush=True,
        )
        time_origin_us = round(
            world.get_snapshot().timestamp.elapsed_seconds * 1_000_000
        )
        print("正在创建传感器（DVS + RGB + 实例分割）…", flush=True)
        camera_rig = spawn_static_camera_rig(
            world,
            carla,
            h5py,
            np_module,
            rig_config.cameras,
            sequence,
            time_origin_us,
            integration_time_us,
            args.flush_frames,
            town,
            weather,
            instance_seg=args.instance_seg,
            instance_seg_scale=args.instance_seg_scale,
        )
        layout.write_dvs_params(
            sequence,
            rig_config.cameras,
            camera_rig.applied_exposure_modes,
        )

        # CARLA 新建 DVS 传感器可能漏掉首个同步回调，先预热再开始计数和落盘。
        world.tick()
        # 预热 tick 之后传感器位姿才真正生效，这时才能读到外参
        # （spawn_actor 返回时 CARLA 还没把生成指令下发到服务器）。
        capture_calibration(camera_rig, rig_config.cameras)
        print("传感器已就绪，开始采集帧", flush=True)
        event_counts, annotation_frames, written_frames, skipped_frames = _collect_frames(
            world,
            camera_rig,
            sequence,
            rig_config.cameras,
            frame_count,
            time_origin_us,
            integration_time_us,
            args.sensor_timeout,
            args.min_speed,
            np_module,
            carla,
            walker_plans=walker_plans,
            instance_seg=args.instance_seg,
            rgb_format=args.rgb_format,
            rgb_quality=args.rgb_quality,
            min_detect_px=args.min_detect_px,
            max_detect_distance=args.max_detect_distance,
        )
        for camera in rig_config.cameras:
            layout.write_annotations(
                sequence,
                camera,
                annotation_frames[camera.camera_id],
            )
        layout.write_calibration(
            sequence,
            rig_config.cameras,
            camera_rig.calibration,
            np_module,
        )
        # 验收项：每机位可见的行人轨迹数量与平均位移（方案 §6.1.5）。
        validation = {
            camera.camera_id: track_statistics(
                annotation_frames[camera.camera_id], integration_time_us
            )
            for camera in rig_config.cameras
        }
        # 类别映射表审计：靠兜底规则归类的蓝图要显式报出来（方案 §6.3）。
        seen_blueprints = _seen_blueprints(annotation_frames)
        fallback_blueprints = [
            blueprint_id
            for blueprint_id in seen_blueprints
            if classify_blueprint(blueprint_id)["by_fallback"]
        ]
        if fallback_blueprints:
            print(
                f"提示：{len(fallback_blueprints)} 个蓝图按兜底规则归为 car，"
                "请核对类别映射表：",
                ", ".join(fallback_blueprints[:10]),
                flush=True,
            )
            _record_unknown_blueprints(layout.root, set(fallback_blueprints))
        ended_at = datetime.now(timezone.utc)
        capture_time.update(
            ended_at_utc=ended_at.isoformat(),
            elapsed_wall_s=round((ended_at - started_at).total_seconds(), 3),
            simulation_start_s=round(time_origin_us / 1_000_000, 3),
            simulation_end_s=round(
                # 跳过的帧也消耗了仿真时间：仿真时间是 tick 数，不是落盘帧数。
                (
                    time_origin_us
                    + (written_frames + skipped_frames) * fixed_delta * 1_000_000
                )
                / 1_000_000,
                3,
            ),
        )
        manifest.update(
            status="completed",
            spawned_traffic=traffic_counts,
            event_counts=event_counts,
            # 实际落盘的帧数与因传感器缺数据被跳过的帧数：
            # 两者都要写进 manifest，事后才知道这条序列完不完整。
            frame_count=written_frames,
            skipped_frames=skipped_frames,
            # 每个机位有多少帧一个事件都没有：画面静止时 DVS 不发数据，
            # 空帧占比过高说明那个机位其实没拍到东西。
            dvs_empty_frames={
                camera_id: writer.empty_frame_count
                for camera_id, writer in camera_rig.writers.items()
            },
            timestamp_adjusted_event_counts={
                camera_id: writer.timestamp_adjusted_event_count
                for camera_id, writer in camera_rig.writers.items()
            },
            timestamp_origin_simulation_us=time_origin_us,
            dvs_exposure_mode_applied=camera_rig.applied_exposure_modes,
            instance_segmentation={
                "enabled": bool(args.instance_seg),
                "scale": args.instance_seg_scale,
            },
            annotation_validation=validation,
            blueprints_seen=seen_blueprints,
            blueprints_by_fallback=fallback_blueprints,
        )
    except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
        manifest.update(status="failed", error=str(exc))
        layout.write_manifest(sequence, manifest)
        raise
    finally:
        _cleanup(
            client,
            carla,
            world,
            traffic_manager,
            previous_settings,
            previous_weather,
            previous_tm_sync,
            camera_rig,
            spawned_actors,
        )

    layout.write_manifest(sequence, manifest)
    # 位置时间索引：每条序列一行，便于按路口或时间段筛选数据。
    center = location.get("junction_center_m") or [None, None]
    geo = location.get("geo") or {}
    layout.write_location_record(
        {
            "sequence_id": sequence.sequence_id,
            "capture_started_at_utc": capture_time["started_at_utc"],
            "capture_started_at_local": capture_time["started_at_local"],
            "timezone": capture_time["timezone"],
            "duration_s": round(duration, 3),
            "town": town,
            "map_name": game_map.name,
            "junction_id": location.get("junction_id"),
            "junction_center_x_m": center[0],
            "junction_center_y_m": center[1],
            "geo_latitude": geo.get("latitude"),
            "geo_longitude": geo.get("longitude"),
            "geo_altitude_m": geo.get("altitude"),
            "camera_count": len(rig_config.cameras),
            "camera_poses": _camera_poses(location),
            "frame_count": frame_count,
            "fps": fps,
            "weather": weather,
            "seed": seed,
            "total_events": sum(event_counts.values()),
            "vehicles": traffic_counts["vehicles"],
            "walkers": traffic_counts["walkers"],
        }
    )
    layout.register_sequence(sequence.sequence_id)
    return layout.root


def _install_stop_handler() -> None:
    """Windows 下把 CTRL_BREAK（网页控制台的“终止任务”）转成 KeyboardInterrupt，
    让 collect() 的 finally 有机会销毁 actor 并恢复 world 设置；强杀进程做不到。"""
    if sys.platform != "win32":
        return

    def _raise(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGBREAK, _raise)
    except (ValueError, OSError):
        pass


def main() -> int:
    # 控制台可能是 GBK 码页，遇到无法编码的字符用替代符而不是崩溃。
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    args = parse_args()
    _install_stop_handler()
    try:
        output = collect(args)
    except KeyboardInterrupt:
        print("收到中断信号，采集已停止（CARLA 状态已恢复）。", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
        print(f"采集失败：{exc}", file=sys.stderr)
        return 1
    print(f"采集完成，数据集目录：{output}")
    return 0
