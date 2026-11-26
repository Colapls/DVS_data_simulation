#!/usr/bin/env python3
"""在 CARLA 地图的每个信号灯杆上生成路侧固定机位组（DVS+RGB）。

扫描地图的全部路口，把每个路口的信号灯杆逐个变成一台相机（安装于杆上灯头
高度），按路口分组成机位组（rig），输出与 dvs_collector 完全同构的 YAML
—— 采集器无需任何改动。

用法：
    python tools/plan_rig.py --town Town03 --dry-run
    python tools/plan_rig.py --town Town03 --out-dir rigs/Town03
    python tools/plan_rig.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRAFFIC_LIGHT_LANDMARK_TYPE = "1000001"
# 灯杆上灯头包围盒与杆心的水平容差（米），用于取杆上灯头高度。
POLE_HEAD_RADIUS_M = 1.5
# 同一根杆上可能注册多个灯头 actor，按水平距离合并。
POLE_MERGE_RADIUS_M = 2.0
# 生成路口 waypoint 的间距（米）；路口判定依赖 waypoint 落在路口内。
WAYPOINT_SPACING_M = 2.0
MAX_CAMERAS_PER_RIG = 8
DEFAULT_FOV = 60.0
DEFAULT_PRESET = "road_day"
DEFAULT_DURATION_S = 90.0
DEFAULT_FPS = 20.0
DEFAULT_MOUNT_HEIGHT_M = 4.0
DEFAULT_TRAFFIC: dict[str, Any] = {
    "vehicles": 80,
    "walkers": 40,
    "ignore_lights_pct": [0, 30],
    "speed_difference_pct": [-10, 20],
    "following_distance_m": [2, 6],
}
DEFAULT_WEATHER_CYCLE = [
    "ClearNoon",
    "CloudyNoon",
    "SoftRainSunset",
    "WetSunset",
    "ClearNight",
]


@dataclass(frozen=True)
class Point:
    """世界坐标点。"""

    x: float
    y: float
    z: float = 0.0

    def as_list(self) -> list[float]:
        return [round(self.x, 3), round(self.y, 3), round(self.z, 3)]


@dataclass
class Junction:
    """一个路口及其服务的信号灯。"""

    junction_id: int
    center: Point
    lights: list[Any] = field(default_factory=list)


@dataclass
class Pole:
    """一根物理信号灯杆及其上规划的相机。"""

    pole_id: str
    junction_id: int
    location: Point
    mount: Point
    head_z: float
    target: Point
    yaw: float
    pitch: float
    aim_source: str
    near_road_m: float
    light_actor_ids: list[int] = field(default_factory=list)
    rig_file: str = ""
    camera_id: str = ""


# --------------------------------------------------------------------------
# 纯计算部分：不依赖 CARLA，便于 --self-test 验证。
# --------------------------------------------------------------------------


def yaw_pitch_to_target(mount: Point, target: Point) -> tuple[float, float]:
    """算出相机指向目标点所需的 yaw/pitch（CARLA 角度约定）。

    CARLA 中 yaw 0 指向 +X、90 指向 +Y，正角度向上旋转；pitch 为负表示俯视。
    """
    dx = target.x - mount.x
    dy = target.y - mount.y
    distance = math.hypot(dx, dy)
    yaw = math.degrees(math.atan2(dy, dx)) % 360.0
    if distance < 1e-3:
        pitch = -90.0
    else:
        pitch = -math.degrees(math.atan2(mount.z - target.z, distance))
    pitch = max(-90.0, min(0.0, pitch))
    return round(yaw, 3), round(pitch, 3)


def chunk(items: list[Any], size: int) -> list[list[Any]]:
    """按固定大小切分列表（一个机位组最多容纳 size 台相机）。"""
    if size < 1:
        raise ValueError("切分大小必须大于 0")
    return [items[index : index + size] for index in range(0, len(items), size)]


def merge_nearby(items: list[tuple[int, Point]], radius: float) -> list[list[tuple[int, Point]]]:
    """把水平距离小于 radius 的项合并成一组（同一根杆上的多个灯头）。"""
    groups: list[list[tuple[int, Point]]] = []
    for item in sorted(items, key=lambda value: (value[1].x, value[1].y, value[0])):
        for group in groups:
            anchor = group[0][1]
            distance = math.hypot(item[1].x - anchor.x, item[1].y - anchor.y)
            if distance <= radius:
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


# --------------------------------------------------------------------------
# CARLA 勘察。
# --------------------------------------------------------------------------


def connect(host: str, port: int, town: str | None) -> tuple[Any, Any]:
    """连接 CARLA；指定 town 时切换地图（会重载当前 world）。"""
    import carla

    client = carla.Client(host, port)
    # 切换地图需要重载关卡，首个 load_world 可能超过默认超时。
    client.set_timeout(180.0)
    world = client.load_world(town) if town else client.get_world()
    return carla, world


def prepare_world(world: Any, warmup_ticks: int) -> None:
    """让 CARLA 把地图的信号灯实例化出来。

    同步模式下 world.get_traffic_light() 一律返回 None，信号灯 actor 也是
    逐帧惰性创建的，所以这里要求异步模式并空跑若干帧。
    """
    if world.get_settings().synchronous_mode:
        raise RuntimeError(
            "当前 CARLA world 处于同步模式（可能有采集正在进行），"
            "信号灯 actor 不可用；请结束采集后再运行本工具。"
        )
    for _ in range(warmup_ticks):
        world.tick()


def junction_center(carla: Any, junction: Any) -> Point:
    """路口几何中心：路口内所有车道两端的均值。"""
    points: list[tuple[float, float]] = []
    for in_waypoint, out_waypoint in junction.get_waypoints(carla.LaneType.Driving):
        for location in (in_waypoint.transform.location, out_waypoint.transform.location):
            points.append((location.x, location.y))
    if not points:
        return Point(0.0, 0.0, 0.0)
    return Point(
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        0.0,
    )


def traffic_light_actors(world: Any) -> list[Any]:
    """枚举地图里全部信号灯 actor（同一根杆上的多个灯头共用一个 actor）。"""
    game_map = world.get_map()
    actors: dict[int, Any] = {}
    for landmark in game_map.get_all_landmarks_of_type(TRAFFIC_LIGHT_LANDMARK_TYPE):
        try:
            light = world.get_traffic_light(landmark)
        except RuntimeError:
            light = None
        if light is not None:
            actors.setdefault(int(light.id), light)
    if not actors:
        for light in world.get_actors().filter("traffic.traffic_light"):
            actors.setdefault(int(light.id), light)
    return [actors[key] for key in sorted(actors)]


def collect_junctions(carla: Any, world: Any) -> list[Junction]:
    """枚举地图中所有路口，并挂上服务该路口的信号灯。

    路口有两条发现路径：waypoint 扫描，以及信号灯所控车道携带的 junction_id
    （后者能补上 waypoint 扫描漏掉的路口）。
    """
    found: dict[int, Junction] = {}

    def register(junction_id: int, junction: Any) -> None:
        if junction_id in found or junction is None:
            return
        found[junction_id] = Junction(
            junction_id=junction_id,
            center=junction_center(carla, junction),
        )

    for waypoint in world.get_map().generate_waypoints(WAYPOINT_SPACING_M):
        if waypoint.is_junction:
            register(int(waypoint.junction_id), waypoint.get_junction())

    for light in traffic_light_actors(world):
        try:
            waypoints = list(light.get_affected_lane_waypoints())
        except RuntimeError:
            continue
        for waypoint in waypoints:
            if waypoint.junction_id is not None and waypoint.junction_id >= 0:
                register(int(waypoint.junction_id), waypoint.get_junction())
                break

    for junction in found.values():
        seen: set[int] = set()
        for light in world.get_traffic_lights_in_junction(junction.junction_id):
            if int(light.id) in seen:
                continue
            seen.add(int(light.id))
            junction.lights.append(light)
    return [found[key] for key in sorted(found)]


def pole_position(light: Any) -> Point:
    """信号灯 actor 的水平位置即灯杆位置（actor 的 z 只是虚拟锚点）。"""
    transform = light.get_transform()
    return Point(transform.location.x, transform.location.y, 0.0)


def pole_head_height(light: Any, fallback: float) -> float:
    """杆上灯头高度：取杆心附近灯头包围盒的最高点（杆顶信号灯约 4 m）。"""
    pole = pole_position(light)
    heights = [
        box.location.z
        for box in light.get_light_boxes()
        if math.hypot(
            box.location.x - pole.x,
            box.location.y - pole.y,
        )
        <= POLE_HEAD_RADIUS_M
    ]
    if not heights:
        return fallback
    return max(heights)


def affected_lane_center(light: Any) -> Point | None:
    """该信号灯所控制车道的中心（沿进口道看过去的目标点）。"""
    try:
        waypoints = list(light.get_affected_lane_waypoints())
    except RuntimeError:
        return None
    if not waypoints:
        return None
    locations = [waypoint.transform.location for waypoint in waypoints]
    return Point(
        sum(location.x for location in locations) / len(locations),
        sum(location.y for location in locations) / len(locations),
        0.0,
    )


def road_distance(carla: Any, world: Any, location: Point) -> float:
    """到最近车道中心线的水平距离，用于判断灯杆是否在车道正上方。"""
    try:
        waypoint = world.get_map().get_waypoint(
            carla.Location(x=location.x, y=location.y, z=location.z)
        )
    except RuntimeError:
        return float("inf")
    if waypoint is None:
        return float("inf")
    lane = waypoint.transform.location
    return math.hypot(location.x - lane.x, location.y - lane.y)


def build_poles(
    carla: Any,
    world: Any,
    junctions: list[Junction],
    aim: str,
    mount_height: float | None,
) -> list[Pole]:
    """把路口的信号灯杆转成相机位姿；无信号灯的路口用路侧虚拟杆位。"""
    # 同一个灯杆可能同时服务两个路口，只归属到路口中心更近的那一个。
    owner: dict[int, Junction] = {}
    for junction in junctions:
        for light in junction.lights:
            light_id = int(light.id)
            pole = pole_position(light)
            current = owner.get(light_id)
            if current is None:
                owner[light_id] = junction
                continue
            new_distance = math.hypot(
                pole.x - junction.center.x,
                pole.y - junction.center.y,
            )
            old_distance = math.hypot(
                pole.x - current.center.x,
                pole.y - current.center.y,
            )
            if new_distance < old_distance:
                owner[light_id] = junction

    by_junction: dict[int, list[tuple[int, Point]]] = {}
    for light_id, junction in owner.items():
        light = next(
            light
            for light in junction.lights
            if int(light.id) == light_id
        )
        by_junction.setdefault(junction.junction_id, []).append(
            (light_id, pole_position(light))
        )

    poles: list[Pole] = []
    index = 0
    for junction in junctions:
        items = by_junction.get(junction.junction_id, [])
        groups = merge_nearby(items, POLE_MERGE_RADIUS_M)
        if not groups:
            # 无信号灯的路口：相机改架在路口周边的路侧（路灯杆/护栏/建筑
            # 墙面），位置由地面 waypoint 推算。Town10HD_Opt 的 9 个路口里
            # 有 4 个没有信号灯，只按信号灯选点会让这些路口完全没有数据。
            for group in virtual_pole_groups(carla, world, junction):
                index += 1
                mount = Point(
                    group[0].x,
                    group[0].y,
                    mount_height
                    if mount_height is not None
                    else DEFAULT_MOUNT_HEIGHT_M,
                )
                target = junction.center
                yaw, pitch = yaw_pitch_to_target(mount, target)
                poles.append(
                    Pole(
                        pole_id=f"pole_{index:03d}",
                        junction_id=junction.junction_id,
                        location=group[0],
                        mount=mount,
                        head_z=round(mount.z, 3),
                        target=target,
                        yaw=yaw,
                        pitch=pitch,
                        aim_source="junction_center",
                        near_road_m=round(road_distance(carla, world, mount), 2),
                        light_actor_ids=[],
                    )
                )
            continue
        for group in groups:
            index += 1
            light_ids = [light_id for light_id, _ in group]
            anchor = group[0][1]
            lights = [
                light
                for light in junction.lights
                if int(light.id) in light_ids
            ]
            head_z = max(pole_head_height(light, DEFAULT_MOUNT_HEIGHT_M) for light in lights)
            mount = Point(
                anchor.x,
                anchor.y,
                mount_height if mount_height is not None else head_z,
            )
            if aim == "approach":
                centers = [
                    center
                    for center in (affected_lane_center(light) for light in lights)
                    if center
                ]
                if centers:
                    target = Point(
                        sum(center.x for center in centers) / len(centers),
                        sum(center.y for center in centers) / len(centers),
                        0.0,
                    )
                    source = "affected_lanes"
                else:
                    target = junction.center
                    source = "junction_center"
            else:
                target = junction.center
                source = "junction_center"
            yaw, pitch = yaw_pitch_to_target(mount, target)
            poles.append(
                Pole(
                    pole_id=f"pole_{index:03d}",
                    junction_id=junction.junction_id,
                    location=anchor,
                    mount=mount,
                    head_z=round(head_z, 3),
                    target=target,
                    yaw=yaw,
                    pitch=pitch,
                    aim_source=source,
                    near_road_m=round(road_distance(carla, world, mount), 2),
                    light_actor_ids=light_ids,
                )
            )
    return poles


def virtual_pole_groups(
    carla: Any,
    world: Any,
    junction: Junction,
    radius_range: tuple[float, float] = (14.0, 26.0),
    buckets: int = 6,
) -> list[list[Point]]:
    """无信号灯路口的虚拟杆位：按方位角分桶，各取一个路侧地面点。

    路侧机位并不需要真的有信号灯杆——架在路灯杆、护栏或建筑外墙上都行。
    这里从路口周边的可走地面 waypoint 里，按方位角均匀取点，保证多台相机
    的视野互不重叠，且都朝向路口中心。
    """
    game_map = world.get_map()
    center = junction.center
    candidates: list[tuple[float, float, Point]] = []
    for waypoint in game_map.generate_waypoints(2.0):
        location = waypoint.transform.location
        distance = math.hypot(
            location.x - center.x, location.y - center.y
        )
        if not radius_range[0] <= distance <= radius_range[1]:
            continue
        bearing = math.degrees(
            math.atan2(location.y - center.y, location.x - center.x)
        ) % 360.0
        candidates.append((distance, bearing, Point(location.x, location.y, 0.0)))
    if not candidates:
        return []
    groups: list[list[Point]] = []
    for bucket in range(buckets):
        low = bucket * 360.0 / buckets
        high = low + 360.0 / buckets
        in_bucket = [
            item
            for item in candidates
            if low <= item[1] < high
        ]
        if not in_bucket:
            continue
        # 取桶内离路口最近的那个（相机离路口太远就拍不到什么）
        distance, _bearing, point = min(in_bucket, key=lambda item: item[0])
        groups.append([point])
    return groups


# --------------------------------------------------------------------------
# 输出。
# --------------------------------------------------------------------------


def group_by_junction(poles: list[Pole]) -> list[tuple[int, list[Pole]]]:
    """按路口归组杆位（路口编号升序）。"""
    groups: dict[int, list[Pole]] = {}
    for pole in poles:
        groups.setdefault(pole.junction_id, []).append(pole)
    return [
        (junction_id, sorted(groups[junction_id], key=lambda pole: pole.pole_id))
        for junction_id in sorted(groups)
    ]


def junction_location(
    carla: Any,
    world: Any,
    junction: Junction,
    aim: str,
) -> dict[str, Any]:
    """机位组的采集位置：路口编号、路口中心和地理坐标。"""
    game_map = world.get_map()
    center = junction.center
    location: dict[str, Any] = {
        "map_name": game_map.name.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1],
        "junction_id": junction.junction_id,
        "junction_center_m": [
            round(center.x, 3),
            round(center.y, 3),
            round(center.z, 3),
        ],
        "aim": aim,
    }
    try:
        geolocation = game_map.transform_to_geolocation(
            carla.Location(x=center.x, y=center.y, z=0.0)
        )
        location["geo"] = {
            "latitude": round(float(geolocation.latitude), 7),
            "longitude": round(float(geolocation.longitude), 7),
            "altitude": round(float(geolocation.altitude), 3),
        }
    except (AttributeError, RuntimeError):
        location["geo"] = None
    return location


def rig_payload(
    town: str,
    cameras: dict[str, dict[str, Any]],
    duration_s: float,
    fps: float,
    location: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造与 dvs_collector 同构的机位组内容。"""
    payload: dict[str, Any] = {
        "town": town,
        "sequence_duration_s": duration_s,
        "fps": fps,
        "seed": None,
        "cameras": cameras,
        "traffic": dict(DEFAULT_TRAFFIC),
        "weather_cycle": list(DEFAULT_WEATHER_CYCLE),
    }
    if location:
        payload["location"] = location
    return payload


def camera_payload(pole: Pole, fov: float, preset: str) -> dict[str, Any]:
    return {
        "location": pole.mount.as_list(),
        "pitch": pole.pitch,
        "yaw": pole.yaw,
        "fov": fov,
        "preset": preset,
    }


def write_rig(
    path: Path,
    town: str,
    cameras: dict[str, dict[str, Any]],
    duration_s: float,
    fps: float,
    location: dict[str, Any] | None = None,
) -> None:
    import yaml

    path.write_text(
        yaml.safe_dump(
            rig_payload(town, cameras, duration_s, fps, location),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def write_report(
    out_dir: Path,
    town: str,
    poles: list[Pole],
    rigs: list[dict[str, Any]],
    settings: dict[str, Any],
    stem: str = "poles",
) -> None:
    """写出杆位清单（JSON + Markdown）和逐机位组的采集命令，便于人工复核。

    ``stem`` 让补充规划（--only-unlit）写到自己的文件里，不覆盖既有的
    poles.json / poles.md。
    """
    payload = {
        "town": town,
        "settings": settings,
        "pole_count": len(poles),
        "rig_count": len(rigs),
        "rigs": [
            {key: value for key, value in rig.items() if key != "poles"}
            for rig in rigs
        ],
        "poles": [
            {
                "pole_id": pole.pole_id,
                "rig_file": pole.rig_file,
                "camera_id": pole.camera_id,
                "junction_id": pole.junction_id,
                "pole_xy_m": [round(pole.location.x, 3), round(pole.location.y, 3)],
                "mount_m": pole.mount.as_list(),
                "target_m": pole.target.as_list(),
                "yaw": pole.yaw,
                "pitch": pole.pitch,
                "aim_source": pole.aim_source,
                "near_road_m": pole.near_road_m,
                "over_roadway": pole.near_road_m < 6.0,
                "light_actor_ids": pole.light_actor_ids,
            }
            for pole in poles
        ],
    }
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"# {town} 信号灯杆机位清单",
        "",
        f"杆位 {len(poles)} 个，机位组 {len(rigs)} 个。",
        "",
        "| 机位组 | camera_id | 杆位 (x, y) m | 安装高度 m | yaw | pitch | 目标来源 | 距车道 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pole in poles:
        lines.append(
            "| {rig} | {cam} | {xy} | {z} | {yaw} | {pitch} | {source} | {road} |".format(
                rig=pole.rig_file,
                cam=pole.camera_id,
                xy=f"{pole.location.x:.1f}, {pole.location.y:.1f}",
                z=pole.mount.z,
                yaw=pole.yaw,
                pitch=pole.pitch,
                source=pole.aim_source,
                road=f"{pole.near_road_m:.1f} m",
            )
        )
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 默认沿用既有的 commands.txt；补充规划写到自己的文件，不覆盖。
    command_name = "commands.txt" if stem == "poles" else f"{stem}_commands.txt"
    (out_dir / command_name).write_text(
        "\n".join(
            f'python .\\collect_dvs.py --rig {_display_path(out_dir / rig["file"])} '
            f"--load-town --output .\\dvs_roadside_multicam --frames 200"
            for rig in rigs
        )
        + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    """优先给出相对当前目录的路径，便于直接复制命令。"""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def validate_rig(path: Path) -> None:
    """用采集器自身的配置解析校验生成结果，避免字段漂移。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dvs_collector.config import load_rig

    load_rig(path)


# --------------------------------------------------------------------------
# 命令行。
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按信号灯杆生成 CARLA 路侧固定 DVS+RGB 机位组 YAML。"
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA 服务端地址")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC 端口")
    parser.add_argument("--town", help="目标地图；不给则使用当前已加载的地图")
    parser.add_argument("--out-dir", type=Path, help="输出目录，默认 rigs/<town>")
    parser.add_argument(
        "--mount-height",
        type=float,
        help="安装高度（m）；不给则使用杆上灯头实际高度",
    )
    parser.add_argument(
        "--aim",
        choices=("junction", "approach"),
        default="junction",
        help="瞄准目标：路口中心 / 该灯控制的进口道",
    )
    parser.add_argument(
        "--warmup-ticks",
        type=int,
        default=10,
        help="空跑帧数，让 CARLA 实例化地图信号灯（同步模式下不可用）",
    )
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV, help="相机视场角")
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="DVS 参数预设")
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=MAX_CAMERAS_PER_RIG,
        help="单个机位组最多容纳的相机数",
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="序列时长")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="采集帧率")
    parser.add_argument("--dry-run", action="store_true", help="只打印清单，不写文件")
    parser.add_argument(
        "--include-unlit",
        action="store_true",
        help="连同无信号灯的路口一起规划（相机改架在路侧虚拟杆位）",
    )
    parser.add_argument(
        "--only-unlit",
        action="store_true",
        help="只规划无信号灯的路口，不动既有机位组",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只跑朝向与切分逻辑的自检，不连接 CARLA",
    )
    return parser.parse_args(argv)


def self_test() -> int:
    """不依赖 CARLA 的自检：朝向数学、切分逻辑、杆位合并、机位组格式。"""
    mount = Point(0.0, 0.0, 6.0)

    yaw, pitch = yaw_pitch_to_target(mount, Point(10.0, 0.0, 0.0))
    assert yaw == 0.0, yaw
    assert abs(pitch - (-math.degrees(math.atan2(6.0, 10.0)))) < 1e-3, pitch

    yaw, _ = yaw_pitch_to_target(mount, Point(0.0, 10.0, 0.0))
    assert yaw == 90.0, yaw

    yaw, _ = yaw_pitch_to_target(mount, Point(-10.0, 0.0, 0.0))
    assert yaw == 180.0, yaw

    yaw, _ = yaw_pitch_to_target(mount, Point(0.0, -10.0, 0.0))
    assert yaw == 270.0, yaw

    # 目标在相机正下方时俯角为 -90，不会产生 NaN。
    _, pitch = yaw_pitch_to_target(mount, Point(0.0, 0.0, 0.0))
    assert pitch == -90.0, pitch

    assert chunk(list(range(10)), 8) == [list(range(8)), [8, 9]]
    assert chunk([], 8) == []

    items = [(1, Point(1.0, 2.0)), (2, Point(1.5, 2.0)), (3, Point(20.0, 2.0))]
    groups = merge_nearby(items, 2.0)
    assert len(groups) == 2, groups
    assert sorted(len(group) for group in groups) == [1, 2], groups

    # 生成的 YAML 必须能被采集器自身的配置解析接受（含单机位机位组）。
    import tempfile

    pole = Pole(
        pole_id="pole_001",
        junction_id=3,
        location=Point(1.0, 2.0),
        mount=Point(1.0, 2.0, 4.05),
        head_z=4.05,
        target=Point(10.0, 2.0, 0.0),
        yaw=0.0,
        pitch=-30.0,
        aim_source="junction_center",
        near_road_m=8.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rig.yaml"
        write_rig(
            path,
            "Town03",
            {"cam_01": camera_payload(pole, DEFAULT_FOV, DEFAULT_PRESET)},
            DEFAULT_DURATION_S,
            DEFAULT_FPS,
        )
        validate_rig(path)

    print("self-test 通过：朝向数学、切分、杆位合并、机位组格式均正常")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.max_cameras < 1:
        print("--max-cameras 必须大于 0", file=sys.stderr)
        return 1

    carla, world = connect(args.host, args.port, args.town)
    prepare_world(world, args.warmup_ticks)
    town = world.get_map().name.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    out_dir = args.out_dir or Path("rigs") / town
    out_dir = out_dir.expanduser().resolve()

    junctions = collect_junctions(carla, world)
    with_lights = [junction for junction in junctions if junction.lights]
    if not junctions:
        print(
            f"{town} 中没有找到路口：确认地图已加载完整，且 CARLA 处于异步模式。",
            file=sys.stderr,
        )
        return 1
    # 默认只处理有信号灯的路口（保持既有行为）；--include-unlit 把无信号灯
    # 的路口也规划成路侧机位，--only-unlit 则只处理无信号灯的路口，
    # 便于在不动既有机位组的前提下补齐。
    if args.only_unlit:
        target_junctions = [j for j in junctions if not j.lights]
    elif args.include_unlit:
        target_junctions = junctions
    else:
        target_junctions = with_lights
    if not target_junctions:
        print(
            f"{town} 中没有符合条件的路口"
            + ("（无信号灯路口为空）" if args.only_unlit else ""),
            file=sys.stderr,
        )
        return 1
    poles = build_poles(carla, world, target_junctions, args.aim, args.mount_height)
    total_lights = len(traffic_light_actors(world))
    assigned_lights = len({actor_id for pole in poles for actor_id in pole.light_actor_ids})

    rigs: list[dict[str, Any]] = []
    for junction_id, group_poles in group_by_junction(poles):
        parts = chunk(group_poles, args.max_cameras)
        for part, part_poles in enumerate(parts, start=1):
            suffix = "" if len(parts) == 1 else f"_p{part}"
            file_name = f"{town}_j{junction_id:03d}{suffix}.yaml"
            for index, pole in enumerate(part_poles, start=1):
                pole.rig_file = file_name
                pole.camera_id = f"cam_{index:02d}"
            rigs.append(
                {
                    "file": file_name,
                    "junction_id": junction_id,
                    "cameras": [
                        {"camera_id": pole.camera_id, "pole_id": pole.pole_id}
                        for pole in part_poles
                    ],
                    "poles": part_poles,
                }
            )

    print(
        f"{town}：{len(junctions)} 个路口，{len(with_lights)} 个带信号灯 → "
        f"{len(poles)} 根杆位 → {len(rigs)} 个机位组"
    )
    unlit = [junction for junction in junctions if not junction.lights]
    if unlit and not (args.only_unlit or args.include_unlit):
        print(
            f"  提示：{len(unlit)} 个无信号灯的路口被跳过"
            f"（{', '.join(str(j.junction_id) for j in unlit)}）；"
            "加 --include-unlit 可为它们规划路侧机位"
        )
    print(
        f"  地图信号灯 actor {total_lights} 个，已归属路口 {assigned_lights} 个"
        f"（未归属 {total_lights - assigned_lights} 个）"
    )
    for rig in rigs:
        print(
            f"  {rig['file']}：{len(rig['cameras'])} 台相机"
            f"（路口 j{rig['junction_id']}）"
        )
    if args.dry_run:
        print("dry-run：未写文件")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for rig in rigs:
        cameras = {
            pole.camera_id: camera_payload(pole, args.fov, args.preset)
            for pole in rig["poles"]
        }
        junction = next(
            item for item in target_junctions if item.junction_id == rig["junction_id"]
        )
        path = out_dir / rig["file"]
        write_rig(
            path,
            town,
            cameras,
            args.duration,
            args.fps,
            junction_location(carla, world, junction, args.aim),
        )
        validate_rig(path)
    write_report(
        out_dir,
        town,
        poles,
        rigs,
        {
            "aim": args.aim,
            "fov": args.fov,
            "preset": args.preset,
            "mount_height_m": args.mount_height,
            "max_cameras_per_rig": args.max_cameras,
        },
        stem="poles_unlit" if args.only_unlit else "poles",
    )
    print(f"机位组已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
