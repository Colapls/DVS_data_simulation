"""为固定路侧机位生成背景交通。

按《DVS 自动标注方案（精简可执行版）》§6 修正了两处会毁掉数据的地方：

- §6.2 车辆按蓝图 ID 显式分成 car / motorcycle 两组，按比例抽取；
  自行车蓝图从生成列表里剔除，不制造 ignore 污染。
- §6.1 行人在路口周边的人行道上出生，朝 15–40 m 外的同侧人行道行走，
  另有一批横穿人行横道；生成后空跑 60 tick，位移过小的行人销毁重生。
- 行人不用 `controller.ai.walker`（要走导航网格，Town03 上会爆栈），
  改用 `carla.WalkerControl` 直接推着走，见 `_spawn_walker` 的说明。

本采集器不创建自车。
"""

from __future__ import annotations

import math
import random
from typing import Any

from .classes import classify_blueprint


#: 每个地图的 waypoint 缓存：同一条序列里反复取路点没有必要重新生成。
_WAYPOINT_CACHE: dict[str, list[Any]] = {}
#: 每个地图里的人行道 waypoint 缓存（是 _WAYPOINT_CACHE 的子集）。
_SIDEWALK_CACHE: dict[str, list[Any]] = {}


def _all_waypoints(world: Any, carla: Any) -> list[Any]:
    """地图的全部 waypoint（按地图名缓存）。"""
    game_map = world.get_map()
    key = game_map.name
    cached = _WAYPOINT_CACHE.get(key)
    if cached is None:
        cached = game_map.generate_waypoints(2.0)
        _WAYPOINT_CACHE[key] = cached
    return cached


def _sidewalk_waypoints(world: Any, carla: Any) -> list[Any]:
    """地图里的人行道 waypoint（按地图名缓存）。"""
    key = world.get_map().name
    cached = _SIDEWALK_CACHE.get(key)
    if cached is None:
        cached = [
            waypoint
            for waypoint in _all_waypoints(world, carla)
            if bool(waypoint.lane_type & carla.LaneType.Sidewalk)
        ]
        _SIDEWALK_CACHE[key] = cached
    return cached


def _distance(first: Any, second: Any) -> float:
    """两个 CARLA 位置之间的平面距离。"""
    return math.sqrt(
        (first.x - second.x) ** 2 + (first.y - second.y) ** 2
    )


def _spawn_point_xy(point: Any) -> tuple[float, float]:
    """出生点的平面坐标（出生点是 Transform，位置在 .location 上）。"""
    location = getattr(point, "location", point)
    return float(location.x), float(location.y)


def _normalize_yaw(degrees: float) -> float:
    """把 CARLA 累积的航向角归一化到 (-180, 180]。"""
    return (float(degrees) + 180.0) % 360.0 - 180.0


def _road_spawn_transforms(
    world: Any,
    carla: Any,
    camera: dict[str, Any],
    min_distance_m: float,
    max_distance_m: float,
    step_m: float = 6.0,
    min_separation_m: float = 8.0,
) -> list[Any]:
    """沿相机朝向，在其视野内的可行驶车道上生成 Transform。

    地图自带出生点只标在路口少数车道上，Town10HD_Opt 里 cam_03 的方位角内
    只有 2 个出生点，光靠出生点无法让该机位拍到足够的交通。这里沿相机
    朝向以固定步长采样路点，取其中可行驶的车道；点之间按最小间距过滤，
    避免车辆互相重叠（注意不能按车道去重：路口附近往往只有一两条车道，
    按车道去重会让某些机位只剩一两个点）。
    """
    game_map = world.get_map()
    location = camera["location"]
    yaw = camera["yaw"]
    transforms: list[Any] = []
    kept: list[tuple[float, float]] = []
    steps = int((max_distance_m - min_distance_m) / step_m) + 1
    for index in range(steps):
        distance = min_distance_m + index * step_m
        x = location[0] + distance * math.cos(math.radians(yaw))
        y = location[1] + distance * math.sin(math.radians(yaw))
        waypoint = game_map.get_waypoint(carla.Location(x=x, y=y, z=0.5))
        if waypoint.lane_type != carla.LaneType.Driving:
            continue
        transform = waypoint.transform
        point = (float(transform.location.x), float(transform.location.y))
        if any(
            math.hypot(point[0] - other[0], point[1] - other[1])
            < min_separation_m
            for other in kept
        ):
            continue
        kept.append(point)
        transforms.append(
            carla.Transform(
                carla.Location(
                    x=point[0],
                    y=point[1],
                    z=float(transform.location.z),
                ),
                carla.Rotation(
                    pitch=0.0,
                    yaw=_normalize_yaw(transform.rotation.yaw),
                    roll=0.0,
                ),
            )
        )
    return transforms


def _in_camera_fov(
    point: Any,
    camera: dict[str, Any],
    half_fov_deg: float,
    min_distance_m: float,
    max_distance_m: float,
) -> bool:
    """出生点是否落在某台相机的视野范围内。

    机位是固定的，"看得见"取决于方位角和距离两个条件：方位角要落在
    水平 FOV 内，距离要落在俯角对应的地面覆盖范围内（相机高 4 m、
    俯角 14°~17° 时大约是 14~39 m）。
    """
    px, py = _spawn_point_xy(point)
    dx = px - camera["location"][0]
    dy = py - camera["location"][1]
    distance = math.hypot(dx, dy)
    if not min_distance_m <= distance <= max_distance_m:
        return False
    bearing = math.degrees(math.atan2(dy, dx)) % 360.0
    offset = (bearing - camera["yaw"] + 180.0) % 360.0 - 180.0
    return abs(offset) <= half_fov_deg


def _prioritize_spawn_points(
    spawn_points: list[Any],
    cameras: list[dict[str, Any]],
    rng: random.Random,
    near_radius_m: float,
    far_radius_m: float,
) -> list[Any]:
    """重排出生点：落在任一相机视野内的排前面，其余保持随机。

    全图随机取点时，路口附近的出生点很少会被抽到——Town10HD_Opt 全图
    155 个出生点里，只有二十来个在路口 60 m 内，靠随机抽签会让某些机位
    几乎看不到车（实测 cam_03 方位角内只有 2 个出生点）。这里把视野内的
    出生点提到最前面，保证每台相机都能拍到交通。
    """
    if not cameras:
        return spawn_points

    def camera_distance(point: Any) -> float:
        px, py = _spawn_point_xy(point)
        return min(
            math.hypot(px - camera["location"][0], py - camera["location"][1])
            for camera in cameras
        )

    prioritized = [
        point
        for point in spawn_points
        if any(
            _in_camera_fov(point, camera, 30.0, 8.0, far_radius_m)
            for camera in cameras
        )
    ]
    # Transform 不便做集合成员判断（CARLA 的 __eq__ 未按预期工作），用下标区分。
    visible_ids = {id(point) for point in prioritized}
    rest = [
        point for index, point in enumerate(spawn_points)
        if id(point) not in visible_ids
    ]
    # 视野内的按"离相机近"排序，让近处目标先入场；其余保持随机。
    prioritized.sort(key=camera_distance)
    rng.shuffle(rest)
    return prioritized + rest


def _camera_road_transforms(
    world: Any,
    carla: Any,
    cameras: list[dict[str, Any]],
    min_distance_m: float,
    max_distance_m: float,
) -> list[Any]:
    """收集各相机视野内道路上的 Transform。

    三台相机看的是同一个路口，靠近路口的位置会被多台相机同时命中。
    这里按相机轮流取点（而不是先到先得），否则排在后面的相机可能
    一个点都轮不到——实测 cam_03 因此完全拍不到交通。
    """
    per_camera = [
        _road_spawn_transforms(world, carla, camera, min_distance_m, max_distance_m)
        for camera in cameras
    ]
    transforms: list[Any] = []
    seen: set[tuple[float, float]] = set()
    for index in range(max((len(items) for items in per_camera), default=0)):
        for items in per_camera:
            if index >= len(items):
                continue
            transform = items[index]
            key = (round(transform.location.x, 1), round(transform.location.y, 1))
            if key in seen:
                continue
            seen.add(key)
            transforms.append(transform)
    return transforms


def _spawn_vehicle(
    world: Any,
    carla: Any,
    traffic_manager: Any,
    blueprint: Any,
    transform: Any,
    rng: random.Random,
    traffic_config: dict[str, Any],
    spawned_actors: list[Any],
) -> Any:
    """按蓝图生成一辆车并交给 Traffic Manager 驾驶。"""
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        if colors:
            blueprint.set_attribute("color", rng.choice(colors))
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "autopilot")
    vehicle = world.try_spawn_actor(blueprint, transform)
    if vehicle is None:
        return None
    spawned_actors.append(vehicle)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    vehicle_min, vehicle_max = traffic_config["speed_difference_pct"]
    light_min, light_max = traffic_config["ignore_lights_pct"]
    distance_min, distance_max = traffic_config["following_distance_m"]
    traffic_manager.vehicle_percentage_speed_difference(
        vehicle, rng.uniform(vehicle_min, vehicle_max)
    )
    traffic_manager.ignore_lights_percentage(
        vehicle, rng.uniform(light_min, light_max)
    )
    traffic_manager.distance_to_leading_vehicle(
        vehicle, rng.uniform(distance_min, distance_max)
    )
    return vehicle


def _ground_z(world: Any, carla: Any, location: Any) -> float:
    """取某个平面位置处的路面高度。

    人行横道的顶点 z 基本都是 0（CARLA 把横道区域存成 z=0 的多边形），
    而真正的路面在 z=0.1 上下。差这十几厘米，行人的胶囊体会陷进路面，
    try_spawn_actor 直接返回 None——实测 40 个横道顶点有 36 个生成失败。
    地图的 waypoint 带的是路面真实高度，拿它来对齐。
    """
    try:
        waypoint = world.get_map().get_waypoint(
            carla.Location(x=location.x, y=location.y, z=location.z)
        )
        return float(waypoint.transform.location.z)
    except (RuntimeError, AttributeError, IndexError):
        return float(location.z)


def _spawn_walker(
    world: Any,
    carla: Any,
    walker_blueprint: Any,
    start: Any,
    goal: Any,
    rng: random.Random,
    spawned_actors: list[Any],
) -> dict[str, Any] | None:
    """在 start 处生成行人，返回它的行走计划。

    **不使用 controller.ai.walker**：那条路径要走导航网格
    （Nav/TownXX.bin），Town03 的导航数据不完整，服务端的路径搜索会一路
    递归到爆栈（exit 0xC00000FD），把整个采集带崩。改为直接用
    WalkerControl 推人走，方向自己算，彻底不碰导航网格。
    """
    # 行人的速度上限来自蓝图属性 speed，默认值偏低会让 WalkerControl
    # 里的速度设不上去（行人走得很慢，60 tick 也挪不动 0.5 m）。
    try:
        if walker_blueprint.has_attribute("speed"):
            walker_blueprint.set_attribute("speed", "2.0")
    except RuntimeError:
        pass
    ground_z = _ground_z(world, carla, start)
    walker = None
    spawn_at = start
    # 出生点抖动：同一条横道上要站多个行人，完全重合的话
    # try_spawn_actor 会因为位置被占用直接失败。抖动主要沿过街方向
    # （start→goal），横向只抖一点点，免得把行人抖到镜头底下。
    span = _distance(start, goal)
    if span > 1e-3:
        ux = (goal.x - start.x) / span
        uy = (goal.y - start.y) / span
    else:
        ux, uy = 1.0, 0.0
    # 抖动幅度逐次放大：同一个位置被静态碰撞体占住时（横道顶点经常
    # 落在路灯/护栏/绿化带上，CARLA 报 "Spawn failed because of
    # collision at spawn position"），小幅抖动还在碰撞体里，要往旁边挪。
    # 同时把落地高度逐级抬高：waypoint 给的 z 与实际碰撞面常有 5~20 cm
    # 偏差（Town10HD_Opt 实测 z=0 时 12/12 全部生成失败，z+0.05 起才有
    # 10/12 成功），行人陷进地面会被判成碰撞。Town03 偏差在容差内，
    # 第一次就成功，不会多花代价。
    height_offsets = (0.0, 0.05, 0.12, 0.25, 0.45)
    for attempt in range(8):
        height = height_offsets[min(attempt, len(height_offsets) - 1)]
        spread = 0.35 if attempt < 3 else 1.2
        along = rng.uniform(-spread, spread) * span
        across = rng.uniform(-1.0, 1.0) if attempt < 3 else rng.uniform(-4.0, 4.0)
        spawn_at = carla.Location(
            x=start.x + along * ux - across * uy,
            y=start.y + along * uy + across * ux,
            z=ground_z + height,
        )
        walker = world.try_spawn_actor(
            walker_blueprint, carla.Transform(spawn_at)
        )
        if walker is not None:
            break
    if walker is None:
        return None
    spawned_actors.append(walker)
    return {
        "actor": walker,
        "start": spawn_at,
        "goal": goal,
        "speed": rng.uniform(0.8, 1.8),
        "to_goal": True,
    }


def update_walkers(carla: Any, walkers: list[dict[str, Any]]) -> None:
    """每个 tick 前调一次：朝当前目标走，到了就掉头。

    没有导航网格可用，行人只能沿直线往返于出生点和目标点之间；
    目标点都选在同一条人行道或同一个人行横道的两侧，路线短且贴着路。

    **方向没变就不要重复下发控制**：apply_control 会让服务端重做一次导航
    网格寻路，40 个行人 × 40 tick/s 就是每秒近千次寻路，服务端帧时间被
    拖到十几秒，传感器等不到数据直接报"等待 … 超时"。行人沿直线走时
    方向是恒定的，只在掉头或偏了 10° 以上时补一次控制即可。
    """
    for plan in walkers:
        actor = plan["actor"]
        try:
            current = actor.get_location()
        except RuntimeError:
            continue
        # 客户端在 actor 生成后需要一次 tick 才有真实位姿，此前
        # get_location() 返回 (0,0,0)。此时算出的方向指向原点，会让
        # 行人朝地图原点乱走，先跳过，等位姿同步了再下发控制。
        if current.x == 0.0 and current.y == 0.0 and current.z == 0.0:
            continue
        target = plan["goal"] if plan["to_goal"] else plan["start"]
        dx = target.x - current.x
        dy = target.y - current.y
        distance = math.hypot(dx, dy)
        if distance < 1.5:
            plan["to_goal"] = not plan["to_goal"]
            target = plan["goal"] if plan["to_goal"] else plan["start"]
            dx = target.x - current.x
            dy = target.y - current.y
            distance = math.hypot(dx, dy)
        if distance < 1e-3:
            continue
        ux, uy = dx / distance, dy / distance
        last = plan.get("last_direction")
        if last is not None and last[0] * ux + last[1] * uy > 0.985:
            # 方向基本没变，不重复下发（见上面的说明）。
            continue
        try:
            actor.apply_control(
                carla.WalkerControl(
                    direction=carla.Vector3D(ux, uy, 0.0),
                    speed=plan["speed"],
                )
            )
        except RuntimeError:
            continue
        plan["last_direction"] = (ux, uy)


def _sidewalk_pairs(
    world: Any,
    carla: Any,
    rng: random.Random,
    center: tuple[float, float],
    radius_range: list[float],
    walk_range: list[float],
    count: int,
) -> list[tuple[Any, Any]]:
    """挑出 (出生点, 目标点) 组合：同侧人行道、相距 15–40 m。

    只取 ``count`` 组：候选人行道可能有几千个，全量两两配对是 O(n²)，
    在密集地图上会把采集卡死。
    """
    waypoints = _sidewalk_waypoints(world, carla)
    print(f"  人行道 waypoint 共 {len(waypoints)} 个", flush=True)
    if not waypoints:
        return []
    origin = carla.Location(x=center[0], y=center[1])
    near = [
        waypoint
        for waypoint in waypoints
        if radius_range[0]
        <= _distance(waypoint.transform.location, origin)
        <= radius_range[1]
    ]
    if not near:
        # 半径内没有人行道：取离中心最近的一段，行人仍然在相机附近。
        near = sorted(
            waypoints,
            key=lambda waypoint: _distance(waypoint.transform.location, origin),
        )[: max(count * 4, 40)]
    targets: list[tuple[Any, Any]] = []
    for _ in range(min(count, len(near))):
        start = rng.choice(near)
        start_location = start.transform.location
        same_side = [
            waypoint
            for waypoint in near
            if waypoint.road_id == start.road_id
            and walk_range[0]
            <= _distance(waypoint.transform.location, start_location)
            <= walk_range[1]
        ]
        candidates = same_side or [
            waypoint
            for waypoint in near
            if walk_range[0]
            <= _distance(waypoint.transform.location, start_location)
            <= walk_range[1]
        ]
        if not candidates:
            continue
        targets.append((start_location, rng.choice(candidates).transform.location))
    return targets


def _roadside_pairs(
    world: Any,
    carla: Any,
    rng: random.Random,
    center: tuple[float, float],
    radius_range: list[float],
    walk_range: list[float],
    count: int,
    cameras: list[dict[str, Any]] | None = None,
    min_camera_distance_m: float = 8.0,
) -> list[tuple[Any, Any]]:
    """沿路肩行走的 (出生点, 目标点)。

    人行横道稀疏的地图拿不到足够的过街通道：Town10HD_Opt 全图只有
    16 条人行横道（Town03 有 73 条），路口周边 60 m 内仅 4 条，其中
    3 条离相机不到 10 m 被滤掉，剩下一条正好落在有静态碰撞体的位置，
    try_spawn_actor 报 "Spawn failed because of collision at spawn
    position"，整条序列一个行人都没有。

    行人其实不必走人行横道——沿路侧在人行道上走同样自然。这里从地图
    的可走地面 waypoint 里挑点，横向偏移到车道边缘（车道宽的一半再加
    一点人行道宽度），沿路方向走 15~40 m。
    """
    origin = carla.Location(x=center[0], y=center[1])
    walkable = (
        getattr(carla.LaneType, "Driving", 1)
        | getattr(carla.LaneType, "Sidewalk", 2)
        | getattr(carla.LaneType, "Shoulder", 8)
        | getattr(carla.LaneType, "Biking", 4)
    )
    # 按 road_id 分组，组内顺序就是沿路顺序。
    by_road: dict[int, list[Any]] = {}
    for waypoint in _all_waypoints(world, carla):
        if not (waypoint.lane_type & walkable):
            continue
        by_road.setdefault(int(waypoint.road_id), []).append(waypoint)

    def far_from_cameras(point: Any) -> bool:
        if not cameras:
            return True
        return all(
            math.hypot(
                point.x - camera["location"][0],
                point.y - camera["location"][1],
            )
            >= min_camera_distance_m
            for camera in cameras
        )

    def roadside(point: Any, side: float) -> Any:
        """把车道中心线上的点横向推到路侧。"""
        location = point.transform.location
        yaw = math.radians(point.transform.rotation.yaw)
        width = float(getattr(point, "lane_width", 3.5) or 3.5)
        offset = width / 2.0 + 0.9
        # 法向：yaw 逆时针 90°。
        return carla.Location(
            x=location.x - math.sin(yaw) * offset * side,
            y=location.y + math.cos(yaw) * offset * side,
            z=location.z,
        )

    pairs: list[tuple[Any, Any]] = []
    candidates: list[tuple[Any, Any, Any]] = []
    for road_points in by_road.values():
        for index, waypoint in enumerate(road_points):
            location = waypoint.transform.location
            distance = _distance(location, origin)
            if not radius_range[0] <= distance <= radius_range[1]:
                continue
            if not far_from_cameras(location):
                continue
            # 沿路往前找 15~40 m 的点作为目标。
            for step in range(
                int(walk_range[0] // 2), int(walk_range[1] // 2) + 1
            ):
                target_index = index + step
                if target_index >= len(road_points):
                    break
                target = road_points[target_index]
                target_location = target.transform.location
                if not far_from_cameras(target_location):
                    continue
                side = 1.0 if rng.random() < 0.5 else -1.0
                candidates.append(
                    (
                        roadside(waypoint, side),
                        roadside(target, side),
                    )
                )
                break
    rng.shuffle(candidates)
    # 出生点之间拉开距离，避免一堆行人挤在一处。
    for start, goal in candidates:
        if len(pairs) >= count:
            break
        if any(
            math.hypot(start.x - other[0].x, start.y - other[0].y) < 4.0
            for other in pairs
        ):
            continue
        pairs.append((start, goal))
    return pairs


def _cluster(points: list[Any], radius: float) -> list[list[Any]]:
    """按距离贪心聚类：彼此在 radius 米内的点归成一组。"""
    clusters: list[list[Any]] = []
    for point in points:
        for cluster in clusters:
            if any(_distance(point, other) <= radius for other in cluster):
                cluster.append(point)
                break
        else:
            clusters.append([point])
    return clusters


def _crosswalk_polygons(world: Any) -> list[list[Any]]:
    """把 get_crosswalks() 的扁平顶点列表还原成一条条人行横道。

    CARLA 的文档写明返回的是"闭合多边形，首点重复表示多边形的起止"，
    所以顶点是按横道顺次排列的：遇到与当前多边形首点重合的点就是闭合。

    万一服务器没按这个格式返回（整张图变成一个大多边形），按 10 m 的
    紧半径重新聚类——横道本身只有几米宽，这样切出来的每组才是一条横道。
    之前用 12 m 半径切，会把整个路口的横道连成一坨，取出来的"对侧点"
    能差出 78 m，行人直接穿楼。
    """
    points = world.get_map().get_crosswalks()
    print(f"  人行横道顶点共 {len(points)} 个", flush=True)
    polygons: list[list[Any]] = []
    current: list[Any] = []
    for point in points:
        if current and _distance(point, current[0]) <= 0.01:
            polygons.append(current)
            current = []
        else:
            current.append(point)
    if len(current) >= 2:
        polygons.append(current)
    # 直径过大的说明分组失败（多边形没按横道切开），重新按紧半径聚类。
    split: list[list[Any]] = []
    for polygon in polygons:
        span = max(
            (
                _distance(polygon[i], polygon[j])
                for i in range(len(polygon))
                for j in range(i + 1, len(polygon))
            ),
            default=0.0,
        )
        split.extend(_cluster(polygon, 10.0) if span > 40.0 else [polygon])
    polygons = split
    print(f"  还原出人行横道 {len(polygons)} 条", flush=True)
    return polygons


def _crosswalk_pairs(
    world: Any,
    carla: Any,
    center: tuple[float, float],
    radius_range: list[float],
    count: int,
    cameras: list[dict[str, Any]] | None = None,
    min_camera_distance_m: float = 10.0,
) -> list[tuple[Any, Any]]:
    """横穿人行横道的 (起点, 对侧目标点)。

    Town03 的人行道 waypoint 取不到（那里没有车道级 OpenDRIVE 数据），
    人行横道是唯一能拿到行人站位的来源。

    横道是横跨马路的窄长矩形：长边横跨马路，短边沿路。行人沿长边走才
    是过街，沿对角线走会斜穿马路。所以取"长度明显大于其他边"的边。

    radius_range 的下界（默认 30 m）是按人行道 waypoint 定的，横道全挤在
    路口上，用它会把横道全部滤光，所以这里只受上界和"离相机至少
    min_camera_distance_m"约束：行人站到镜头底下会瞬间占满整幅画面，
    事件缓冲区被灌爆，那个机位的 DVS 就此停摆。
    """
    origin = carla.Location(x=center[0], y=center[1])
    pairs: list[tuple[Any, Any]] = []
    for polygon in _crosswalk_polygons(world):
        if len(polygon) < 2:
            continue
        edges = [
            (polygon[index], polygon[(index + 1) % len(polygon)])
            for index in range(len(polygon))
        ]
        lengths = [_distance(start, end) for start, end in edges]
        longest = max(lengths)
        if longest <= 0.0:
            continue
        for (start, end), length in zip(edges, lengths):
            # 只取长边（横跨马路的那一段），且长度得像个过街距离。
            # 实测 Town03 有长边 30 m 的横道，那种"过街"要走 25 s，
            # 行人有一半时间杵在马路中间，直接丢掉。
            # 上限 22 m：路口唯一那条横道的长边是 20.16 m，卡在 20 会被
            # 整条丢掉，只剩一条过街线，一堆行人挤成一股。
            if length < longest * 0.6 or length > 22.0:
                continue
            pairs.append((start, end))
    if not pairs:
        return []

    def midpoint(pair: tuple[Any, Any]) -> Any:
        return carla.Location(
            x=(pair[0].x + pair[1].x) / 2.0,
            y=(pair[0].y + pair[1].y) / 2.0,
        )

    def midpoint_distance(pair: tuple[Any, Any]) -> float:
        return _distance(midpoint(pair), origin)

    def far_from_cameras(pair: tuple[Any, Any]) -> bool:
        if not cameras:
            return True
        # 两端都要离镜头够远：行人会沿整条横道走，只看中点会漏掉
        # "一头就杵在镜头底下"的横道。
        for point in (pair[0], pair[1]):
            for camera in cameras:
                if (
                    math.hypot(
                        point.x - camera["location"][0],
                        point.y - camera["location"][1],
                    )
                    < min_camera_distance_m
                ):
                    return False
        return True

    def camera_distance(pair: tuple[Any, Any]) -> float:
        """横道两端到最近相机的距离：兜底排序用，越大越安全。"""
        if not cameras:
            return 0.0
        return min(
            math.hypot(
                point.x - camera["location"][0],
                point.y - camera["location"][1],
            )
            for point in (pair[0], pair[1])
            for camera in cameras
        )

    # 先按"离路口中心的距离"排序：近的优先，人数不够再用远的补齐。
    order = sorted(
        range(len(pairs)), key=lambda index: midpoint_distance(pairs[index])
    )
    in_range = [
        pairs[index]
        for index in order
        if midpoint_distance(pairs[index]) <= radius_range[1]
    ]
    selected = [
        pair for pair in in_range if far_from_cameras(pair)
    ][:count]
    if not selected:
        # 一条横道都不够远：路口就这么大，行人只能站近一点。取离镜头最远的
        # 那几条，别让整条序列一个行人都没有（近到什么程度由
        # _sanitize_walkers 兜底）。排序键是到最近相机的距离——按到路口
        # 中心的距离排会把行人放到镜头跟前。
        print(
            "提示：没有离相机足够远的人行横道，改用离镜头最远的几条", flush=True
        )
        selected = sorted(
            in_range, key=camera_distance, reverse=True
        )[:count]
    return selected


def _spawn_point_pairs(
    world: Any,
    carla: Any,
    center: tuple[float, float],
    radius_range: list[float],
    walk_range: list[float],
    count: int,
) -> list[tuple[Any, Any]]:
    """兜底：用地图自带出生点配对出人行路线。

    人行道和人行横道都拿不到时的最后手段（地图数据异常）。
    """
    origin = carla.Location(x=center[0], y=center[1])
    points = [
        point
        for point in world.get_map().get_spawn_points()
        if radius_range[0]
        <= _distance(point.location, origin)
        <= radius_range[1]
    ]
    points.sort(key=lambda point: _distance(point.location, origin))
    pairs: list[tuple[Any, Any]] = []
    for point in points:
        for other in points:
            if walk_range[0] <= _distance(point.location, other.location) <= walk_range[1]:
                pairs.append((point.location, other.location))
                break
        if len(pairs) >= count:
            break
    return pairs[:count]


def _prune_stuck_walkers(
    world: Any,
    carla: Any,
    walkers: list[dict[str, Any]],
    spawned_actors: list[Any],
) -> tuple[list[dict[str, Any]], int]:
    """空跑 60 tick 后，位移不足 0.5 m 的行人判定为卡住并销毁。

    返回 (留下的行人计划, 被销毁的数量)。卡住的行人在旧数据里表现为
    "几十个行人生成在同一个点、速度恒为 0"（方案 §3）。

    60 tick（约 1.5 s）配 0.5 m 阈值：正常步速 0.8~1.8 m/s 的行人这段时间
    能走 1.2~2.7 m，而原地不动的行人位移为 0，两者分得开。之前是
    20 tick（0.5 s）配 1 m 阈值——行人走不到 1 m，全被判成卡住销毁，
    实测 36 个行人只剩 1 个（Town10HD 序列的验收记录）。
    """
    if not walkers:
        return [], 0
    # 生成后客户端要等一次 tick 才有真实位姿，此前 get_location() 一律
    # 返回 (0,0,0)。不先同步就取起点，位移恒为"真实位置到原点的距离"
    # （几十米），卡住的行人也永远检测不出来——这个判据形同虚设。
    world.tick()
    before = {
        plan["actor"].id: plan["actor"].get_location() for plan in walkers
    }
    for _ in range(60):
        update_walkers(carla, walkers)
        world.tick()
    stuck = [
        plan
        for plan in walkers
        if _distance(before[plan["actor"].id], plan["actor"].get_location()) < 0.5
    ]
    stuck_ids = {plan["actor"].id for plan in stuck}
    # 只按 id 剔除，不拿 actor 对象做集合成员判断：CARLA 的 Actor.__eq__
    # 不可靠（Transform 同理），塞进 set 既可能报错也可能判错。
    spawned_actors[:] = [
        actor for actor in spawned_actors if actor.id not in stuck_ids
    ]
    for plan in stuck:
        try:
            plan["actor"].destroy()
        except RuntimeError:
            pass
    alive = [plan for plan in walkers if plan["actor"].id not in stuck_ids]
    return alive, len(stuck)


def _sanitize_walkers(
    world: Any,
    carla: Any,
    walkers: list[dict[str, Any]],
    spawned_actors: list[Any],
    cameras: list[dict[str, Any]],
    min_camera_distance_m: float = 6.0,
) -> tuple[list[dict[str, Any]], int]:
    """销毁离相机太近、或悬在半空的行人。

    行人贴着镜头走时整幅画面同时突变，DVS 的事件缓冲区会被灌爆，那个
    传感器从此不再出帧（实测 cam_02_dvs 在第一帧就停摆，整条序列报废）。
    相机就架在路口中心，正常目标点在 30~60 m 外不会靠近镜头；但生成时
    陷进地面、被物理挤飞的行人可能飞到镜头前。

    阈值 6 m：相机在 6 m 高，行人站到 6 m 外时已经在画面下边缘之外，
    再近就是整个人糊在镜头上。
    """
    if not walkers or not cameras:
        return walkers, 0
    doomed: list[dict[str, Any]] = []
    for plan in walkers:
        try:
            location = plan["actor"].get_location()
        except RuntimeError:
            doomed.append(plan)
            continue
        if location.z > 2.5:
            # 悬在半空：生成时被挤飞了。
            doomed.append(plan)
            continue
        for camera in cameras:
            cx, cy = camera["location"][0], camera["location"][1]
            if math.hypot(location.x - cx, location.y - cy) < min_camera_distance_m:
                doomed.append(plan)
                break
    if not doomed:
        return walkers, 0
    doomed_ids = {plan["actor"].id for plan in doomed}
    spawned_actors[:] = [
        actor for actor in spawned_actors if actor.id not in doomed_ids
    ]
    for plan in doomed:
        try:
            plan["actor"].destroy()
        except RuntimeError:
            pass
    alive = [plan for plan in walkers if plan["actor"].id not in doomed_ids]
    return alive, len(doomed)


def spawn_road_traffic(
    world: Any,
    carla: Any,
    traffic_manager: Any,
    vehicle_count: int,
    walker_count: int,
    seed: int,
    traffic_config: dict[str, Any],
    spawned_actors: list[Any],
    junction_center: list[float] | None = None,
    cameras: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """按种子生成可复现的车辆和行人，并随机化车辆行为参数。

    返回 (各类目标的生成数量, 存活的行人计划)。行人计划交给采集主循环，
    每个 tick 前用 update_walkers() 推一步（行人不用 AI 控制器，
    见 _spawn_walker 的说明）。
    """
    rng = random.Random(seed)
    library = world.get_blueprint_library()
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    camera_list = cameras or []
    near_radius = traffic_config.get("spawn_near_camera_m", 45.0)
    far_radius = traffic_config.get("spawn_far_camera_m", 60.0)
    # 相机视野内的出生点优先：否则全图随机会让某些机位几乎拍不到车。
    spawn_points = _prioritize_spawn_points(
        spawn_points, camera_list, rng, near_radius, far_radius
    )

    car_blueprints = []
    motorcycle_blueprints = []
    for blueprint in library.filter("vehicle.*"):
        category = classify_blueprint(blueprint.id)["category"]
        if category == "car":
            car_blueprints.append(blueprint)
        elif category == "motorcycle":
            motorcycle_blueprints.append(blueprint)
    if not car_blueprints:
        raise RuntimeError("当前 CARLA 地图没有可用车辆蓝图")

    motorcycle_ratio = traffic_config.get("motorcycle_ratio", 0.3)
    vehicles_spawned = 0
    motorcycles_spawned = 0
    # 相机正前方的车道优先放车：地图自带出生点只覆盖路口少数车道，
    # cam_03 这类机位靠出生点几乎拍不到交通。沿朝向取到的车道点排在最前，
    # 其后才是地图出生点。
    road_transforms = _camera_road_transforms(
        world, carla, camera_list, near_radius * 0.3, far_radius
    )
    rng.shuffle(road_transforms)
    vehicle_placements = road_transforms + spawn_points
    # 类别按固定序列轮转，而不是每个位置随机抽：随机抽会让摩托挤占
    # 道路点（摩托更容易生成成功），某些机位就只剩摩托没有车。
    for index, transform in enumerate(vehicle_placements):
        if vehicles_spawned >= vehicle_count:
            break
        use_motorcycle = (
            bool(motorcycle_blueprints)
            and motorcycle_ratio > 0
            and (index % max(1, int(round(1 / motorcycle_ratio)))) == 0
        )
        pool = motorcycle_blueprints if use_motorcycle else car_blueprints
        vehicle = _spawn_vehicle(
            world,
            carla,
            traffic_manager,
            rng.choice(pool),
            transform,
            rng,
            traffic_config,
            spawned_actors,
        )
        if vehicle is None:
            continue
        vehicles_spawned += 1
        if use_motorcycle:
            motorcycles_spawned += 1

    print(
        f"交通流：车辆 {vehicles_spawned} 辆已生成（含摩托 {motorcycles_spawned} 辆）",
        flush=True,
    )
    walkers_spawned = 0
    walker_blueprints = list(library.filter("walker.pedestrian.*"))
    if walker_count and not walker_blueprints:
        raise RuntimeError("当前 CARLA 未提供行人蓝图")
    center = (
        (float(junction_center[0]), float(junction_center[1]))
        if junction_center
        else (
            # 没有路口坐标的 rig 以第一台相机为中心生成行人：行人生成在
            # 相机周边的人行道上，相机才拍得到。
            # 绝不退回 get_random_location_from_navigation() 的全城撒点：
            # 那个原生导航调用在 Windows 客户端上会栈溢出（exit 0xC00000FD），
            # 把整个采集带崩，而且撒出来的行人在相机视野外根本拍不到。
            (float(camera_list[0]["location"][0]), float(camera_list[0]["location"][1]))
            if camera_list
            else None
        )
    )
    walkers: list[dict[str, Any]] = []
    if walker_count:
        sidewalk_pairs: list[tuple[Any, Any]] = []
        crosswalk_pairs: list[tuple[Any, Any]] = []
        roadside_pairs: list[tuple[Any, Any]] = []
        radius_range = traffic_config.get("pedestrian_spawn_radius_m", [30, 60])
        walk_range = traffic_config.get("pedestrian_walk_distance_m", [15, 40])
        if center is not None:
            print("行人：取人行道目标点 …", flush=True)
            sidewalk_pairs = _sidewalk_pairs(
                world,
                carla,
                rng,
                center,
                radius_range,
                walk_range,
                walker_count,
            )
            print(f"行人：人行道目标点 {len(sidewalk_pairs)} 组", flush=True)
            print("行人：取人行横道目标点 …", flush=True)
            crosswalk_pairs = _crosswalk_pairs(
                world,
                carla,
                center,
                radius_range,
                walker_count,
                cameras=camera_list,
            )
            print(f"行人：人行横道目标点 {len(crosswalk_pairs)} 组", flush=True)
            print("行人：取路侧步行目标点 …", flush=True)
            # 路侧步行是横道稀疏地图的主力来源：Town10HD_Opt 全图只有
            # 16 条人行横道（Town03 有 73 条），光靠横道凑不出足够的行人，
            # 剩下的位置会落到有静态碰撞体的地点，生成直接失败。
            roadside_pairs = _roadside_pairs(
                world,
                carla,
                rng,
                center,
                radius_range,
                walk_range,
                walker_count,
                cameras=camera_list,
            )
            print(f"行人：路侧步行目标点 {len(roadside_pairs)} 组", flush=True)
        if not (sidewalk_pairs or crosswalk_pairs or roadside_pairs) and center is not None:
            print("行人：人行道、人行横道、路侧都不可用，退回地图出生点 …", flush=True)
            sidewalk_pairs = _spawn_point_pairs(
                world,
                carla,
                center,
                radius_range,
                walk_range,
                walker_count,
            )
            print(f"行人：出生点目标点 {len(sidewalk_pairs)} 组", flush=True)
        if not (sidewalk_pairs or crosswalk_pairs or roadside_pairs):
            # 静默产出"0 行人"序列的代价是白跑一次几小时的采集（实测
            # Town10HD_Opt 的 5 个机位组全部中招）。直接失败，让操作者
            # 第一时间看到，而不是事后翻 manifest。
            raise RuntimeError(
                f"行人目标点为 0（中心 {center}，半径 "
                f"{radius_range} m 内没有人行道/人行横道/路侧可用）"
            )
        crosswalk_ratio = traffic_config.get("crosswalk_ratio", 0.4)
        # 每一类目标点各用一个游标，轮流取用：同一个游标会被"跳过"，
        # 交叉抽签时一半的目标点永远轮不到。
        sidewalk_cursor = 0
        crosswalk_cursor = 0
        roadside_cursor = 0
        for _ in range(walker_count):
            # 目标点来源按权重轮转：横道 crosswalk_ratio，人行道和路侧
            # 平分剩下的。横道不够时自动多走路侧，不浪费名额。
            use_crosswalk = bool(crosswalk_pairs) and (
                not (sidewalk_pairs or roadside_pairs)
                or rng.random() < crosswalk_ratio
            )
            if use_crosswalk:
                start, destination = crosswalk_pairs[
                    crosswalk_cursor % len(crosswalk_pairs)
                ]
                crosswalk_cursor += 1
            elif sidewalk_pairs and (
                not roadside_pairs or rng.random() < 0.5
            ):
                start, destination = sidewalk_pairs[
                    sidewalk_cursor % len(sidewalk_pairs)
                ]
                sidewalk_cursor += 1
            elif roadside_pairs:
                start, destination = roadside_pairs[
                    roadside_cursor % len(roadside_pairs)
                ]
                roadside_cursor += 1
            elif sidewalk_pairs:
                start, destination = sidewalk_pairs[
                    sidewalk_cursor % len(sidewalk_pairs)
                ]
                sidewalk_cursor += 1
            else:
                # 既没有路口坐标也没有相机位置，给不出行人目标点，直接不生成。
                # 曾经在这里退回 get_random_location_from_navigation()，那是
                # 崩溃根因。
                continue
            plan = _spawn_walker(
                world,
                carla,
                rng.choice(walker_blueprints),
                start,
                destination,
                rng,
                spawned_actors,
            )
            if plan is None:
                continue
            walkers.append(plan)
            walkers_spawned += 1

    alive, stuck = _prune_stuck_walkers(
        world, carla, walkers, spawned_actors
    )
    alive, unsafe = _sanitize_walkers(
        world, carla, alive, spawned_actors, camera_list
    )
    if walker_count and not alive:
        # 生成了但全被销毁：同样直接失败，避免静默产出无行人序列。
        raise RuntimeError(
            f"行人生成 {walkers_spawned} 个但存活 0 个"
            f"（卡住销毁 {stuck}，离镜头过近销毁 {unsafe}）"
        )
    print(
        f"交通流：行人 {walkers_spawned} 个已生成（卡住销毁 {stuck} 个，"
        f"离镜头过近销毁 {unsafe} 个，存活 {len(alive)} 个）",
        flush=True,
    )
    counts = {
        "vehicles": vehicles_spawned,
        "motorcycles": motorcycles_spawned,
        "walkers": walkers_spawned,
        "walkers_stuck_removed": stuck,
        "walkers_alive": len(alive),
    }
    # 行人计划要交给采集主循环，每个 tick 前推一次（见 update_walkers）。
    return counts, alive
