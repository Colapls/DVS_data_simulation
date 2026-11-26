#!/usr/bin/env python3
"""行人生成/行走/清理的自检（不依赖 CARLA）。

回归背景：

- 控制器与行人的配对曾用 spawned_actors[-2] 取控制器，实际取到的是行人
  自己；卡住销毁时对 Walker 调 .stop() 直接 AttributeError 崩掉整个采集。
- 卡住判定曾用「20 tick / 1 m」，行人走不到 1 m 就被判卡住，
  实测 36 个行人只剩 1 个。
- 行人不再使用 controller.ai.walker（那条路径要走导航网格，Town03 的
  导航数据不完整，服务端路径搜索会递归到爆栈），改由 WalkerControl
  手动推人，走行逻辑在这里验。

    python tools/test_traffic.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvs_collector.traffic import (  # noqa: E402
    _prune_stuck_walkers,
    _spawn_walker,
    update_walkers,
)


class _Loc:
    """最小化的 carla.Location 替身。"""

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Loc):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __repr__(self) -> str:
        return f"_Loc({self.x}, {self.y}, {self.z})"


class _Vec:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Control:
    def __init__(self, direction: _Vec, speed: float) -> None:
        self.direction = direction
        self.speed = speed


class _FakeCarla:
    """只实现行人生成/行走用到的接口。"""

    Location = staticmethod(lambda x=0.0, y=0.0, z=0.0: _Loc(x, y, z))
    Vector3D = staticmethod(lambda x=0.0, y=0.0, z=0.0: _Vec(x, y, z))
    WalkerControl = staticmethod(
        lambda direction=None, speed=0.0, jump=False: _Control(direction, speed)
    )
    Transform = staticmethod(lambda location=None, rotation=None: ("transform", location))


class _FakeBlueprint:
    """最小化的 carla.ActorBlueprint 替身。"""

    def __init__(self) -> None:
        self.attributes: dict[str, str] = {}

    def has_attribute(self, name: str) -> bool:
        return name == "speed"

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value


class _FakeWalker:
    def __init__(self, walker_id: int, positions: list[_Loc]) -> None:
        self.id = walker_id
        self._positions = list(positions)  # 每次 get_location 弹出下一个
        self.destroyed = False
        self.controls: list[_Control] = []

    def get_location(self) -> _Loc:
        if len(self._positions) > 1:
            return self._positions.pop(0)
        return self._positions[0]

    def apply_control(self, control: _Control) -> None:
        self.controls.append(control)

    def destroy(self) -> None:
        self.destroyed = True


class _FakeWorld:
    """只实现 _spawn_walker / _prune_stuck_walkers 用到的接口。"""

    def __init__(self, walker: _FakeWalker | None = None) -> None:
        self.walker = walker
        self.ticks = 0

    def try_spawn_actor(self, blueprint: object, transform: object) -> _FakeWalker:
        return self.walker

    def tick(self) -> None:
        self.ticks += 1


def test_spawn_walker_returns_plan() -> None:
    walker = _FakeWalker(101, [_Loc(0.0, 0.0)])
    world = _FakeWorld(walker)
    blueprint = _FakeBlueprint()
    spawned_actors: list[object] = []
    plan = _spawn_walker(
        world, _FakeCarla(), blueprint, _Loc(0, 0), _Loc(10, 0),
        random.Random(1), spawned_actors,
    )
    assert plan is not None
    # 回归点：不再生成 controller.ai.walker 演员，只返回行走计划。
    assert plan["actor"] is walker
    assert plan["goal"] == _Loc(10, 0)
    assert plan["to_goal"] is True
    assert spawned_actors == [walker]
    # 出生点沿过街方向抖动（横向不超过 1 m，免得抖到镜头底下），
    # 但仍在目标点附近。
    assert abs(plan["start"].x - 0.0) <= 3.5
    assert abs(plan["start"].y - 0.0) <= 1.0
    # 行人的速度上限要抬起来，否则 WalkerControl 的速度设不上去。
    assert blueprint.attributes["speed"] == "2.0"


def test_spawn_walker_jitter_spreads_pedestrians() -> None:
    """同一批目标点上生成的行人不能全挤在同一个坐标。"""
    walker = _FakeWalker(1, [_Loc(0.0, 0.0)])
    world = _FakeWorld(walker)
    rng = random.Random(7)
    starts = []
    for _ in range(8):
        plan = _spawn_walker(
            world, _FakeCarla(), _FakeBlueprint(), _Loc(0, 0), _Loc(10, 0), rng, []
        )
        starts.append((plan["start"].x, plan["start"].y))
    assert len(set(starts)) > 1


def test_update_walkers_pushes_toward_goal() -> None:
    walker = _FakeWalker(1, [_Loc(1.0, 0.0)])
    plan = {
        "actor": walker,
        "start": _Loc(1.0, 0.0),
        "goal": _Loc(11.0, 0.0),
        "speed": 1.2,
        "to_goal": True,
    }
    update_walkers(_FakeCarla(), [plan])
    assert len(walker.controls) == 1
    control = walker.controls[0]
    assert control.speed == 1.2
    # 朝目标点走：+x 方向，单位长度。
    assert abs(control.direction.x - 1.0) < 1e-6
    assert abs(control.direction.y) < 1e-6


def test_update_walkers_flips_at_goal() -> None:
    walker = _FakeWalker(1, [_Loc(10.8, 0.0)])  # 已经站在目标点旁边
    plan = {
        "actor": walker,
        "start": _Loc(1.0, 0.0),
        "goal": _Loc(11.0, 0.0),
        "speed": 1.0,
        "to_goal": True,
    }
    update_walkers(_FakeCarla(), [plan])
    assert plan["to_goal"] is False
    control = walker.controls[0]
    # 掉头走回起点：-x 方向。
    assert abs(control.direction.x + 1.0) < 1e-6


def test_update_walkers_does_not_reapply_same_direction() -> None:
    """回归点：方向没变时不能重复下发控制。

    apply_control 会让服务端重做一次导航网格寻路，40 个行人 × 40 tick/s
    就是每秒近千次寻路，帧时间被拖垮，传感器直接超时。
    """
    walker = _FakeWalker(1, [_Loc(1.0, 0.0)])
    plan = {
        "actor": walker,
        "start": _Loc(1.0, 0.0),
        "goal": _Loc(11.0, 0.0),
        "speed": 1.0,
        "to_goal": True,
    }
    update_walkers(_FakeCarla(), [plan])
    assert len(walker.controls) == 1
    for _ in range(10):
        update_walkers(_FakeCarla(), [plan])
    # 方向一直是 +x，只下发过一次。
    assert len(walker.controls) == 1


def test_crosswalk_pairs_use_long_edges() -> None:
    """回归点：横道是横跨马路的窄长矩形，只有长边才是过街方向。

    取对角线会让行人斜穿马路，取短边则是沿路走，两者都不对。
    """
    from dvs_collector.traffic import _crosswalk_pairs

    class _FakeMap:
        def get_crosswalks(self) -> list[_Loc]:
            # 两条横道，每条 4 个顶点 + 首点重复（文档里的闭合多边形）。
            points = []
            for base_x in (0.0, 100.0):
                corners = [
                    _Loc(base_x, 0.0),
                    _Loc(base_x + 8.0, 0.0),
                    _Loc(base_x + 8.0, 3.0),
                    _Loc(base_x, 3.0),
                ]
                points.extend(corners)
                points.append(corners[0])
            return points

    class _FakeWorld:
        def get_map(self) -> _FakeMap:
            return _FakeMap()

    pairs = _crosswalk_pairs(
        _FakeWorld(), _FakeCarla(), (50.0, 0.0), [0.0, 500.0], 10
    )
    # 每条横道两条长边 → 两条过街通道。
    assert len(pairs) == 4
    for start, goal in pairs:
        span = ((start.x - goal.x) ** 2 + (start.y - goal.y) ** 2) ** 0.5
        # 长边正好是横跨马路的 8 m；短边 3 m、对角线 8.54 m 都不该被选中。
        assert span == 8.0


def test_crosswalk_pairs_skip_pairs_too_close_to_camera() -> None:
    """回归点：行人不能站到镜头底下。

    横道全挤在路口上，"按半径补齐人数"的旧逻辑会把行人塞到相机跟前，
    整幅画面同时突变，DVS 的事件缓冲区被灌爆后那个机位就彻底停摆。
    """
    from dvs_collector.traffic import _crosswalk_pairs

    class _FakeMap:
        def get_crosswalks(self) -> list[_Loc]:
            points = []
            # 一条贴着相机的横道（离相机 3 m），一条远处的横道（40 m）。
            for base_x, base_y in ((3.0, 0.0), (40.0, 0.0)):
                corners = [
                    _Loc(base_x, base_y),
                    _Loc(base_x + 8.0, base_y),
                    _Loc(base_x + 8.0, base_y + 3.0),
                    _Loc(base_x, base_y + 3.0),
                ]
                points.extend(corners)
                points.append(corners[0])
            return points

    class _FakeWorld:
        def get_map(self) -> _FakeMap:
            return _FakeMap()

    cameras = [{"location": (0.0, 0.0, 6.0)}]
    pairs = _crosswalk_pairs(
        _FakeWorld(), _FakeCarla(), (0.0, 0.0), [30.0, 500.0], 10,
        cameras=cameras, min_camera_distance_m=10.0,
    )
    assert len(pairs) == 2
    for start, goal in pairs:
        for point in (start, goal):
            assert point.x >= 10.0


def test_crosswalk_polygons_recluster_when_no_polygon_markers() -> None:
    """回归点：服务器没给"首点重复"标记时，整张图会是一个大多边形。

    这时必须按紧半径重新切分，否则取出来的"对侧点"能差出 78 m，
    行人直接穿楼。
    """
    from dvs_collector.traffic import _crosswalk_pairs

    class _FakeMap:
        def get_crosswalks(self) -> list[_Loc]:
            # 两条横道，但没有首点重复（不按闭合多边形返回）。
            points = []
            for base_x in (0.0, 100.0):
                points.extend([
                    _Loc(base_x, 0.0),
                    _Loc(base_x + 8.0, 0.0),
                    _Loc(base_x + 8.0, 3.0),
                    _Loc(base_x, 3.0),
                ])
            return points

    class _FakeWorld:
        def get_map(self) -> _FakeMap:
            return _FakeMap()

    pairs = _crosswalk_pairs(
        _FakeWorld(), _FakeCarla(), (50.0, 0.0), [0.0, 500.0], 10
    )
    assert len(pairs) == 4
    for start, goal in pairs:
        span = ((start.x - goal.x) ** 2 + (start.y - goal.y) ** 2) ** 0.5
        # 过街距离就几米到十几米。
        assert 3.0 < span <= 40.0


def test_prune_destroys_stuck_pair() -> None:
    stuck = _FakeWalker(1, [_Loc(0.0, 0.0), _Loc(0.0, 0.0)])  # 空跑后仍在原地
    moving = _FakeWalker(2, [_Loc(0.0, 0.0), _Loc(50.0, 0.0)])
    world = _FakeWorld()
    spawned = [stuck, moving]
    plans = [
        {"actor": stuck, "start": _Loc(0, 0), "goal": _Loc(50, 0), "speed": 1.0, "to_goal": True},
        {"actor": moving, "start": _Loc(0, 0), "goal": _Loc(50, 0), "speed": 1.0, "to_goal": True},
    ]
    alive, removed = _prune_stuck_walkers(world, _FakeCarla(), plans, spawned)
    assert removed == 1
    assert [plan["actor"].id for plan in alive] == [2]
    assert stuck.destroyed
    assert not moving.destroyed
    # 被销毁的已从清单剔除，其余保留。
    assert stuck not in spawned and moving in spawned
    # 60 次判定空跑 + 1 次位姿同步（生成后 get_location() 要等一次 tick）。
    assert world.ticks == 61


def test_prune_keeps_normal_pedestrian() -> None:
    """回归点：正常步速的行人 60 tick 只走 1.2 m，不能被当成卡住销毁。

    旧阈值是 20 tick / 1 m，行人走不到 1 m 就被判卡住，
    实测 36 个行人只剩 1 个。
    """
    pedestrian = _FakeWalker(3, [_Loc(0.0, 0.0), _Loc(1.2, 0.0)])
    world = _FakeWorld()
    spawned = [pedestrian]
    plans = [
        {"actor": pedestrian, "start": _Loc(0, 0), "goal": _Loc(50, 0), "speed": 1.0, "to_goal": True},
    ]
    alive, removed = _prune_stuck_walkers(world, _FakeCarla(), plans, spawned)
    assert removed == 0
    assert [plan["actor"].id for plan in alive] == [3]
    assert not pedestrian.destroyed


def test_crosswalk_fallback_picks_far_from_camera() -> None:
    """回归点：全部横道都离相机过近时，兜底要取"离镜头最远"的横道。

    旧实现按"离路口中心的距离"倒序取（midpoint_distance 是到中心的距离，
    注释却写着离镜头最远），会把行人放到镜头跟前 0.1 m 的横道上，
    DVS 事件缓冲区被灌爆。正确的排序键是到最近相机的距离。
    """
    from dvs_collector.traffic import _crosswalk_pairs

    class _FakeMap:
        def get_crosswalks(self) -> list[_Loc]:
            points = []
            # A 横道在 (6..14, 0..3)，离中心远（10 m）；相机1 在 (2,0)，
            # A 端点离相机1 4 m —— 两条横道里离镜头更远的那条（4 m vs 0.1 m）。
            # B 横道在 (76..84, 0..3)，离中心最远（80 m）；相机2 贴着它（0.1 m）。
            for base_x in (6.0, 76.0):
                corners = [
                    _Loc(base_x, 0.0),
                    _Loc(base_x + 8.0, 0.0),
                    _Loc(base_x + 8.0, 3.0),
                    _Loc(base_x, 3.0),
                ]
                points.extend(corners)
                points.append(corners[0])
            return points

    class _FakeWorld:
        def get_map(self) -> _FakeMap:
            return _FakeMap()

    cameras = [
        {"location": (2.0, 0.0, 6.0)},
        {"location": (79.9, 0.0, 6.0)},
    ]
    pairs = _crosswalk_pairs(
        _FakeWorld(), _FakeCarla(), (0.0, 0.0), [0.0, 500.0], 1,
        cameras=cameras, min_camera_distance_m=10.0,
    )
    assert len(pairs) == 1
    start, goal = pairs[0]
    # 选中的必须是 A 横道（离镜头 4 m），不是贴着相机 2 的 B 横道。
    assert 0.0 <= start.x <= 14.0, f"选中了贴着镜头的横道：start.x={start.x}"


def test_roadside_pairs_cover_sparse_crosswalk_maps() -> None:
    """回归点：横道稀疏的地图也要有足够的行人目标点。

    Town10HD_Opt 全图只有 16 条人行横道（Town03 有 73 条）：路口周边
    60 m 内仅 4 条，3 条离相机不到 10 m 被滤掉，剩下一条正好落在有静态
    碰撞体的位置，生成直接失败——整条序列 0 行人。行人可以沿路肩走，
    不必非得走人行横道。
    """
    from dvs_collector.traffic import _roadside_pairs

    class _FakeWp:
        def __init__(self, x, y, road_id, index, lane_width=3.5):
            self.x, self.y = x, y
            self.road_id = road_id
            self.index = index
            self.lane_width = lane_width
            self.lane_type = 1  # Driving
            self.transform = type(
                "T", (), {"location": _Loc(x, y, 0.0), "rotation": type("R", (), {"yaw": 0.0})()}
            )()

    class _FakeMap:
        name = "FakeMapRoadside"

        def generate_waypoints(self, spacing):
            # 一条东西向路：road 1，20 个点，每点间隔 2 m。
            return [_FakeWp(i * 2.0, 10.0, 1, i) for i in range(20)]

    class _FakeWorld:
        def get_map(self):
            return _FakeMap()

    class _FakeCarla:
        Location = staticmethod(lambda x=0.0, y=0.0, z=0.0: _Loc(x, y, z))
        LaneType = type("L", (), {"Sidewalk": 2, "Driving": 1, "Shoulder": 8, "Biking": 4})

    cameras = [{"location": (0.0, 0.0, 4.0), "yaw": 0.0}]
    pairs = _roadside_pairs(
        _FakeWorld(), _FakeCarla(), random.Random(3),
        center=(0.0, 0.0), radius_range=(5.0, 60.0), walk_range=(15.0, 40.0),
        count=10, cameras=cameras,
    )
    assert len(pairs) >= 5, f"路侧目标点太少：{len(pairs)}"
    for start, goal in pairs:
        # 出生点和目标点都要横向偏到车道中心线之外（车道中心 y=10），
        # 行人才像在路边走而不是站在路中间。
        assert abs(start.y - 10.0) > 1.0, f"出生点没偏到路侧：y={start.y}"
        assert abs(goal.y - 10.0) > 1.0, f"目标点没偏到路侧：y={goal.y}"
        span = ((goal.x - start.x) ** 2 + (goal.y - start.y) ** 2) ** 0.5
        assert 10.0 <= span <= 45.0, f"步行距离异常：{span}"


def test_roadside_pairs_skip_points_too_close_to_camera() -> None:
    """行人不能站到镜头跟前（会把 DVS 事件缓冲区灌爆）。"""
    from dvs_collector.traffic import _roadside_pairs

    class _FakeWp:
        def __init__(self, x, y, road_id, index):
            self.x, self.y = x, y
            self.road_id, self.index = road_id, index
            self.lane_width = 3.5
            self.lane_type = 1
            self.transform = type(
                "T", (), {"location": _Loc(x, y, 0.0), "rotation": type("R", (), {"yaw": 0.0})()}
            )()

    class _FakeMap:
        name = "FakeMapRoadsideNear"

        def generate_waypoints(self, spacing):
            return [_FakeWp(i * 2.0, 0.0, 1, i) for i in range(30)]

    class _FakeWorld:
        def get_map(self):
            return _FakeMap()

    class _FakeCarla:
        Location = staticmethod(lambda x=0.0, y=0.0, z=0.0: _Loc(x, y, z))
        LaneType = type("L", (), {"Sidewalk": 2, "Driving": 1, "Shoulder": 8, "Biking": 4})

    cameras = [{"location": (0.0, 0.0, 4.0), "yaw": 0.0}]
    pairs = _roadside_pairs(
        _FakeWorld(), _FakeCarla(), random.Random(1),
        center=(0.0, 0.0), radius_range=(5.0, 60.0), walk_range=(15.0, 40.0),
        count=30, cameras=cameras, min_camera_distance_m=10.0,
    )
    for start, goal in pairs:
        for point in (start, goal):
            assert abs(point.x) >= 10.0 or abs(point.y) >= 10.0, (
                f"行人离相机太近：({point.x}, {point.y})"
            )


def test_spawn_walker_lifts_ground_level_on_collision() -> None:
    """回归点：地面比 waypoint 的 z 高一点时，行人必须能生成。

    Town10HD_Opt 上 waypoint 的 z=0，实际碰撞面在 +0.05~+0.2 m：行人在
    z=0 生成一律报 "Spawn failed because of collision"（12/12 全失败），
    抬到 z+0.05/+0.20 才能成功（10/12、11/12）。Town03 的偏差在容差内，
    所以那边一直正常——同一个函数在两张地图上表现不同，根因就在这里。
    """
    class _GroundWorld(_FakeWorld):
        """只接受"脚底高于地面 0.05 m"的生成位置。"""

        def __init__(self) -> None:
            super().__init__()
            self.tried: list[float] = []

        def try_spawn_actor(self, blueprint, transform):
            # _FakeCarla.Transform 返回 ("transform", location) 元组
            location = (
                transform.location
                if hasattr(transform, "location")
                else transform[1]
            )
            z = location.z
            self.tried.append(z)
            if z < 0.05:
                return None
            walker = _FakeWalker(1, [_Loc(0.0, 0.0)])
            self.walker = walker
            return walker

    world = _GroundWorld()
    plan = _spawn_walker(
        world, _FakeCarla(), _FakeBlueprint(), _Loc(0, 0), _Loc(10, 0),
        random.Random(1), [],
    )
    assert plan is not None, "地面抬高后仍应能生成出行人"
    # 尝试过的 z 必须逐次抬高：先试 waypoint 给的高度，再往上抬。
    assert world.tried[0] == 0.0, f"第一次应该用 waypoint 的 z：{world.tried}"
    assert max(world.tried) >= 0.05, f"没有尝试更高的高度：{world.tried}"
    assert world.tried == sorted(world.tried), f"抬高顺序不是递增的：{world.tried}"


def test_prune_uses_synced_location_not_stale_zero() -> None:
    """回归点：卡住判定要用同步后的真实位置。

    CARLA 客户端在 actor 生成后需要一次 tick 才返回真实位姿，此前
    get_location() 一律是 (0,0,0)。旧代码在这时取"起点"，于是位移恒为
    "真实位置到原点的距离"（几十米），卡住的行人也永远检测不出来。
    """
    from dvs_collector.traffic import _prune_stuck_walkers

    class _SyncWorld:
        def __init__(self, walker):
            self.walker = walker
            self.ticks = 0

        def tick(self):
            self.ticks += 1

    class _LateSyncWalker:
        """tick 之前 get_location() 返回 (0,0,0)，之后才是真实位置。"""

        def __init__(self, world):
            self.id = 1
            self._world = world
            self.destroyed = False

        def get_location(self):
            if self._world.ticks == 0:
                return _Loc(0.0, 0.0)
            return _Loc(50.0, 30.0)

        def destroy(self):
            self.destroyed = True

        def apply_control(self, control):
            pass

    world = _SyncWorld(None)
    walker = _LateSyncWalker(world)
    world.walker = walker
    spawned = [walker]
    plans = [
        {
            "actor": walker,
            "start": _Loc(50, 30),
            "goal": _Loc(50, 60),
            "speed": 1.0,
            "to_goal": True,
        }
    ]
    alive, removed = _prune_stuck_walkers(world, _FakeCarla(), plans, spawned)
    # 行人一动不动（位置始终 50,30），应被判为卡住并销毁。
    assert removed == 1, f"卡住行人没有被销毁：removed={removed}"
    assert alive == []
    assert walker.destroyed


def test_update_walkers_skips_unsynced_location() -> None:
    """位姿未同步时不下发控制，否则方向会指向原点。"""
    walker = _FakeWalker(1, [_Loc(0.0, 0.0)])
    plan = {
        "actor": walker,
        "start": _Loc(0.0, 0.0),
        "goal": _Loc(10.0, 0.0),
        "speed": 1.2,
        "to_goal": True,
    }
    update_walkers(_FakeCarla(), [plan])
    assert len(walker.controls) == 0, "位姿未同步时不应下发控制"


def main() -> int:
    tests = [
        test_spawn_walker_returns_plan,
        test_spawn_walker_jitter_spreads_pedestrians,
        test_update_walkers_pushes_toward_goal,
        test_update_walkers_flips_at_goal,
        test_update_walkers_does_not_reapply_same_direction,
        test_crosswalk_pairs_use_long_edges,
        test_crosswalk_pairs_skip_pairs_too_close_to_camera,
        test_crosswalk_polygons_recluster_when_no_polygon_markers,
        test_crosswalk_fallback_picks_far_from_camera,
        test_roadside_pairs_cover_sparse_crosswalk_maps,
        test_roadside_pairs_skip_points_too_close_to_camera,
        test_spawn_walker_lifts_ground_level_on_collision,
        test_prune_uses_synced_location_not_stale_zero,
        test_update_walkers_skips_unsynced_location,
        test_prune_destroys_stuck_pair,
        test_prune_keeps_normal_pedestrian,
    ]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print("行人生成/行走/清理自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
