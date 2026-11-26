#!/usr/bin/env python3
"""逐项测试 CARLA 地图/导航操作，把原生崩溃隔离在子进程里。

背景：town03_intersection_a 的采集在「生成行人」这一步以 exit 0xC00000FD
（原生栈溢出）崩掉，而同一张图上的「生成车辆」正常。栈溢出是进程级的
硬崩，try/except 拦不住，只能把每个可疑调用放进独立子进程，用退出码判断。

    python tools/probe_map_ops.py --port 2000 --map Town03

退出码 3221225725 = 0xC00000FD = 栈溢出；0 = 该操作正常。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 每个用例是一段独立运行的代码；子进程崩溃不影响主进程。
CASES: list[tuple[str, str]] = [
    (
        "get_spawn_points（车辆出生点，已知可用）",
        """
w = c.get_world()
points = w.get_map().get_spawn_points()
print('spawn points:', len(points))
""",
    ),
    (
        "get_waypoint（单点查路，已知可用）",
        """
w = c.get_world()
wp = w.get_map().get_waypoint(carla.Location(x=88.2, y=-12.5, z=0.5))
print('waypoint:', wp.transform.location, 'lane', wp.lane_type)
""",
    ),
    (
        "generate_waypoints(2.0)（人行道 waypoint 来源）",
        """
w = c.get_world()
wps = w.get_map().generate_waypoints(2.0)
print('waypoints:', len(wps))
sidewalk = [x for x in wps if x.lane_type & carla.LaneType.Sidewalk]
print('sidewalk waypoints:', len(sidewalk))
""",
    ),
    (
        "get_crosswalks（人行横道）",
        """
import math
import sys
sys.path.insert(0, '.')
from dvs_collector.traffic import _crosswalk_polygons
w = c.get_world()
cw = w.get_map().get_crosswalks()
print('crosswalk points:', len(cw))
polys = _crosswalk_polygons(w)
print('polygons:', len(polys))
for poly in polys[:3]:
    corners = [(round(p.x, 1), round(p.y, 1), round(p.z, 2)) for p in poly]
    edges = [
        round(math.dist((poly[i].x, poly[i].y), (poly[(i + 1) % len(poly)].x, poly[(i + 1) % len(poly)].y)), 2)
        for i in range(len(poly))
    ]
    print('vertices:', len(poly), corners)
    print('edge lengths:', edges)
""",
    ),
    (
        "横道顶点能否生成行人（站位可用性 + 顶点高度）",
        """
import sys
sys.path.insert(0, '.')
from dvs_collector.traffic import _crosswalk_polygons
w = c.get_world()
lib = w.get_blueprint_library()
bp = lib.filter('walker.pedestrian.*')[0]
polys = _crosswalk_polygons(w)
ok = 0
fail = 0
for poly in polys[:10]:
    for p in poly:
        actor = w.try_spawn_actor(bp, carla.Transform(p))
        if actor is None:
            fail += 1
            print('spawn failed at x=%.1f y=%.1f z=%.2f'
                  % (p.x, p.y, p.z))
        else:
            ok += 1
            actor.destroy()
print('vertices tried:', ok + fail, 'ok:', ok, 'failed:', fail)
""",
    ),
    (
        "DVS 传感器是否每帧都发数据（画面静止时会不会停）",
        """
import time
w = c.get_world()
lib = w.get_blueprint_library()
bp = lib.find('sensor.camera.dvs')
bp.set_attribute('image_size_x', '640')
bp.set_attribute('image_size_y', '360')
bp.set_attribute('exposure_mode', 'manual')   # 与生产配置一致
seen = []
sensor = w.spawn_actor(
    bp,
    carla.Transform(
        carla.Location(x=88.2, y=-12.5, z=6.0),
        carla.Rotation(pitch=-25, yaw=180),
    ),
)
sensor.listen(lambda data: seen.append((data.frame, len(data.raw_data))))
for _ in range(60):
    w.tick()
    time.sleep(0.02)
print('ticks: 60, dvs frames received:', len(seen))
print('first three:', seen[:3])
print('last three:', seen[-3:])
if len(seen) < 60:
    print('结论：DVS 不是每帧都发数据，画面静止时它就停了')
    print('→ 采集端不能死等某个 frame 号，必须把 DVS 当事件流处理')
else:
    print('结论：DVS 每帧都发数据，采集端另有原因')
sensor.destroy()
"""
    ),
    (
        "横道相对机位的分布（机位到底在不在路口上）",
        """
import math
import sys
sys.path.insert(0, '.')
from dvs_collector.traffic import _crosswalk_polygons

CAM = (88.2, -12.5)   # 机位（= 路口中心）
w = c.get_world()
polys = _crosswalk_polygons(w)
rows = []
for poly in polys:
    edges = [
        (poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))
    ]
    lengths = [
        math.dist((a.x, a.y), (b.x, b.y)) for a, b in edges
    ]
    longest = max(lengths)
    mid = (
        sum(p.x for p in poly) / len(poly),
        sum(p.y for p in poly) / len(poly),
    )
    dist = math.dist(mid, CAM)
    rows.append((dist, longest, sorted(lengths), len(poly)))
rows.sort()
print('横道数:', len(rows))
for dist, longest, lengths, n in rows[:12]:
    print('  距机位 %6.1f m  边长 %s' % (dist, lengths))
print('  ...')
for dist, longest, lengths, n in rows[-3:]:
    print('  距机位 %6.1f m  边长 %s' % (dist, lengths))
for limit in (20, 40, 60, 100, 200):
    near = [r for r in rows if r[0] <= limit]
    print('距机位 <= %3d m 的横道: %d 条' % (limit, len(near)))
usable = [r for r in rows if r[0] <= 60 and r[1] <= 20]
print('同时满足 距机位<=60m 且 长边<=20m 的横道: %d 条' % len(usable))
usable2 = [r for r in rows if r[0] <= 60 and r[1] <= 35]
print('同时满足 距机位<=60m 且 长边<=35m 的横道: %d 条' % len(usable2))
""",
    ),
    (
        "路口周边的可行机位（把杆子从路口正中挪到路边）",
        """
import math
import sys
sys.path.insert(0, '.')
from dvs_collector.traffic import _crosswalk_polygons

CENTER = (88.2, -12.5)   # 路口中心
HEIGHT = 6.0            # 相机高度


def lane_name(lane):
    names = []
    for attr in ('Driving', 'Sidewalk', 'Biking', 'Shoulder', 'Parking',
                 'OffRoad', 'Restricted'):
        value = getattr(carla.LaneType, attr, None)
        if value is not None and (lane & value):
            names.append(attr)
    return '|'.join(names) or str(lane)


w = c.get_world()
m = w.get_map()
polys = _crosswalk_polygons(w)
info = []
for poly in polys:
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    lengths = [math.dist((a.x, a.y), (b.x, b.y)) for a, b in edges]
    mid = (sum(p.x for p in poly) / len(poly), sum(p.y for p in poly) / len(poly))
    info.append((mid, max(lengths), sorted(lengths)))

print('--- 离路口最近的 8 条横道（中心, 离路口中心, 边长）---')
for mid, longest, lengths in sorted(info, key=lambda r: math.dist(r[0], CENTER))[:8]:
    print('  (%7.1f, %7.1f)  离路口中心 %5.1f m  边长 %s'
          % (mid[0], mid[1], math.dist(mid, CENTER), lengths))

print()
print('--- yaw 约定：前向向量 ---')
for deg in (0, 90, 180, 270):
    v = carla.Rotation(yaw=deg).get_forward_vector()
    print('  yaw=%3d -> (%.2f, %.2f)' % (deg, v.x, v.y))

print()
print('--- 候选机位：路口中心外 14/17.5/22/28 m 的一圈 ---')
print('    （参照 Town10HD_j719：d=17.6 m, h=4.05 m, pitch=-13.0°）')
for radius in (14.0, 17.5, 22.0, 28.0):
    for deg in range(0, 360, 15):
        rad = math.radians(deg)
        x = CENTER[0] + radius * math.cos(rad)
        y = CENTER[1] + radius * math.sin(rad)
        wp = m.get_waypoint(carla.Location(x=x, y=y, z=0.5))
        near = [r for r in info
                if 12.0 <= math.dist(r[0], (x, y)) <= 45.0 and r[1] <= 20.0]
        yaw = math.degrees(math.atan2(CENTER[1] - y, CENTER[0] - x))
        pitch = -math.degrees(math.atan(HEIGHT / radius))
        print('  r=%4.1f 角=%3d (%7.1f, %7.1f) z=%5.2f lane=%-12s 横道 %d 条  '
              '建议 yaw=%7.1f pitch=%6.1f'
              % (radius, deg, x, y, wp.transform.location.z,
                 lane_name(wp.lane_type), len(near), yaw, pitch))
""",
    ),
    (
        "最近横道的顶点与推荐机位（站在路边、正对过街中段）",
        """
import math
import sys
sys.path.insert(0, '.')
from dvs_collector.traffic import _crosswalk_polygons

CENTER = (88.2, -12.5)
HEIGHT = 6.0


def lane_name(lane):
    names = []
    for attr in ('Driving', 'Sidewalk', 'Biking', 'Shoulder', 'Parking',
                 'OffRoad', 'Restricted'):
        value = getattr(carla.LaneType, attr, None)
        if value is not None and (lane & value):
            names.append(attr)
    return '|'.join(names) or str(lane)


w = c.get_world()
m = w.get_map()
polys = _crosswalk_polygons(w)
rows = []
for poly in polys:
    mid = (sum(p.x for p in poly) / len(poly), sum(p.y for p in poly) / len(poly))
    rows.append((math.dist(mid, CENTER), mid, poly))
rows.sort(key=lambda r: r[0])

for dist, mid, poly in rows[:2]:
    print('横道 过街中点=(%.1f, %.1f) 距路口中心 %.1f m' % (mid[0], mid[1], dist))
    for p in poly:
        print('   顶点 (%8.2f, %8.2f, %.2f)' % (p.x, p.y, p.z))
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    lengths = [math.dist((a.x, a.y), (b.x, b.y)) for a, b in edges]
    print('   边长 %s' % [round(v, 2) for v in lengths])
    order = sorted(range(len(edges)), key=lambda i: lengths[i])
    # 最短边是人行道侧那一头；它的中点与过街中点的连线 = 路的走向（法线）。
    short = edges[order[0]]
    side = ((short[0].x + short[1].x) / 2.0, (short[0].y + short[1].y) / 2.0)
    nx, ny = side[0] - mid[0], side[1] - mid[1]
    norm = math.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    print('   短边中点=(%.1f, %.1f)  路走向单位向量=(%.3f, %.3f)'
          % (side[0], side[1], nx, ny))
    for back in (24.0, 28.0, 32.0):
        for sign in (1.0, -1.0):
            x = mid[0] + sign * nx * back
            y = mid[1] + sign * ny * back
            wp = m.get_waypoint(carla.Location(x=x, y=y, z=0.5))
            visible = sum(
                1 for r in rows
                if 12.0 <= math.dist(r[1], (x, y)) <= 45.0
                and max(math.dist((a.x, a.y), (b.x, b.y))
                        for a, b in [(r[2][i], r[2][(i + 1) % len(r[2])])
                                     for i in range(len(r[2]))]) <= 21.0
            )
            yaw = math.degrees(math.atan2(mid[1] - y, mid[0] - x))
            pitch = -math.degrees(math.atan(HEIGHT / back))
            print('   退 %4.1f m 侧%+.0f -> (%8.2f, %8.2f) z=%5.2f lane=%-10s '
                  '视野内横道 %d 条  yaw=%7.1f pitch=%6.1f'
                  % (back, sign, x, y, wp.transform.location.z,
                     lane_name(wp.lane_type), visible, yaw, pitch))
    print()
""",
    ),
    (
        "get_random_location_from_navigation（原崩溃疑点）",
        """
w = c.get_world()
loc = w.get_random_location_from_navigation()
print('random nav location:', loc)
""",
    ),
    (
        "生成 walker + controller.ai.walker + go_to_location",
        """
w = c.get_world()
lib = w.get_blueprint_library()
wps = w.get_map().generate_waypoints(2.0)
sidewalk = [x for x in wps if x.lane_type & carla.LaneType.Sidewalk]
print('sidewalk waypoints:', len(sidewalk))
if not sidewalk:
    print('no sidewalk, skip')
    raise SystemExit(0)
start = sidewalk[0].transform.location
dest = sidewalk[len(sidewalk) // 2].transform.location
walker = w.try_spawn_actor(
    lib.filter('walker.pedestrian.*')[0], carla.Transform(start)
)
print('walker spawned:', walker is not None)
ctrl = w.spawn_actor(
    lib.find('controller.ai.walker'), carla.Transform(), walker
)
print('controller spawned:', ctrl is not None)
ctrl.start()
print('controller started')
ctrl.go_to_location(dest)
print('go_to_location returned')
ctrl.set_max_speed(1.2)
for _ in range(20):
    w.tick()
print('walker moved to:', walker.get_location())
""",
    ),
    (
        "生成 walker + WalkerControl 手动行走（当前方案，不碰导航网格）",
        """
import time
w = c.get_world()
lib = w.get_blueprint_library()
bp = lib.filter('walker.pedestrian.*')[0]
if bp.has_attribute('speed'):
    bp.set_attribute('speed', '2.0')
cw = w.get_map().get_crosswalks()
print('crosswalk points:', len(cw))
start = carla.Location(x=88.2, y=-12.5, z=0.5)
if cw:
    start = cw[0]
walker = w.try_spawn_actor(bp, carla.Transform(start))
print('walker spawned:', walker is not None)
if walker is None:
    raise SystemExit(0)
first = walker.get_location()
# 每 tick 都下发控制 vs 只下发一次，对比服务端每帧的耗时。
began = time.perf_counter()
for _ in range(60):
    walker.apply_control(
        carla.WalkerControl(direction=carla.Vector3D(1.0, 0.0, 0.0), speed=1.2)
    )
    w.tick()
every_tick = time.perf_counter() - began
last = walker.get_location()
moved = ((last.x - first.x) ** 2 + (last.y - first.y) ** 2) ** 0.5
print('walker moved: %.2f m' % moved)
print('60 ticks, control every tick: %.2f s (%.3f s/tick)'
      % (every_tick, every_tick / 60.0))
began = time.perf_counter()
for _ in range(60):
    w.tick()
only_once = time.perf_counter() - began
print('60 ticks, control applied once: %.2f s (%.3f s/tick)'
      % (only_once, only_once / 60.0))
walker.destroy()
""",
    ),
]

PREAMBLE = """
import carla
c = carla.Client('127.0.0.1', {port})
c.set_timeout(30.0)
w = c.get_world()
_s = w.get_settings()
_s.synchronous_mode = True
_s.fixed_delta_seconds = 0.05
w.apply_settings(_s)
"""


def run_case(body: str, port: int) -> tuple[str, str]:
    """在独立子进程里跑一个用例，返回 (状态, 输出)。"""
    script = PREAMBLE.format(port=port) + body
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    output = (process.stdout + process.stderr).strip()
    if process.returncode == 0:
        return "正常", output
    if process.returncode == 3221225725:
        return "栈溢出", output or "(无输出，进程被原生栈溢出打死)"
    return f"失败({process.returncode})", output


def sweep(client: Any) -> int:
    """清掉探测过程生成的行人和控制器，别污染后续采集。"""
    import carla

    world = client.get_world()
    doomed = [
        actor
        for actor in world.get_actors()
        if actor.type_id.startswith("walker.pedestrian.")
        or actor.type_id.startswith("controller.ai.walker")
    ]
    if not doomed:
        return 0
    client.apply_batch_sync(
        [carla.command.DestroyActor(actor.id) for actor in doomed], do_tick=True
    )
    return len(doomed)


def main() -> int:
    parser = argparse.ArgumentParser(description="逐项测试 CARLA 地图操作")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default="", help="先切换到这张地图再测")
    parser.add_argument(
        "--only",
        default="",
        help="只跑名字里含这个片段的用例（默认全跑）",
    )
    args = parser.parse_args()

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    # 上一次采集崩在同步模式里没还原，先切回异步，否则换图或探测都会卡住。
    try:
        settings = client.get_world().get_settings()
        if settings.synchronous_mode:
            settings.synchronous_mode = False
            client.get_world().apply_settings(settings)
            print("提示：已把服务端切回异步模式")
    except RuntimeError as exc:
        print(f"提示：切换异步模式失败（{exc}），继续")
    if args.map:
        print(f"加载地图 {args.map} …")
        world = client.load_world(args.map)
        print(f"当前地图：{world.get_map().name}")
    # 换图后 world 是新对象，原始设置要在换图之后再取。
    original_settings = client.get_world().get_settings()

    crashed = 0
    cases = CASES
    if args.only:
        cases = [item for item in CASES if args.only in item[0]]
        if not cases:
            print(f"没有名字里含 {args.only!r} 的用例")
            return 2
    for name, body in cases:
        print(f"\n=== {name} ===", flush=True)
        status, output = run_case(body, args.port)
        if output:
            print(output)
        print(f"--> {status}", flush=True)
        if status != "正常":
            crashed += 1

    removed = sweep(client)
    if removed:
        print(f"\n已清理 {removed} 个探测残留的行人/控制器")
    # 探测时把 world 切成了同步模式，退出前还原，别影响后续采集。
    client.get_world().apply_settings(original_settings)
    print(f"\n结论：{len(cases) - crashed}/{len(cases)} 项正常")
    return 0 if crashed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
